from __future__ import absolute_import
import logging
import numpy as np
import os
import shutil
from io import BytesIO
from sklearn.cross_decomposition import PLSRegression
from sklearn.externals.joblib.numpy_pickle import (
    ZipNumpyUnpickler, dump as dump_pickle)
from sklearn.metrics import r2_score, mean_squared_error
from tempfile import mkstemp
from tornado.escape import json_encode

from .base import BaseHandler
from .baseline_handlers import setup_blr_object
from .dataset_handlers import DatasetHandler
from ..web_datasets import NumericMetadata, CompositionMetadata


class ModelIOHandler(BaseHandler):
  def get(self, fignum):
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    _, tmp_path = mkstemp()
    fig_data.pred_model.save(tmp_path)

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    with open(tmp_path, 'rb') as fh:
      shutil.copyfileobj(fh, self)
    self.finish()
    os.remove(tmp_path)

  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return
    if not self.request.files:
      logging.error('LoadModelHandler: no file uploaded')
      self.set_status(403)
      return

    f = self.request.files['modelfile'][0]
    fname = f['filename']
    logging.info('Loading model file: %s', fname)
    fh = BytesIO(f['body'])
    try:
      model = _PLS.load(fh)
    except Exception as e:
      logging.error('Failed to parse uploaded model file: %s', e.message)
      self.set_status(415)
      return

    # do some validation
    ds_kind = self.get_argument('ds_kind')
    if model.ds_kind != ds_kind:
      logging.warning('Mismatching model kind. Expected %r, got %r', ds_kind,
                      model.ds_kind)

    # stash the loaded model
    fig_data.pred_model = model
    return self.write(json_encode(dict(info=model.info())))


class RegressionModelHandler(BaseHandler):
  def get(self, fignum):
    '''Download predictions as CSV.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    names, actuals, preds = [], [], []
    for ax in fig_data.figure.axes:
      if not ax.collections:
        break
      names.append(ax.get_title())
      scat, = ax.collections
      actual, pred = scat.get_offsets().T
      actuals.append(actual)
      preds.append(pred)

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    # first header line: foo,,bar,,baz,
    self.write(','.join(n+',' for n in names) + '\n')
    # secondary header: Actual,Pred,Actual,Pred,Actual,Pred
    self.write(','.join(['Actual,Pred']*len(names)) + '\n')
    row = np.empty((len(names) * 2,), dtype=float)

    if actuals and preds:
      actuals = np.column_stack(actuals)
      preds = np.column_stack(preds)
      for arow, prow in zip(actuals, preds):
        row[::2] = arow
        row[1::2] = prow
        self.write(','.join('%g' % x for x in row) + '\n')
    self.finish()

  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return
    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    ds = self.get_dataset(ds_kind, ds_name)
    if ds is None:
      logging.error("Failed to look up dataset: %s [%s]" % (ds_name, ds_kind))
      self.set_status(404)
      return

    mask = fig_data.filter_mask[ds]
    bl_obj, segmented, inverted, lb, ub, params = setup_blr_object(self)
    chan_mask = bool(int(self.get_argument('chan_mask', 0)))
    trans = dict(pp=self.get_argument('pp', ''), blr_obj=bl_obj,
                 blr_segmented=segmented, blr_inverted=inverted,
                 crop=(lb, ub), chan_mask=chan_mask, nan_gap=None)
    ds_view = ds.view(mask=mask, **trans)

    variables = {key: ds_view.get_metadata(key)
                 for key in self.get_arguments('pred_meta[]')}
    X = ds_view.get_data()

    if bool(int(self.get_argument('do_train'))):
      pls_comps = int(self.get_argument('pls_comps'))
      pls_kind = self.get_argument('pls_kind')
      logging.info('Training %s(%d) on %d inputs, predicting %d vars',
                   pls_kind, pls_comps, X.shape[0], len(variables))
      if pls_kind == 'pls1':
        _train_fn = PLS1.train
      else:
        _train_fn = PLS2.train
      fig_data.pred_model = _train_fn(X, variables, pls_comps, ds_kind)

    # get predictions for each variable
    preds, stats = fig_data.pred_model.predict(X, variables, ds_kind)

    # plot actual vs predicted
    fig = fig_data.figure
    fig.clf(keep_observers=True)
    n = len(preds)
    if n > 0:
      r = np.floor(np.sqrt(n))
      r, c = int(r), int(np.ceil(n / r))
      for i, key in enumerate(sorted(preds)):
        ax = fig.add_subplot(r, c, i+1)
        y, name = variables[key]
        p = preds[key].ravel()
        ax.scatter(y, p)
        ax.set_title(name)
        # plot best fit line
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])]
        best_fit = np.poly1d(np.polyfit(y, p, 1))(lims)
        ax.plot(lims, best_fit, 'k--', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        if i % c == 0:
          ax.set_ylabel('Predicted')
        if i >= c * (r - 1):
          ax.set_xlabel('Actual')
    fig_data.manager.canvas.draw()

    res = dict(stats=stats, info=fig_data.pred_model.info())
    return self.write(json_encode(res))


class GetPredictableMetadataHandler(DatasetHandler):
  def post(self):
    ds = self.get_ds()
    names = ds.metadata_names(allowed_baseclasses=(NumericMetadata,
                                                   CompositionMetadata))
    return self.write(json_encode(list(names)))


class _PLS(object):
  @staticmethod
  def load(fh):
    return ZipNumpyUnpickler('', fh).load()

  def save(self, fname):
    dump_pickle(self, fname, compress=3)

  def predict(self, X, variables, ds_kind):
    if ds_kind != self.ds_kind:
      logging.warning('Mismatching ds_kind in PLS prediction: %r != %r',
                      ds_kind, self.ds_kind)
    preds = {}
    stats = []
    for key, p in self._predict(X, variables):
      y, name = variables[key]
      preds[key] = p
      stats.append(dict(r2=r2_score(y, p),
                        rmse=np.sqrt(mean_squared_error(y, p)),
                        name=name))
    stats.sort(key=lambda s: s['name'])
    return preds, stats

  def info(self):
    return '%s(%s): %s' % (self.__class__.__name__, self.ds_kind,
                           ','.join(sorted(self.var_names)))


class PLS1(_PLS):
  @staticmethod
  def train(X, variables, k, ds_kind):
    res = PLS1()
    res.ds_kind = ds_kind
    res.models = {}
    res.var_names = variables.items()
    for key in variables:
      clf = PLSRegression(scale=False, n_components=k)
      y, _ = variables[key]
      res.models[key] = clf.fit(X, y)
    return res

  def _predict(self, X, variables):
    for key in variables:
      if key not in self.models:
        logging.warning('No trained model for variable: %r', key)
        continue
      clf = self.models[key]
      yield key, clf.predict(X)


class PLS2(_PLS):
  @staticmethod
  def train(X, variables, k, ds_kind):
    res = PLS2()
    res.ds_kind = ds_kind
    res.clf = PLSRegression(scale=False, n_components=k)
    res.var_keys = variables.keys()
    res.var_names = [variables[key][1] for key in res.var_keys]
    Y = np.column_stack([variables[key][0] for key in res.var_keys])
    res.clf.fit(X, Y)
    return res

  def _predict(self, X, variables):
    P = self.clf.predict(X)
    for i, key in enumerate(self.var_keys):
      if key not in variables:
        logging.warning('No input variable for predicted: %r', key)
        continue
      yield key, P[:,i]


routes = [
    (r'/_predictable', GetPredictableMetadataHandler),
    (r'/_run_model', RegressionModelHandler),
    (r'/([0-9]+)/pls_predictions\.csv', RegressionModelHandler),
    (r'/_load_model', ModelIOHandler),
    (r'/([0-9]+)/pls_model\.pkl', ModelIOHandler),
]
