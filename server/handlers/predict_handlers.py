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
    dump_pickle(fig_data.pred_model, tmp_path, compress=3)

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
      model = ZipNumpyUnpickler('', fh).load()
    except Exception as e:
      logging.error('Failed to parse uploaded model file: %s', e.message)
      self.set_status(415)
      return

    # do some validation
    ds_kind = self.get_argument('ds_kind')
    if model['kind'] != ds_kind:
      logging.error('Mismatching model kind. Expected %r, got %r', ds_kind,
                    model['kind'])
      self.set_status(415)
      return

    # save the loaded model
    fig_data.pred_model = model


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
    actuals = np.column_stack(actuals)
    preds = np.column_stack(preds)

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    # first header line: foo,,bar,,baz,
    self.write(','.join(n+',' for n in names) + '\n')
    # secondary header: Actual,Pred,Actual,Pred,Actual,Pred
    self.write(','.join(['Actual,Pred']*len(names)) + '\n')
    row = np.empty((len(names) * 2,), dtype=float)
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

    Y, pred_names = [], []
    for i, key in enumerate(self.get_arguments('pred_meta[]')):
      y, name = ds_view.get_metadata(key)
      Y.append(y)
      pred_names.append(name)
    Y = np.column_stack(Y)
    X = ds_view.get_data()

    if bool(int(self.get_argument('do_train'))):
      pls_comps = int(self.get_argument('pls_comps'))
      clf = PLSRegression(scale=False, n_components=pls_comps)
      logging.info('Training model %r on %d inputs, predicting %d vars', clf,
                   X.shape[0], Y.shape[1])
      clf.fit(X, Y)
      # stash the model away, in case we want to download / run it later
      fig_data.pred_model = dict(model=clf, trans=trans, kind=ds_kind)
    else:
      clf = fig_data.pred_model['model']
      #TODO: check the ds_kind vs the model's kind, etc?

    # run predictions
    pred_y = clf.predict(X)
    r2 = r2_score(Y, pred_y)
    mse = mean_squared_error(Y, pred_y)

    # plot actual vs predicted
    fig = fig_data.figure
    fig.clf(keep_observers=True)
    n = len(pred_names)
    r = np.floor(np.sqrt(n))
    r, c = int(r), int(np.ceil(n / r))
    axes = fig.subplots(nrows=r, ncols=c, squeeze=False)
    for i, name in enumerate(pred_names):
      ax = axes.flat[i]
      ax.scatter(Y[:,i], pred_y[:,i])
      ax.set_title(name)
    for ax in axes[-1]:
      ax.set_xlabel('Actual')
    for ax in axes[:,0]:
      ax.set_ylabel('Predicted')
    for ax in axes.flat[n:]:
      ax.set_axis_off()
    fig_data.manager.canvas.draw()

    return self.write(json_encode(dict(r2=r2, mse=mse)))


class GetPredictableMetadataHandler(DatasetHandler):
  def post(self):
    ds = self.get_ds()
    names = ds.metadata_names(allowed_baseclasses=(NumericMetadata,
                                                   CompositionMetadata))
    return self.write(json_encode(list(names)))


routes = [
    (r'/_predictable', GetPredictableMetadataHandler),
    (r'/_run_model', RegressionModelHandler),
    (r'/([0-9]+)/pls_predictions\.csv', RegressionModelHandler),
    (r'/_load_model', ModelIOHandler),
    (r'/([0-9]+)/pls_model\.pkl', ModelIOHandler),
]
