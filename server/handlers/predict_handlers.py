from __future__ import absolute_import
import logging
import numpy as np
import os
import shutil
from io import BytesIO
from sklearn.cross_decomposition import PLSRegression
from sklearn.externals.joblib.numpy_pickle import (
    load as load_pickle, dump as dump_pickle)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from tempfile import mkstemp
from tornado.escape import json_encode

from .base import BaseHandler
from .baseline_handlers import ds_view_kwargs


class ModelIOHandler(BaseHandler):
  def get(self, fignum, ext):
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    model = fig_data.pred_model
    if ext.lower() == 'pkl':
      self._serve_pickle(model)
    else:
      self._serve_csv(model)

  def _serve_pickle(self, pred_model):
    _, tmp_path = mkstemp()
    pred_model.save(tmp_path)

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', 'attachment; filename='+fname)
    with open(tmp_path, 'rb') as fh:
      shutil.copyfileobj(fh, self)
    self.finish()
    os.remove(tmp_path)

  def _serve_csv(self, pred_model):
    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition', 'attachment; filename='+fname)
    # write header comment with model info
    self.write('# %s - %s\n' % (pred_model, pred_model.ds_kind))
    # write model coefficients as CSV
    self.write('wavelength,' + ','.join(pred_model.var_names) + '\n')
    if isinstance(pred_model, PLS2):
      all_coefs = pred_model.clf.coef_
    else:
      all_coefs = np.column_stack([pred_model.models[key].coef_
                                   for key in pred_model.var_keys])
    wave = pred_model.wave
    fmt_str = ','.join(['%g'] * (all_coefs.shape[1] + 1)) + '\n'
    for i, row in enumerate(all_coefs):
      self.write(fmt_str % ((wave[i],) + tuple(row)))
    self.finish()

  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return self.finish('Invalid internal state.')
    if not self.request.files:
      logging.error('LoadModelHandler: no file uploaded')
      self.set_status(403)
      return self.finish('No file uploaded.')

    f = self.request.files['modelfile'][0]
    fname = f['filename']
    logging.info('Loading model file: %s', fname)
    fh = BytesIO(f['body'])
    try:
      model = _PLS.load(fh)
    except Exception as e:
      logging.error('Failed to parse uploaded model file: %s', e.message)
      self.set_status(415)
      return self.finish('Invalid model file.')

    # do some validation
    if not isinstance(model, _PLS):
      logging.error('Uploaded model file out of date: %r', model)
      self.set_status(415)
      return self.finish('Invalid model file.')

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
    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    ds = self.get_dataset(ds_kind, ds_name)
    if ds is None:
      self.write("Couldn't find the requested dataset.")
      return

    mask = fig_data.filter_mask[ds]
    if ds.pkey is None:
      pkey, = np.where(mask)
    else:
      pkey = ds.pkey.keys[mask]

    names, actuals, preds = [], [], []
    for ax in fig_data.figure.axes:
      if not ax.collections:
        break
      names.append(ax.get_title())
      scat = ax.collections[0]
      actual, pred = scat.get_offsets().T
      actuals.append(actual)
      preds.append(pred)

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    # first header line: spectrum,foo,,bar,,baz,
    self.write('Spectrum,' + ',,'.join(names) + ',\n')
    # secondary header: ,Actual,Pred,Actual,Pred,Actual,Pred
    self.write(',' + ','.join(['Actual,Pred']*len(names)) + '\n')

    if actuals and preds:
      actuals = np.column_stack(actuals)
      preds = np.column_stack(preds)
      row = np.empty((len(names) * 2,), dtype=float)
      for i, key in enumerate(pkey):
        row[::2] = actuals[i]
        row[1::2] = preds[i]
        self.write('%s,' % key + ','.join('%g' % x for x in row) + '\n')
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
    ds_view = ds.view(**ds_view_kwargs(self, mask=mask, nan_gap=None))
    variables = {key: ds_view.get_metadata(key)
                 for key in self.get_arguments('pred_meta[]')}

    try:
      wave, X = ds_view.get_vector_data()
    except ValueError as e:
      logging.error("Failed to get vector data: %s", e.message)
      self.set_status(400)
      return

    pls_kind = self.get_argument('pls_kind')
    do_train = self.get_argument('do_train', None)
    if do_train is None:
      # run cross validation
      folds = int(self.get_argument('cv_folds'))
      comps = np.arange(int(self.get_argument('cv_min_comps')),
                        int(self.get_argument('cv_max_comps')) + 1)
      logging.info('Running %d-fold cross-val in range [%d,%d]', folds,
                   comps[0], comps[-1])

      # convert pls2 into pls1 format via hacks
      if pls_kind == 'pls2':
        Y = np.column_stack([y for y, name in variables.values()])
        variables = dict(combined=(Y, None))

      # run the cross-val and plot scores for each n_components
      fig_data.figure.clf(keep_observers=True)
      axes = _axes_grid(fig_data.figure, len(variables),
                        '# components', 'MSE')
      for i, key in enumerate(sorted(variables)):
        pls = GridSearchCV(PLSRegression(scale=False),
                           dict(n_components=comps),
                           cv=folds, scoring='neg_mean_squared_error',
                           return_train_score=False)
        pls.fit(X, variables[key][0])
        axes[i].errorbar(comps, -pls.cv_results_['mean_test_score'],
                         yerr=pls.cv_results_['std_test_score'])
      fig_data.manager.canvas.draw()
      return

    if bool(int(do_train)):
      # train on all the data
      cls = PLS1 if pls_kind == 'pls1' else PLS2
      model = cls(int(self.get_argument('pls_comps')), ds_kind, wave)
      logging.info('Training %s on %d inputs, predicting %d vars',
                   model, X.shape[0], len(variables))
      model.train(X, variables)
      fig_data.pred_model = model
    else:
      model = fig_data.pred_model
      if model.ds_kind != ds_kind:
        logging.warning('Mismatching model kind. Expected %r, got %r', ds_kind,
                        model.ds_kind)

    # get predictions for each variable
    preds, stats = model.predict(X, variables)

    _plot_actual_vs_predicted(preds, stats, fig_data.figure, variables)
    fig_data.manager.canvas.draw()

    res = dict(stats=stats, info=fig_data.pred_model.info_html())
    return self.write(json_encode(res))


def _plot_actual_vs_predicted(preds, stats, fig, variables):
  fig.clf(keep_observers=True)
  axes = _axes_grid(fig, len(preds), 'Actual', 'Predicted')
  for i, key in enumerate(sorted(preds)):
    ax = axes[i]
    y, name = variables[key]
    p = preds[key].ravel()
    ax.scatter(y, p)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.errorbar(y, p, yerr=stats[i]['rmse'], fmt='none', ecolor='k',
                elinewidth=1, capsize=0, alpha=0.5, zorder=0)
    ax.set_title(name)
    # plot best fit line
    xylims = [np.min([xlims, ylims]), np.max([xlims, ylims])]
    best_fit = np.poly1d(np.polyfit(y, p, 1))(xylims)
    ax.plot(xylims, best_fit, 'k--', alpha=0.75, zorder=-1)
    ax.set_aspect('equal')
    ax.set_xlim(xylims)
    ax.set_ylim(xylims)


def _axes_grid(fig, n, xlabel, ylabel):
  r = np.floor(np.sqrt(n))
  r, c = int(r), int(np.ceil(n / r))
  axes = []
  for i in range(n):
    ax = fig.add_subplot(r, c, i+1)
    if i % c == 0:
      ax.set_ylabel(ylabel)
    if i >= c * (r - 1):
      ax.set_xlabel(xlabel)
    axes.append(ax)
  return axes


class _PLS(object):
  def __init__(self, k, ds_kind, wave):
    self.n_components = k
    self.ds_kind = ds_kind
    self.wave = wave
    self.var_names = []
    self.var_keys = []

  @staticmethod
  def load(fh):
    return load_pickle(fh)

  def save(self, fname):
    dump_pickle(self, fname, compress=3)

  def predict(self, X, variables):
    preds, stats = {}, []
    for key, p in self._predict(X, variables):
      y, name = variables[key]
      preds[key] = p
      stats.append(dict(r2=r2_score(y, p),
                        rmse=np.sqrt(mean_squared_error(y, p)),
                        name=name, key=key))
    stats.sort(key=lambda s: s['key'])
    return preds, stats

  def info_html(self):
    return '%s &mdash; %s' % (self, self.ds_kind)

  def __str__(self):
    return '%s(%d)' % (self.__class__.__name__, self.n_components)


class PLS1(_PLS):
  def train(self, X, variables):
    self.models = {}
    for key in variables:
      clf = PLSRegression(scale=False, n_components=self.n_components)
      y, name = variables[key]
      self.models[key] = clf.fit(X, y)
      self.var_keys.append(key)
      self.var_names.append(name)

  def _predict(self, X, variables):
    for key in variables:
      if key not in self.models:
        logging.warning('No trained model for variable: %r', key)
        continue
      clf = self.models[key]
      yield key, clf.predict(X)


class PLS2(_PLS):
  def train(self, X, variables):
    self.clf = PLSRegression(scale=False, n_components=self.n_components)
    self.var_keys = variables.keys()
    y_cols = []
    for key in self.var_keys:
      y, name = variables[key]
      y_cols.append(y)
      self.var_names.append(name)
    self.clf.fit(X, np.column_stack(y_cols))

  def _predict(self, X, variables):
    P = self.clf.predict(X)
    for i, key in enumerate(self.var_keys):
      if key not in variables:
        logging.warning('No input variable for predicted: %r', key)
        continue
      yield key, P[:,i]


routes = [
    (r'/_run_model', RegressionModelHandler),
    (r'/([0-9]+)/pls_predictions\.csv', RegressionModelHandler),
    (r'/_load_model', ModelIOHandler),
    (r'/([0-9]+)/pls_model\.(\w+)', ModelIOHandler),
]
