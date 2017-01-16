from __future__ import absolute_import
import logging
import numpy as np
import os
import shutil
from io import BytesIO
from tempfile import mkstemp
from threading import Thread
from tornado import gen
from tornado.escape import json_encode

from .base import BaseHandler, MultiDatasetHandler
from ..models import REGRESSION_MODELS, RegressionModel


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
    all_bands, all_coefs = pred_model.coefficients()
    var_names = pred_model.var_names
    share_bands = isinstance(pred_model,
                             tuple(REGRESSION_MODELS['pls'].values()))

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition', 'attachment; filename='+fname)
    # write header comment with model info
    self.write('# %s - %s\n' % (pred_model, pred_model.ds_kind))
    # write model coefficients as CSV
    if share_bands:
      wave = next(all_bands)
      self.write('wavelength,%s\n' % ','.join('%g' % x for x in wave))
      for name, coefs in zip(var_names, all_coefs):
        self.write('%s,%s\n' % (name, ','.join('%g' % x for x in coefs)))
    else:
      for name, wave, coefs in zip(var_names, all_bands, all_coefs):
        self.write('wavelength,%s\n' % ','.join('%g' % x for x in wave))
        self.write('%s,%s\n' % (name, ','.join('%g' % x for x in coefs)))
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
      model = RegressionModel.load(fh)
    except Exception as e:
      logging.error('Failed to parse uploaded model file: %s', e.message)
      self.set_status(415)
      return self.finish('Invalid model file.')

    # do some validation
    if not isinstance(model, RegressionModel):
      logging.error('Uploaded model file out of date: %r', model)
      self.set_status(415)
      return self.finish('Invalid model file.')

    ds_kind = self.get_argument('ds_kind')
    if model.ds_kind != ds_kind:
      logging.warning('Mismatching model kind. Expected %r, got %r', ds_kind,
                      model.ds_kind)

    # stash the loaded model
    fig_data.pred_model = model
    return self.write(json_encode(dict(info=model.info_html())))


class MultiVectorDatasetHandler(MultiDatasetHandler):
  def collect_spectra(self, all_ds_views):
    '''collect vector-format data from all datasets'''
    ds_kind, wave, X = None, None, []
    for dv in all_ds_views:
      try:
        w, x = dv.get_vector_data()
      except ValueError as e:
        self.visible_error(400, e.message,
                           "Couldn't get vector data from %s: %s", dv.ds,
                           e.message)
        return ds_kind, wave, None
      if wave is None:
        wave = w
        ds_kind = dv.ds.kind
      else:
        if wave.shape != w.shape or not np.allclose(wave, w):
          self.visible_error(400, "Mismatching wavelength data in %s." % dv.ds)
          return ds_kind, wave, None
        if ds_kind != dv.ds.kind:
          self.visible_error(400, "Mismatching dataset types.",
                             "Mismatching ds_kind: %s not in %s",
                             dv.ds, ds_kind)
          return ds_kind, wave, None
      X.append(x)
    return ds_kind, wave, np.vstack(X)


class RegressionModelHandler(MultiVectorDatasetHandler):
  def get(self, fignum):
    '''Download predictions as CSV.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return
    if fig_data.last_plot != 'regression_preds':
      self.write('No plotted data to download.')
      return

    all_ds = self.request_many_ds()
    if not all_ds:
      self.write('No datasets selected.')
      return

    # collect primary keys for row labels
    all_pkeys = []
    for ds in all_ds:
      dv = ds.view(mask=fig_data.filter_mask[ds])
      all_pkeys.extend(dv.get_primary_keys())

    # get data from the scatterplots
    names, actuals, preds = [], [], []
    for ax in fig_data.figure.axes:
      if not ax.collections:
        break
      names.append(ax.get_title())
      scat = ax.collections[0]
      actual, pred = scat.get_offsets().T
      preds.append(pred)
      # HACK: if there are 6 lines on the plot, it's a boxplot, and thus
      # there are no actual values to report. Instead, they're random jitter.
      if len(ax.lines) == 6:
        actual.fill(np.nan)
      actuals.append(actual)

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
      for i, key in enumerate(all_pkeys):
        row[::2] = actuals[i]
        row[1::2] = preds[i]
        self.write('%s,' % key + ','.join('%g' % x for x in row) + '\n')
    self.finish()

  @gen.coroutine
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.visible_error(403, "Broken connection to server.")
      return

    all_ds_views, _ = self.prepare_ds_views(fig_data, nan_gap=None)
    if all_ds_views is None:
      self.visible_error(404, "Failed to look up dataset(s).")
      return

    ds_kind, wave, X = self.collect_spectra(all_ds_views)
    if X is None:
      # self.visible_error has already been called in collect_spectra
      return

    variables = _collect_variables(all_ds_views,
                                   self.get_arguments('pred_meta[]'))
    regress_kind = self.get_argument('regress_kind')
    variate_kind = self.get_argument('variate_kind')
    model_cls = REGRESSION_MODELS[regress_kind][variate_kind]
    params = dict(pls=int(self.get_argument('pls_comps')),
                  lasso=float(self.get_argument('lasso_alpha')),
                  lars=int(self.get_argument('lars_num_channels')))

    do_train = self.get_argument('do_train', None)
    if do_train is None:
      if len(variables) == 0:
        self.visible_error(400, "No variables to predict.")
        return

      no_crossval = (len(variables) > 1 and variate_kind == 'multi'
                     and regress_kind == 'lasso')
      if no_crossval:
        msg = "Cross validation for %svariate %s is not yet supported." % (
            variate_kind, regress_kind.title())
        self.visible_error(400, msg)
        return

      # set up cross validation info
      folds = int(self.get_argument('cv_folds'))
      stratify_meta = self.get_argument('cv_stratify', '')
      if stratify_meta:
        tmp = _collect_variables(all_ds_views, (stratify_meta,))
        vals, _ = tmp[stratify_meta]
        _, stratify_labels = np.unique(vals, return_inverse=True)
      else:
        stratify_labels = None
      num_vars = 1 if variate_kind == 'multi' else len(variables)
      cv_args = (X, variables)
      cv_kwargs = dict(num_folds=folds, labels=stratify_labels)
      logging.info('Running %d-fold (%s) cross-val for %s', folds,
                   stratify_meta, model_cls.__name__)

      if regress_kind == 'pls':
        comps = np.arange(int(self.get_argument('cv_min_comps')),
                          int(self.get_argument('cv_max_comps')) + 1)
        cv_kwargs['comps'] = comps
        plot_kwargs = dict(xlabel='# components')
      elif regress_kind == 'lasso':
        plot_kwargs = dict(xlabel='alpha', logx=True)
      else:
        chans = np.arange(int(self.get_argument('cv_min_chans')),
                          int(self.get_argument('cv_max_chans')) + 1)
        cv_kwargs['chans'] = chans
        plot_kwargs = dict(xlabel='# channels')

      # run the cross validation
      yield gen.Task(_async_crossval, fig_data, model_cls, num_vars, cv_args,
                     cv_kwargs, **plot_kwargs)
      return

    if bool(int(do_train)):
      # train on all the data
      model = model_cls(params[regress_kind], ds_kind, wave)
      logging.info('Training %s on %d inputs, predicting %d vars',
                   model, X.shape[0], len(variables))
      model.train(X, variables)
      fig_data.pred_model = model
    else:
      # use existing model
      model = fig_data.pred_model
      if model.ds_kind != ds_kind:
        logging.warning('Mismatching model kind. Expected %r, got %r', ds_kind,
                        model.ds_kind)
      # use the model's variables, with None instead of actual values
      dummy_vars = {key: (None, name) for key, name in
                    zip(model.var_keys, model.var_names)}
      # use the actual variables if we have them
      for key in model.var_keys:
        if key in variables:
          dummy_vars[key] = variables[key]
      variables = dummy_vars
      # make sure we're using the same wavelengths
      if wave.shape != model.wave.shape or not np.allclose(wave, model.wave):
        if wave[-1] <= model.wave[0] or wave[0] >= model.wave[-1]:
          self.visible_error(400, "Data to predict doesn't overlap "
                                  "with training wavelengths.")
          return
        Xnew = np.empty((X.shape[0], model.wave.shape[0]), dtype=X.dtype)
        for i, y in enumerate(X):
          Xnew[i] = np.interp(model.wave, wave, y)
        X = Xnew

    # get predictions for each variable
    preds, stats = model.predict(X, variables)

    # plot
    _plot_actual_vs_predicted(preds, stats, fig_data.figure, variables)
    fig_data.manager.canvas.draw()
    fig_data.last_plot = 'regression_preds'

    res = dict(stats=stats, info=fig_data.pred_model.info_html())
    # NaN isn't valid JSON, but json_encode doesn't catch it. :'(
    self.write(json_encode(res).replace('NaN', 'null'))


class ModelPlottingHandler(MultiVectorDatasetHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.visible_error(403, "Broken connection to server.")
      return

    all_ds_views, _ = self.prepare_ds_views(fig_data, nan_gap=None)
    if all_ds_views is None:
      self.visible_error(404, "Failed to look up dataset(s).")
      return

    ds_kind, wave, X = self.collect_spectra(all_ds_views)
    if X is None:
      # self.visible_error has already been called in collect_spectra
      return

    model = fig_data.pred_model
    all_bands, all_coefs = model.coefficients()

    # Do the plot
    fig_data.figure.clf(keep_observers=True)
    ax1 = fig_data.figure.gca()
    ax1.plot(wave, X.T, 'k-', alpha=0.5, lw=1)
    ax2 = ax1.twinx()
    ax2.axhline(lw=1, ls='--', color='gray')
    size = 20 * float(self.get_argument('line_width'))
    alpha = float(self.get_argument('alpha'))
    for name, x, y in zip(model.var_names, all_bands, all_coefs):
      ax2.scatter(x, y, label=name, s=size, alpha=alpha)
    if bool(int(self.get_argument('legend'))) and len(model.var_names) > 1:
      ax2.legend()
    fig_data.manager.canvas.draw()
    fig_data.last_plot = 'regression_coefs'


def _collect_variables(all_ds_views, meta_keys):
  '''Collect metadata variables to predict from all loaded datasets.
  Returns a dict of {key: (array, display_name)}
  '''
  variables = {}
  for key in meta_keys:
    yy, name = [], None
    for dv in all_ds_views:
      y, name = dv.get_metadata(key)
      yy.append(y)
    variables[key] = (np.concatenate(yy), name)
  return variables


def _async_crossval(fig_data, model_cls, num_vars, cv_args, cv_kwargs,
                    xlabel='param', ylabel='MSE', logx=False, callback=None):
  '''Wrap cross-validation calls in a Thread to avoid hanging the server.'''
  def helper():
    fig_data.figure.clf(keep_observers=True)
    axes = _axes_grid(fig_data.figure, num_vars, xlabel, ylabel)
    if logx:
      for ax in axes:
        ax.set_xscale('log')

    cv_gen = model_cls.cross_validate(*cv_args, **cv_kwargs)
    for i, (name, x, y, yerr) in enumerate(cv_gen):
      axes[i].set_title(name)
      axes[i].errorbar(x, y, yerr=yerr, lw=2, fmt='k-', ecolor='r',
                       elinewidth=1, capsize=0)

    fig_data.manager.canvas.draw()
    fig_data.last_plot = '%s_crossval' % model_cls.__name__
    callback()

  t = Thread(target=helper)
  t.daemon = True
  t.start()


def _plot_actual_vs_predicted(preds, stats, fig, variables):
  fig.clf(keep_observers=True)
  axes = _axes_grid(fig, len(preds), 'Actual', 'Predicted')
  for i, key in enumerate(sorted(preds)):
    ax = axes[i]
    y, name = variables[key]
    p = preds[key].ravel()
    ax.set_title(name)
    # validate y
    if y is not None:
      mask = np.isfinite(y)
      nnz = np.count_nonzero(mask)
      if nnz == 0:
        y = None
      elif nnz < len(y):
        y, p = y[mask], p[mask]
    if y is not None:
      # actual values exist, so plot them
      ax.scatter(y, p)
      xlims = ax.get_xlim()
      ylims = ax.get_ylim()
      ax.errorbar(y, p, yerr=stats[i]['rmse'], fmt='none', ecolor='k',
                  elinewidth=1, capsize=0, alpha=0.5, zorder=0)
      # plot best fit line
      xylims = [np.min([xlims, ylims]), np.max([xlims, ylims])]
      best_fit = np.poly1d(np.polyfit(y, p, 1))(xylims)
      ax.plot(xylims, best_fit, 'k--', alpha=0.75, zorder=-1)
      ax.set_aspect('equal')
      ax.set_xlim(xylims)
      ax.set_ylim(xylims)
    else:
      # no actual values exist, so only plot the predictions
      ax.boxplot(p, showfliers=False)
      # overlay jitter plot
      x = np.ones_like(p) + np.random.normal(scale=0.025, size=len(p))
      ax.scatter(x, p, alpha=0.9)


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

routes = [
    (r'/_run_model', RegressionModelHandler),
    (r'/([0-9]+)/regression_predictions\.csv', RegressionModelHandler),
    (r'/_load_model', ModelIOHandler),
    (r'/([0-9]+)/regression_model\.(\w+)', ModelIOHandler),
    (r'/_plot_model_coefs', ModelPlottingHandler),
]
