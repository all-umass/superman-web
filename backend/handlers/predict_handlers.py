from __future__ import absolute_import
import logging
import numpy as np
import os
from tornado import gen
from tornado.escape import json_encode

from .model_handlers import GenericModelHandler, async_crossval, axes_grid
from ..models import REGRESSION_MODELS


class RegressionModelHandler(GenericModelHandler):
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
    res = self.validate_inputs()
    if res is None:
      return
    fig_data, all_ds_views, ds_kind, wave, X = res

    variables = self.collect_variables(all_ds_views,
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
        vals, _ = self.collect_one_variable(all_ds_views, stratify_meta)
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
      yield gen.Task(async_crossval, fig_data, model_cls, num_vars, cv_args,
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


class ModelPlottingHandler(GenericModelHandler):
  def post(self):
    res = self.validate_inputs()
    if res is None:
      return
    fig_data, all_ds_views, ds_kind, wave, X = res
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


def _plot_actual_vs_predicted(preds, stats, fig, variables):
  fig.clf(keep_observers=True)
  axes = axes_grid(fig, len(preds), 'Actual', 'Predicted')
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


routes = [
    (r'/_run_regression', RegressionModelHandler),
    (r'/([0-9]+)/regression_predictions\.csv', RegressionModelHandler),
    (r'/_plot_model_coefs', ModelPlottingHandler),
]
