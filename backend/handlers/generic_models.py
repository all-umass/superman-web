from __future__ import absolute_import, print_function, division
import logging
import numpy as np
import os
from io import BytesIO
from threading import Thread

from .common import BaseHandler, MultiDatasetHandler
from ..models import GenericModel, REGRESSION_MODELS
from six.moves import range
from six.moves import zip

__all__ = [
    'GenericModelHandler', 'async_crossval', 'axes_grid'
]


class ModelIOHandler(BaseHandler):
  def get(self, fignum, model_type, ext):
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    if model_type == 'regression':
      model = fig_data.pred_model
    else:
      model = fig_data.classify_model

    if ext == 'bin':
      self._serve_binary(model)
    else:
      self._serve_csv(model)

  def _serve_binary(self, model):
    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', 'attachment; filename='+fname)
    model.save(self)  # use self as a file-like object
    self.finish()

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
      return self.visible_error(403, 'Invalid internal state.')

    if not self.request.files:
      return self.visible_error(403, 'No file uploaded.')

    f = self.request.files['modelfile'][0]
    fname = f['filename']
    logging.info('Loading model file: %s', fname)
    fh = BytesIO(f['body'])
    try:
      model = GenericModel.load(fh)
    except Exception:
      logging.exception('Failed to parse model file.')
      return self.visible_error(415, 'Invalid model file.')

    # do some validation
    if not isinstance(model, GenericModel):
      return self.visible_error(415, 'Invalid model file.',
                                'Uploaded model file out of date: %r', model)

    ds_kind = self.get_argument('ds_kind')
    if model.ds_kind != ds_kind:
      logging.warning('Mismatching model kind. Expected %r, got %r', ds_kind,
                      model.ds_kind)

    # stash the loaded model
    if self.get_argument('model_type') == 'regression':
      fig_data.pred_model = model
    else:
      fig_data.classify_model = model
    return self.write_json(dict(info=model.info_html()))


class GenericModelHandler(MultiDatasetHandler):
  def validate_inputs(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.visible_error(403, "Broken connection to server.")
      return

    ds_views = self.prepare_ds_views(fig_data, nan_gap=None)
    if ds_views is None:
      self.visible_error(404, "Invalid dataset selection.")
      return

    ds_kind, wave, X = self.collect_spectra(ds_views)
    if X is None:
      # self.visible_error has already been called in collect_spectra
      return
    return fig_data, ds_views, ds_kind, wave, X

  def collect_trajectories(self, ds_views):
    '''collect traj-format data from all datasets.'''
    trajs = ds_views.get_trajectories(avoid_nan_gap=True)
    return ds_views.ds_kind, trajs

  def collect_spectra(self, ds_views):
    '''collect vector-format data from all datasets.'''
    try:
      wave, X = ds_views.get_vector_data()
    except ValueError as e:
      logging.exception("Failed to get vector data from %s", ds_views.ds_views)
      self.visible_error(400, e.message)
      return None, None, None
    return ds_views.ds_kind, wave, X

  @classmethod
  def collect_variables(cls, ds_views, meta_keys):
    '''Collect metadata variables to predict from all loaded datasets.
    Returns a dict of {key: (array, display_name)}
    '''
    return {key: ds_views.get_metadata(key) for key in meta_keys}

  @classmethod
  def collect_one_variable(cls, ds_views, meta_key):
    return ds_views.get_metadata(meta_key)


def async_crossval(fig_data, model_cls, num_vars, cv_args, cv_kwargs,
                   xlabel='param', ylabel='MSE', logx=False, callback=None):
  '''Wrap cross-validation calls in a Thread to avoid hanging the server.'''
  def helper():
    fig_data.figure.clf(keep_observers=True)
    axes = axes_grid(fig_data.figure, num_vars, xlabel, ylabel)
    if logx:
      for ax in axes:
        ax.set_xscale('log')

    cv_gen = model_cls.cross_validate(*cv_args, **cv_kwargs)
    for i, (name, x, y, yerr) in enumerate(cv_gen):
      axes[i].set_title(name)
      axes[i].errorbar(x, y, yerr=yerr, lw=2, fmt='k-', ecolor='r',
                       elinewidth=1, capsize=0)
      if ylabel == 'MSE':
        idx = np.argmin(y)
        axes[i].annotate('RMSE: %.3f' % np.sqrt(y[idx]), xy=(x[idx], y[idx]),
                         xycoords='data', xytext=(0.5, 0.95),
                         textcoords='axes fraction',
                         horizontalalignment='center',
                         verticalalignment='top',
                         arrowprops=dict(
                             arrowstyle='->',
                             connectionstyle='arc3'))

    fig_data.manager.canvas.draw()
    fig_data.last_plot = '%s_crossval' % model_cls.__name__
    callback()

  t = Thread(target=helper)
  t.daemon = True
  t.start()


def axes_grid(fig, n, xlabel, ylabel):
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
    (r'/_load_model', ModelIOHandler),
    (r'/([0-9]+)/(regression|classifier)_model\.(csv|bin)', ModelIOHandler),
]
