from __future__ import absolute_import, print_function, division
import logging
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from tornado import gen

from .generic_models import GenericModelHandler, async_crossval
from ..models import CLASSIFICATION_MODELS, KNN
from six.moves import map
from six.moves import zip


class ClassificationModelHandler(GenericModelHandler):
  def get(self, fignum):
    '''Download predictions as CSV.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    ds_views = self.prepare_ds_views(fig_data)
    if ds_views is None:
      self.write('No datasets selected / too many spectra.')
      return

    # load the model from the session data
    model = fig_data.classify_model

    # collect primary keys for row labels
    all_pkeys = ds_views.get_primary_keys()

    # collect data and variables to predict
    variables = self.collect_variables(ds_views, model.var_keys)
    actuals, var_name = variables[model.var_keys[0]]

    if isinstance(model, KNN):
      ds_kind, X = self.collect_trajectories(ds_views)
    else:
      ds_kind, _, X = self.collect_spectra(ds_views)
      if X is None:
        # self.visible_error has already been called in collect_spectra
        return

    # re-run the classifier
    pred_dict = model.predict(X, variables)
    preds = pred_dict[model.var_keys[0]].ravel()

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    self.write('Spectrum,%s,Predicted\n' % var_name)
    for row in zip(all_pkeys, actuals, preds):
      self.write('%s,%s,%s\n' % row)
    self.finish()

  @gen.coroutine
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.visible_error(403, "Broken connection to server.")
      return

    ds_views = self.prepare_ds_views(fig_data, nan_gap=None)
    if ds_views is None:
      self.visible_error(404, "Invalid dataset selection.")
      return

    model_kind = self.get_argument('model_kind')
    model_cls = CLASSIFICATION_MODELS[model_kind]
    params = dict(knn=int(self.get_argument('knn_k')),
                  logistic=float(self.get_argument('logistic_C')))
    pred_key = self.get_argument('pred_var')
    variables = self.collect_variables(ds_views, (pred_key,))

    if model_kind == 'knn':
      # TODO: detect case where collect_spectra would work,
      #  and use vectors then instead of forcing trajectory format
      ds_kind = ds_views.ds_kind
      wave = None
      X = [np.array(t, dtype=np.float32, order='C')
           for t in ds_views.get_trajectories()]
    else:
      ds_kind, wave, X = self.collect_spectra(ds_views)
      if X is None:
        # self.visible_error has already been called in collect_spectra
        return

    do_train = self.get_argument('do_train', None)
    if do_train is None:
      # set up cross validation info
      folds = int(self.get_argument('cv_folds'))
      stratify_meta = self.get_argument('cv_stratify', '')
      if stratify_meta:
        vals, _ = self.collect_one_variable(ds_views, stratify_meta)
        _, stratify_labels = np.unique(vals, return_inverse=True)
      else:
        stratify_labels = None
      cv_args = (X, variables)
      cv_kwargs = dict(num_folds=folds, labels=stratify_labels)
      logging.info('Running %d-fold (%s) cross-val for %s', folds,
                   stratify_meta, model_cls.__name__)

      plot_kwargs = dict(ylabel='Accuracy')
      if model_kind == 'knn':
        cv_kwargs['ks'] = np.arange(int(self.get_argument('cv_min_k')),
                                    int(self.get_argument('cv_max_k')) + 1)
        plot_kwargs['xlabel'] = '# neighbors'
      else:
        start, stop = list(map(int, (self.get_argument('cv_min_logC'),
                                self.get_argument('cv_max_logC'))))
        cv_kwargs['Cs'] = np.logspace(start, stop, num=20, endpoint=True)
        plot_kwargs['xlabel'] = 'C'
        plot_kwargs['logx'] = True

      # run the cross validation
      yield gen.Task(async_crossval, fig_data, model_cls, len(variables),
                     cv_args, cv_kwargs, **plot_kwargs)
      return

    if bool(int(do_train)):
      # train on all the data
      model = model_cls(params[model_kind], ds_kind, wave)
      logging.info('Training %s on %d inputs, predicting %d variables',
                   model, len(X), len(variables))
      model.train(X, variables)
      fig_data.classify_model = model
    else:
      # use existing model
      model = fig_data.classify_model
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
      if (model_kind != 'knn' and (wave.shape != model.wave.shape or
                                   not np.allclose(wave, model.wave))):
        if wave[-1] <= model.wave[0] or wave[0] >= model.wave[-1]:
          self.visible_error(400, "Data to predict doesn't overlap "
                                  "with training wavelengths.")
          return
        Xnew = np.empty((X.shape[0], model.wave.shape[0]), dtype=X.dtype)
        for i, y in enumerate(X):
          Xnew[i] = np.interp(model.wave, wave, y)
        X = Xnew

    # get predictions for each variable
    preds = model.predict(X, variables)

    # plot
    stats = _plot_confusion(preds, fig_data.figure, variables)
    fig_data.manager.canvas.draw()
    fig_data.last_plot = 'classify_preds'

    res = dict(stats=stats, info=fig_data.classify_model.info_html())
    self.write_json(res)


def _plot_confusion(preds, fig, variables):
  fig.clf(keep_observers=True)
  ax = fig.add_subplot(1, 1, 1)

  key, = list(variables.keys())
  p = preds[key].ravel()
  y, name = variables[key]
  ax.set_title(name)

  if y is None:
    # no actual values exist, so only plot the predictions
    classes, counts = np.unique(p, return_counts=True)
    tick_locs = np.arange(len(classes))
    ax.bar(tick_locs, counts, tick_label=classes, align='center')
    ax.set_ylabel('# Predicted')
    return []

  # true labels exist, so plot a confusion matrix
  classes, counts = np.unique(y, return_counts=True)
  conf = confusion_matrix(y, p)
  correct = conf.diagonal()
  conf = conf.T * 100 / counts.astype(float)
  im = ax.imshow(conf, interpolation='nearest')
  fig.colorbar(im, label='% of Actual')
  tick_locs = np.arange(len(classes))
  ax.set_xticks(tick_locs)
  ax.set_yticks(tick_locs)
  ax.set_xticklabels(classes, rotation=45)
  ax.set_yticklabels(classes)
  ax.set_xlabel('Actual')
  ax.set_ylabel('Predicted')
  # return stats for showing in the model_error table
  return [{'class': k, 'correct': c, 'total': t} for k, c, t in
          zip(classes, correct, counts)]


routes = [
    (r'/_run_classifier', ClassificationModelHandler),
    (r'/([0-9]+)/classifier_predictions\.csv', ClassificationModelHandler),
]
