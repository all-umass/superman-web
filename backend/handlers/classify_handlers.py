from __future__ import absolute_import
import logging
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from tornado import gen
from tornado.escape import json_encode

from .model_handlers import GenericModelHandler, async_crossval
from ..models import CLASSIFICATION_MODELS


class ClassificationModelHandler(GenericModelHandler):
  def get(self, fignum):
    '''Download predictions as CSV.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
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

    # TODO: just re-run the classifier
    model = fig_data.classify_model
    actuals = [None] * len(all_pkeys)
    preds = actuals

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    self.write('Spectrum,%s,Predicted\n' % model.var_name)
    for row in zip(all_pkeys, actuals, preds):
      self.write('%s,%s,%s\n' % row)
    self.finish()

  @gen.coroutine
  def post(self):
    res = self.validate_inputs()
    if res is None:
      return
    fig_data, all_ds_views, ds_kind, wave, X = res

    model_kind = self.get_argument('model_kind')
    model_cls = CLASSIFICATION_MODELS[model_kind]
    params = dict(knn=int(self.get_argument('knn_k')),
                  logistic=int(self.get_argument('logistic_C')))
    pred_key = self.get_argument('pred_var')
    variables = self.collect_variables(all_ds_views, (pred_key,))

    do_train = self.get_argument('do_train', None)
    if do_train is None:
      # set up cross validation info
      folds = int(self.get_argument('cv_folds'))
      stratify_meta = self.get_argument('cv_stratify', '')
      if stratify_meta:
        vals, _ = self.collect_one_variable(all_ds_views, stratify_meta)
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
        start, stop = map(int, (self.get_argument('cv_min_logC'),
                                self.get_argument('cv_max_logC')))
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
                   model, X.shape[0], len(variables))
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
    _plot_confusion(preds, stats, fig_data.figure, variables)
    fig_data.manager.canvas.draw()
    fig_data.last_plot = 'classify_preds'

    res = dict(stats=stats, info=fig_data.classify_model.info_html())
    # NaN isn't valid JSON, but json_encode doesn't catch it. :'(
    self.write(json_encode(res).replace('NaN', 'null'))


def _plot_confusion(preds, stats, fig, variables):
  fig.clf(keep_observers=True)
  ax = fig.add_subplot(1, 1, 1)

  key, = variables.keys()
  p = preds[key].ravel()
  y, name = variables[key]
  ax.set_title(name)

  if y is not None:
    # true labels exist, so plot a confusion matrix
    classes = np.unique(y)
    conf = confusion_matrix(y, p).T
    im = ax.imshow(conf, interpolation='nearest')
    fig.colorbar(im)
    tick_locs = np.arange(len(classes))
    ax.set_xticks(tick_locs)
    ax.set_yticks(tick_locs)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    # TODO: add text labels to the diagonal with %acc?
  else:
    # no actual values exist, so only plot the predictions
    classes, counts = np.unique(p, return_counts=True)
    tick_locs = np.arange(len(classes))
    ax.bar(tick_locs, counts, tick_label=classes, align='center')
    ax.set_ylabel('# Predicted')


routes = [
    (r'/_run_classifier', ClassificationModelHandler),
    (r'/([0-9]+)/classifier_predictions\.csv', ClassificationModelHandler),
]
