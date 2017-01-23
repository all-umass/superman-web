from __future__ import absolute_import
import ast
import logging
import numpy as np
from itertools import repeat
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
from sklearn.linear_model import (
    LassoLars, LassoLarsCV, Lars, LogisticRegression)
from superman.distance import pairwise_within, pairwise_dists

__all__ = ['GenericModel', 'REGRESSION_MODELS', 'CLASSIFICATION_MODELS']


class GenericModel(object):
  def __init__(self, param, ds_kind, wave):
    self.parameter = param
    self.ds_kind = ds_kind
    self.wave = wave
    self.var_names = []
    self.var_keys = []

  @staticmethod
  def load(fh):
    class_name = fh.readline().strip()
    cls = globals().get(class_name)
    if not issubclass(cls, GenericModel):
      raise ValueError('Invalid model file.')
    param = ast.literal_eval(fh.readline().strip())
    ds_kind = fh.readline().strip()
    var_names = ast.literal_eval(fh.readline().strip())
    var_keys = ast.literal_eval(fh.readline().strip())
    wave_len = int(fh.readline().strip())
    wave = np.fromstring(fh.read(wave_len * 8))
    model = cls(param, ds_kind, wave)
    model.var_names = var_names
    model.var_keys = var_keys
    model._finish_loading(fh)
    return model

  def save(self, fh):
    w = np.array(self.wave, dtype=float, copy=False)
    fh.write(b'%s\n%r\n%s\n%r\n%r\n%d\n' % (
        self.__class__.__name__, self.parameter, self.ds_kind, self.var_names,
        self.var_keys, w.shape[0]))
    # don't use w.tofile(fh) here, because it requires an actual file object
    fh.write(w.tobytes())

  def info_html(self):
    return '%s &mdash; %s' % (self, self.ds_kind)

  def __str__(self):
    return '%s(%g)' % (self.__class__.__name__, self.parameter)


class _Classifier(GenericModel):
  def train(self, X, variables):
    key, = variables.keys()
    y, name = variables[key]
    self.var_keys = [key]
    self.var_names = [name]
    self._train(X, y)

  def predict(self, X, variables):
    key, = variables.keys()
    preds = {key: None}
    if key != self.var_keys[0]:
      logging.warning('No trained model for variable: %r', key)
    else:
      preds[key] = self._predict(X)
    return preds


class Logistic(_Classifier):
  def _train(self, X, y):
    self.clf = LogisticRegression(C=self.parameter, fit_intercept=False)
    self.clf.fit(X, y)

  def _predict(self, X):
    return self.clf.predict(X)

  @classmethod
  def cross_validate(cls, X, variables, Cs=None, num_folds=5, labels=None):
    if labels is None:
      cv = KFold(n_splits=num_folds)
    else:
      cv = GroupKFold(n_splits=num_folds)

    key, = variables.keys()
    y, name = variables[key]
    # TODO: use LogisticRegressionCV here instead?
    model = GridSearchCV(LogisticRegression(fit_intercept=False), dict(C=Cs),
                         cv=cv, return_train_score=False, n_jobs=1)
    model.fit(X, y, groups=labels)
    acc_mean = model.cv_results_['mean_test_score']
    acc_stdv = model.cv_results_['std_test_score']
    yield name, Cs, acc_mean, acc_stdv

  def _finish_loading(self, fh):
    params = ast.literal_eval(fh.readline().strip())
    classes = ast.literal_eval(fh.readline().strip())
    coef_shape = ast.literal_eval(fh.readline().strip())
    self.clf = LogisticRegression().set_params(**params)
    self.clf.classes_ = np.array(classes)
    self.clf.intercept_ = 0.0
    n = np.prod(coef_shape) * 8
    self.clf.coef_ = np.fromstring(fh.read(n)).reshape(coef_shape)

  def save(self, fh):
    GenericModel.save(self, fh)
    fh.write('%r\n%r\n%r\n' % (self.clf.get_params(),
                               self.clf.classes_.tolist(),
                               self.clf.coef_.shape))
    fh.write(self.clf.coef_.tobytes())


class KNN(_Classifier):
  def _train(self, X, y):
    self.library = X
    self.classes, self.labels = np.unique(y, return_inverse=True)
    # TODO: expose these in the UI
    self.metric = 'cosine'
    self.weighting = 'distance'

  def _predict(self, X):
    if X is self.library:
      dists = pairwise_within(X, self.metric, num_procs=5)
      ks = slice(1, min(self.parameter+1, dists.shape[0]))
    else:
      dists = pairwise_dists(self.library, X, self.metric, num_procs=5)
      ks = slice(0, min(self.parameter, dists.shape[0]))
    top_k = np.argsort(dists, axis=0)[ks]  # shape: (k, nX)

    num_classes = len(self.classes)
    votes = np.zeros((num_classes, len(X)))
    idx = np.arange(len(X))
    if self.weighting == 'uniform':
      for kk in top_k:
        labels = self.labels[kk]
        votes[labels[None], idx] += 1
    else:
      for kk in top_k:
        labels = self.labels[kk]
        votes[labels[None], idx] += 1./(1 + dists[kk, idx])
    winner = votes.argmax(axis=0)
    return self.classes[winner]

  @classmethod
  def cross_validate(cls, X, variables, ks=None, num_folds=5, labels=None):
    if labels is None:
      cv = KFold(n_splits=num_folds)
    else:
      cv = GroupKFold(n_splits=num_folds)

    key, = variables.keys()
    y, name = variables[key]

    acc = np.zeros((num_folds, len(ks)))
    for i, (train_idx, test_idx) in enumerate(cv.split(y, groups=labels)):
      trainY, testY = y[train_idx], y[test_idx]
      if hasattr(X, 'shape'):
        trainX, testX = X[train_idx], X[test_idx]
      else:
        trainX = [X[ti] for ti in train_idx]
        testX = [X[ti] for ti in test_idx]
      for j, k in enumerate(ks):
        clf = cls(k, None, None)
        clf._train(trainX, trainY)
        pred = clf._predict(testX)
        acc[i, j] = (pred == testY).mean()
    yield name, ks, acc.mean(axis=0), acc.std(axis=0)

  def _finish_loading(self, fh):
    raise NotImplementedError('Cannot load a KNN model from file.')

  def save(self, fh):
    raise NotImplementedError('Cannot serialize a KNN model.')


class _RegressionModel(GenericModel):
  def predict(self, X, variables):
    preds, stats = {}, []
    for key, p in self._predict(X, variables):
      y, name = variables[key]
      preds[key] = p
      stats_entry = dict(name=name, key=key, r2=np.nan, rmse=np.nan)
      if y is not None:
        mask = np.isfinite(y)
        nnz = np.count_nonzero(mask)
        if nnz != 0:
          if nnz < len(y):
            y, p = y[mask], p[mask]
          stats_entry['r2'] = r2_score(y, p)
          stats_entry['rmse'] = np.sqrt(mean_squared_error(y, p))
      stats.append(stats_entry)
    stats.sort(key=lambda s: s['key'])
    return preds, stats


class _UnivariateRegression(_RegressionModel):
  def train(self, X, variables):
    self.models = {}
    for key in variables:
      y, name = variables[key]
      m = self._construct()
      _try_to_fit(m, X, y)
      self.models[key] = m
      self.var_keys.append(key)
      self.var_names.append(name)

  def _predict(self, X, variables):
    for key in variables:
      if key not in self.models:
        logging.warning('No trained model for variable: %r', key)
        continue
      clf = self.models[key]
      yield key, clf.predict(X)

  @classmethod
  def _run_cv(cls, X, variables, grid, num_folds, labels=None):
    if labels is None:
      cv = KFold(n_splits=num_folds)
    else:
      cv = GroupKFold(n_splits=num_folds)

    for key in sorted(variables):
      y, name = variables[key]
      model = GridSearchCV(cls._cv_construct(), grid, cv=cv,
                           scoring='neg_mean_squared_error',
                           return_train_score=False, n_jobs=1)
      _try_to_fit(model, X, y, groups=labels)
      mse_mean = -model.cv_results_['mean_test_score']
      mse_stdv = model.cv_results_['std_test_score']
      yield name, mse_mean, mse_stdv

  def _finish_loading(self, fh):
    self.models = {key: self._load_model(fh) for key in self.var_keys}

  def save(self, fh):
    GenericModel.save(self, fh)
    for key in self.var_keys:
      self._save_model(self.models[key], fh)


class _MultivariateRegression(_RegressionModel):
  def train(self, X, variables):
    self.clf = self._construct()
    self.var_keys = variables.keys()
    y_cols = []
    for key in self.var_keys:
      y, name = variables[key]
      y_cols.append(y)
      self.var_names.append(name)
    _try_to_fit(self.clf, X, np.column_stack(y_cols))

  def _predict(self, X, variables):
    P = self.clf.predict(X)
    for i, key in enumerate(self.var_keys):
      if key not in variables:
        logging.warning('No input variable for predicted: %r', key)
        continue
      yield key, P[:,i]

  @classmethod
  def _run_cv(cls, X, variables, grid, num_folds, labels=None):
    if labels is None:
      cv = KFold(n_splits=num_folds)
    else:
      cv = GroupKFold(n_splits=num_folds)

    pls = GridSearchCV(cls._cv_construct(), grid, cv=cv,
                       scoring='neg_mean_squared_error',
                       return_train_score=False, n_jobs=1)
    Y, names = zip(*variables.values())
    _try_to_fit(pls, X, np.column_stack(Y), groups=labels)
    mse_mean = -pls.cv_results_['mean_test_score']
    mse_stdv = pls.cv_results_['std_test_score']
    return '/'.join(names), mse_mean, mse_stdv

  def _finish_loading(self, fh):
    self.clf = self._load_model(fh)

  def save(self, fh):
    GenericModel.save(self, fh)
    self._save_model(self.clf, fh)


class _PLS(object):
  def _construct(self):
    return PLSRegression(scale=False, n_components=self.parameter)

  @classmethod
  def _cv_construct(cls):
    return PLSRegression(scale=False)

  @classmethod
  def _save_model(cls, pls, fh):
    fh.write('%r\n%r\n' % (pls.get_params(), pls.coef_.shape))
    fh.write(pls.x_mean_.tobytes())
    fh.write(pls.y_mean_.tobytes())
    fh.write(pls.coef_.tobytes())

  @classmethod
  def _load_model(cls, fh):
    params = ast.literal_eval(fh.readline().strip())
    coef_shape = ast.literal_eval(fh.readline().strip())
    pls = PLSRegression().set_params(**params)
    pls.x_mean_ = np.fromstring(fh.read(coef_shape[0] * 8))
    pls.y_mean_ = np.fromstring(fh.read(coef_shape[1] * 8))
    pls.x_std_ = np.ones(coef_shape[0])
    pls.y_std_ = np.ones(coef_shape[1])
    n = coef_shape[0] * coef_shape[1] * 8
    pls.coef_ = np.fromstring(fh.read(n)).reshape(coef_shape)
    return pls


class _Lasso(object):
  def _construct(self):
    return LassoLars(alpha=self.parameter, fit_intercept=False)

  @classmethod
  def _save_model(cls, m, fh):
    fh.write('%r\n%r\n%r\n' % (m.get_params(), m.active_, m.coef_.shape))
    fh.write(m.coef_.tobytes())

  @classmethod
  def _load_model(cls, fh):
    params = ast.literal_eval(fh.readline().strip())
    active = ast.literal_eval(fh.readline().strip())
    coef_shape = ast.literal_eval(fh.readline().strip())
    m = LassoLars().set_params(**params)
    m.intercept_ = 0.0
    n = coef_shape[0] * coef_shape[1] * 8
    m.coef_ = np.fromstring(fh.read(n)).reshape(coef_shape)
    m.active_ = active
    return m


class _Lars(object):
  def _construct(self):
    return Lars(n_nonzero_coefs=self.parameter, fit_intercept=False)

  @classmethod
  def _cv_construct(cls):
    return Lars(fit_intercept=False, fit_path=False)

  @classmethod
  def _save_model(cls, m, fh):
    fh.write('%r\n%r\n%r\n' % (m.get_params(), m.active_, m.coef_.shape))
    fh.write(m.coef_.tobytes())

  @classmethod
  def _load_model(cls, fh):
    params = ast.literal_eval(fh.readline().strip())
    active = ast.literal_eval(fh.readline().strip())
    coef_shape = ast.literal_eval(fh.readline().strip())
    m = Lars().set_params(**params)
    m.intercept_ = 0.0
    n = coef_shape[0] * coef_shape[1] * 8
    m.coef_ = np.fromstring(fh.read(n)).reshape(coef_shape)
    m.active_ = active
    return m


class _LassoOrLars1(_UnivariateRegression):
  def train(self, X, variables):
    _UnivariateRegression.train(self, X, variables)
    for clf in self.models.values():
      # XXX: work around a bug in sklearn
      # see https://github.com/scikit-learn/scikit-learn/pull/8160
      clf.coef_ = np.array(clf.coef_)

  def coefficients(self):
    all_bands = []
    all_coefs = []
    for key in self.var_keys:
      clf = self.models[key]
      all_bands.append(self.wave[clf.active_])
      all_coefs.append(clf.coef_[clf.active_])
    return all_bands, all_coefs


class _LassoOrLars2(_MultivariateRegression):
  def train(self, X, variables):
    _MultivariateRegression.train(self, X, variables)
    # XXX: work around a bug in sklearn
    # see https://github.com/scikit-learn/scikit-learn/pull/8160
    self.clf.coef_ = np.array(self.clf.coef_)

  def coefficients(self):
    coef = self.clf.coef_
    active = self.clf.active_
    if coef.ndim == 1:
      all_bands = [self.wave[active]]
      all_coefs = [coef[active]]
    else:
      all_bands = [self.wave[idx] for idx in active]
      all_coefs = [cc[idx] for idx,cc in zip(active, coef)]
    return all_bands, all_coefs


class PLS1(_UnivariateRegression, _PLS):
  def coefficients(self):
    all_bands = repeat(self.wave)
    all_coefs = [self.models[key].coef_.ravel() for key in self.var_keys]
    return all_bands, all_coefs

  @classmethod
  def cross_validate(cls, X, variables, comps=None, num_folds=5, labels=None):
    grid = dict(n_components=comps)
    for name, mse_mean, mse_stdv in cls._run_cv(X, variables, grid, num_folds,
                                                labels=labels):
      yield name, comps, mse_mean, mse_stdv


class PLS2(_MultivariateRegression, _PLS):
  def coefficients(self):
    return repeat(self.wave), self.clf.coef_.T

  @classmethod
  def cross_validate(cls, X, variables, comps=None, num_folds=5, labels=None):
    grid = dict(n_components=comps)
    name, mse_mean, mse_stdv = cls._run_cv(X, variables, grid, num_folds,
                                           labels=labels)
    yield name, comps, mse_mean, mse_stdv


class Lars1(_LassoOrLars1, _Lars):
  @classmethod
  def cross_validate(cls, X, variables, chans=None, num_folds=5, labels=None):
    grid = dict(n_nonzero_coefs=chans)
    for name, mse_mean, mse_stdv in cls._run_cv(X, variables, grid, num_folds,
                                                labels=labels):
      yield name, chans, mse_mean, mse_stdv


class Lars2(_LassoOrLars2, _Lars):
  @classmethod
  def cross_validate(cls, X, variables, chans=None, num_folds=5, labels=None):
    grid = dict(n_nonzero_coefs=chans)
    name, mse_mean, mse_stdv = cls._run_cv(X, variables, grid, num_folds,
                                           labels=labels)
    yield name, chans, mse_mean, mse_stdv


class Lasso1(_LassoOrLars1, _Lasso):
  @classmethod
  def cross_validate(cls, X, variables, num_folds=5, labels=None):
    if labels is None:
      cv = KFold(n_splits=num_folds)
    else:
      cv = HackAroundSklearnCV(n_splits=num_folds, groups=labels)

    for key in sorted(variables):
      y, name = variables[key]
      lasso = LassoLarsCV(fit_intercept=False, max_iter=2000, cv=cv, n_jobs=1)
      _try_to_fit(lasso, X, y)
      cv_mse = lasso.mse_path_
      mse_mean = cv_mse.mean(axis=1)
      mse_stdv = cv_mse.std(axis=1)
      yield name, lasso.cv_alphas_, mse_mean, mse_stdv


class Lasso2(_LassoOrLars2, _Lasso):
  @classmethod
  def cross_validate(cls, X, variables, num_folds=5, labels=None):
    raise NotImplementedError("Multivariate Lasso doesn't yet support CV.")


class HackAroundSklearnCV(GroupKFold):
  """LassoLarsCV doesn't pass along `groups`, so we have to hack it in here."""
  def __init__(self, groups=None, **kwargs):
    GroupKFold.__init__(self, **kwargs)
    self.groups = groups

  def split(self, X, y):
    return GroupKFold.split(self, X, y=y, groups=self.groups)


def _try_to_fit(model, X, y, groups=None):
  """Fit a sklearn model, retrying in case there are non-finites in y."""
  try:
    if groups is None:
      model.fit(X, y)
    else:
      model.fit(X, y, groups=groups)
  except ValueError as e:
    # if there are NaNs in y, try again without them
    mask = np.isfinite(y)
    if mask.all():
      # no NaNs in y, so it must have been something else
      raise e
    if groups is None:
      model.fit(X[mask, :], y[mask])
    else:
      model.fit(X[mask, :], y[mask], groups=groups[mask])


REGRESSION_MODELS = dict(
    pls=dict(uni=PLS1, multi=PLS2),
    lasso=dict(uni=Lasso1, multi=Lasso2),
    lars=dict(uni=Lars1, multi=Lars2),
)
CLASSIFICATION_MODELS = dict(knn=KNN, logistic=Logistic)
