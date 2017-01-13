from __future__ import absolute_import
import logging
import numpy as np
from itertools import repeat
from sklearn.cross_decomposition import PLSRegression
from sklearn.externals.joblib.numpy_pickle import (
    load as load_pickle, dump as dump_pickle)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
from sklearn.linear_model import LassoLarsCV, LassoLars


def _gridsearch_pls(X, variables, comps, num_folds, labels=None):
  grid = dict(n_components=comps)
  if labels is None:
    cv = KFold(n_splits=num_folds)
  else:
    cv = GroupKFold(n_splits=num_folds)

  for key in sorted(variables):
    y, name = variables[key]
    pls = GridSearchCV(PLSRegression(scale=False), grid, cv=cv,
                       scoring='neg_mean_squared_error',
                       return_train_score=False, n_jobs=1)
    pls.fit(X, y=y, groups=labels)
    mse_mean = -pls.cv_results_['mean_test_score']
    mse_stdv = pls.cv_results_['std_test_score']
    yield name, comps, mse_mean, mse_stdv


def _gridsearch_lasso(X, variables, num_folds, labels=None):
  if labels is None:
    cv = KFold(n_splits=num_folds)
  else:
    cv = HackAroundSklearnCV(n_splits=num_folds, groups=labels)

  for key in sorted(variables):
    y, name = variables[key]
    lasso = LassoLarsCV(fit_intercept=False, max_iter=2000, cv=cv, n_jobs=1)
    lasso.fit(X, y)
    cv_mse = lasso.mse_path_
    mse_mean = cv_mse.mean(axis=1)
    mse_stdv = cv_mse.std(axis=1)
    yield name, lasso.cv_alphas_, mse_mean, mse_stdv


class HackAroundSklearnCV(GroupKFold):
  """LassoLarsCV doesn't pass along `groups`, so we have to hack it in here."""
  def __init__(self, groups=None, **kwargs):
    GroupKFold.__init__(self, **kwargs)
    self.groups = groups

  def split(self, X, y):
    return GroupKFold.split(self, X, y=y, groups=self.groups)


class _RegressionModel(object):
  def __init__(self, k, ds_kind, wave):
    self.parameter = k
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
      if y is not None and np.isfinite(y).all():
        stats.append(dict(r2=r2_score(y, p),
                          rmse=np.sqrt(mean_squared_error(y, p)),
                          name=name, key=key))
      else:
        stats.append(dict(name=name, key=key, r2=np.nan, rmse=np.nan))
    stats.sort(key=lambda s: s['key'])
    return preds, stats

  def info_html(self):
    return '%s &mdash; %s' % (self, self.ds_kind)

  def __str__(self):
    return '%s(%g)' % (self.__class__.__name__, self.parameter)


class PLS1(_RegressionModel):
  def train(self, X, variables):
    self.models = {}
    for key in variables:
      clf = PLSRegression(scale=False, n_components=self.parameter)
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

  def coefficients(self):
    all_bands = repeat(self.wave)
    all_coefs = [self.models[key].coef_.ravel() for key in self.var_keys]
    return all_bands, all_coefs


class PLS2(_RegressionModel):
  def train(self, X, variables):
    self.clf = PLSRegression(scale=False, n_components=self.parameter)
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

  def coefficients(self):
    return repeat(self.wave), self.clf.coef_.T


class Lasso1(_RegressionModel):
  def train(self, X, variables):
    self.models = {}
    for key in variables:
      clf = LassoLars(alpha=self.parameter, fit_intercept=False)
      y, name = variables[key]
      self.models[key] = clf.fit(X, y)
      # XXX: work around a bug in sklearn
      # see https://github.com/scikit-learn/scikit-learn/pull/8160
      clf.coef_ = np.array(clf.coef_)
      self.var_keys.append(key)
      self.var_names.append(name)

  def _predict(self, X, variables):
    for key in variables:
      if key not in self.models:
        logging.warning('No trained model for variable: %r', key)
        continue
      clf = self.models[key]
      yield key, clf.predict(X)[:, None]

  def coefficients(self):
    all_bands = []
    all_coefs = []
    for key in self.var_keys:
      clf = self.models[key]
      all_bands.append(self.wave[clf.active_])
      all_coefs.append(clf.coef_[clf.active_])
    return all_bands, all_coefs


class Lasso2(_RegressionModel):
  def train(self, X, variables):
    self.clf = LassoLars(alpha=self.parameter, fit_intercept=False)
    self.var_keys = variables.keys()
    y_cols = []
    for key in self.var_keys:
      y, name = variables[key]
      y_cols.append(y)
      self.var_names.append(name)
    self.clf.fit(X, np.column_stack(y_cols))
    # XXX: work around a bug in sklearn
    # see https://github.com/scikit-learn/scikit-learn/pull/8160
    self.clf.coef_ = np.array(self.clf.coef_)

  def _predict(self, X, variables):
    P = self.clf.predict(X)
    if P.ndim == 1:
      assert len(variables) == 1
      P = P[:,None]
    for i, key in enumerate(self.var_keys):
      if key not in variables:
        logging.warning('No input variable for predicted: %r', key)
        continue
      yield key, P[:,i]

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
