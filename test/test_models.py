import unittest
import numpy as np
import warnings
from io import BytesIO
from numpy.testing import assert_array_equal, assert_array_almost_equal

from backend.models import (
    GenericModel, REGRESSION_MODELS, CLASSIFICATION_MODELS)

WAVE = np.linspace(100, 1000, 50)
DATA = np.arange(500).reshape((10, 50))
LABELS = ['a', 'a', 'a', 'b', 'b', 'c', 'b', 'b', 'c', 'd']
GROUPS = ['x', 'y', 'x', 'y', 'x', 'y', 'x', 'y', 'x', 'y']


class TestLogistic(unittest.TestCase):
    _expected_preds = ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'd']
    _variables = dict(foo=(LABELS, 'Foobar'))

    def test_train_predict(self):
        model_cls = CLASSIFICATION_MODELS['logistic']
        m = model_cls(1.5, 'NIR', WAVE)
        self.assertEqual(m.info_html(), 'Logistic(1.5) &mdash; NIR')

        m.train(DATA, self._variables)
        self.assertEqual(m.var_names, ['Foobar'])
        self.assertEqual(m.var_keys, ['foo'])

        preds = m.predict(DATA, self._variables)
        self.assertEqual(list(preds.keys()), ['foo'])
        assert_array_equal(preds['foo'], self._expected_preds)

    def test_load_save(self):
        m = CLASSIFICATION_MODELS['logistic'](1.5, 'NIR', WAVE)
        m.train(DATA, self._variables)

        buf = BytesIO()
        m.save(buf)
        buf.seek(0)

        x = GenericModel.load(buf)
        self.assertEqual(x.info_html(), 'Logistic(1.5) &mdash; NIR')
        self.assertEqual(x.var_names, ['Foobar'])
        self.assertEqual(x.var_keys, ['foo'])
        preds = m.predict(DATA, self._variables)
        self.assertEqual(list(preds.keys()), ['foo'])
        assert_array_equal(preds['foo'], self._expected_preds)

    def test_crossval(self):
        model_cls = CLASSIFICATION_MODELS['logistic']
        cv_gen = model_cls.cross_validate(
            DATA, self._variables, Cs=[0.1, 1, 10], num_folds=2, labels=None)
        for name, x, y, yerr in cv_gen:
            self.assertEqual(name, 'Foobar')
            assert_array_equal(x, [0.1, 1, 10])
            self.assertEqual(y.shape, (3,))
            self.assertEqual(yerr.shape, (3,))
            self.assertTrue(np.isfinite(y).all())
            self.assertTrue(np.isfinite(yerr).all())


class TestKNN(unittest.TestCase):
    _expected_preds = ['a', 'a', 'b', 'b', 'c', 'b', 'b', 'c', 'd', 'c']
    _variables = dict(predvar=(LABELS, 'The Label'))

    def test_train_predict(self):
        model_cls = CLASSIFICATION_MODELS['knn']
        m = model_cls(2, 'Raman', WAVE)
        self.assertEqual(m.info_html(), 'KNN(2) &mdash; Raman')

        m.train(DATA, self._variables)
        self.assertEqual(m.var_names, ['The Label'])
        self.assertEqual(m.var_keys, ['predvar'])

        preds = m.predict(DATA, self._variables)
        self.assertEqual(list(preds.keys()), ['predvar'])
        assert_array_equal(preds['predvar'], self._expected_preds)

    def test_crossval(self):
        model_cls = CLASSIFICATION_MODELS['knn']
        cv_gen = model_cls.cross_validate(DATA, self._variables, ks=[1, 2],
                                          num_folds=2, labels=None)
        for name, x, y, yerr in cv_gen:
            self.assertEqual(name, 'The Label')
            assert_array_equal(x, [1, 2])
            self.assertEqual(y.shape, (2,))
            self.assertEqual(yerr.shape, (2,))
            self.assertTrue(np.isfinite(y).all())
            self.assertTrue(np.isfinite(yerr).all())


class TestUnivariateRegression(unittest.TestCase):
    _variables = dict(foo=(np.linspace(1, 7, 10), 'Foo!'),
                      bar=(np.arange(10) * 3, 'Bar'))

    def test_train_predict(self):
        for kind, param in [('pls', 3), ('lasso', 0.1), ('lars', 2)]:
            model_cls = REGRESSION_MODELS[kind]['uni']
            m = model_cls(param, 'NIR', WAVE)
            m.train(DATA, self._variables)

            self.assertEqual(sorted(m.var_keys), ['bar', 'foo'])
            self.assertEqual(sorted(m.var_names), ['Bar', 'Foo!'])

            preds, stats = m.predict(DATA, self._variables)
            self.assertEqual(sorted(preds.keys()), ['bar', 'foo'])
            self.assertEqual(preds['foo'].size, 10)
            self.assertEqual(len(stats), 2)
            for s in stats:
                self.assertIn('name', s)
                self.assertIn('key', s)
                self.assertIn('r2', s)
                self.assertIn('rmse', s)

    def test_coefficients(self):
        v = dict(foo=self._variables['foo'])

        m = REGRESSION_MODELS['pls']['uni'](3, 'NIR', WAVE)
        m.train(DATA, v)
        ws, cs = m.coefficients()
        for w, c in zip(ws, cs):
            assert_array_equal(w, WAVE)
            assert_array_almost_equal(c, np.full(50, 0.000267))

        m = REGRESSION_MODELS['lasso']['uni'](0.1, 'NIR', WAVE)
        m.train(DATA, v)
        ws, cs = m.coefficients()
        for w, c in zip(ws, cs):
            assert_array_equal(w, [1000, 100])
            assert_array_almost_equal(c, [0.020098, -0.006715])

        m = REGRESSION_MODELS['lars']['uni'](2, 'NIR', WAVE)
        m.train(DATA, v)
        ws, cs = m.coefficients()
        for w, c in zip(ws, cs):
            assert_array_equal(w, [1000, 100])
            assert_array_almost_equal(c, [0.020408, -0.007075])

    def test_crossval_pls(self):
        model_cls = REGRESSION_MODELS['pls']['uni']
        for labels in (None, GROUPS):
            cv_gen = model_cls.cross_validate(
                DATA,
                self._variables,
                comps=[3, 4],
                num_folds=2,
                labels=labels)
            for name, x, y, yerr in cv_gen:
                assert_array_equal(x, [3, 4])
                self.assertEqual(y.shape, (2,))
                self.assertEqual(yerr.shape, (2,))
                self.assertTrue(np.isfinite(y).all())
                self.assertTrue(np.isfinite(yerr).all())

    def test_crossval_lars(self):
        model_cls = REGRESSION_MODELS['lars']['uni']
        cv_gen = model_cls.cross_validate(DATA, self._variables, chans=[1, 3],
                                          num_folds=2, labels=None)
        for name, x, y, yerr in cv_gen:
            assert_array_equal(x, [1, 3])
            self.assertEqual(y.shape, (2,))
            self.assertEqual(yerr.shape, (2,))
            self.assertTrue(np.isfinite(y).all())
            self.assertTrue(np.isfinite(yerr).all())

    def test_crossval_lasso(self):
        model_cls = REGRESSION_MODELS['lasso']['uni']
        for labels in (None, GROUPS):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                cv_gen = model_cls.cross_validate(
                    DATA, self._variables, num_folds=2, labels=labels)
                for name, x, y, yerr in cv_gen:
                    self.assertEqual(y.shape, (len(x),))
                    self.assertEqual(yerr.shape, (len(x),))
                    self.assertTrue(np.isfinite(y).all())
                    self.assertTrue(np.isfinite(yerr).all())

    def test_save_load(self):
        for kind, param in [('pls', 3), ('lasso', 0.1), ('lars', 2)]:
            model_cls = REGRESSION_MODELS[kind]['uni']
            m = model_cls(param, 'NIR', WAVE)
            m.train(DATA, self._variables)
            mpreds, mstats = m.predict(DATA, self._variables)

            buf = BytesIO()
            m.save(buf)
            buf.seek(0)

            x = GenericModel.load(buf)
            self.assertEqual(x.info_html(), m.info_html())
            self.assertEqual(x.var_names, m.var_names)
            self.assertEqual(x.var_keys, m.var_keys)
            xpreds, xstats = m.predict(DATA, self._variables)
            assert_array_equal(xpreds['foo'], mpreds['foo'])
            assert_array_equal(xpreds['bar'], mpreds['bar'])
            self.assertEqual(xstats, mstats)


class TestMultivariateRegression(unittest.TestCase):
    _variables = dict(foo=(np.linspace(1, 7, 10), 'Foo!'),
                      bar=(np.arange(10) * 3, 'Bar'))

    def test_train_predict(self):
        for kind, param in [('pls', 3), ('lasso', 0.1), ('lars', 2)]:
            model_cls = REGRESSION_MODELS[kind]['multi']
            m = model_cls(param, 'NIR', WAVE)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.train(DATA, self._variables)

            self.assertEqual(sorted(m.var_keys), ['bar', 'foo'])
            self.assertEqual(sorted(m.var_names), ['Bar', 'Foo!'])

            preds, stats = m.predict(DATA, self._variables)
            self.assertEqual(sorted(preds.keys()), ['bar', 'foo'])
            self.assertEqual(preds['foo'].size, 10)
            self.assertEqual(len(stats), 2)
            for s in stats:
                self.assertIn('name', s)
                self.assertIn('key', s)
                self.assertIn('r2', s)
                self.assertIn('rmse', s)

    def test_coefficients_pls(self):
        m = REGRESSION_MODELS['pls']['multi'](3, 'NIR', WAVE)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.train(DATA, self._variables)
        ws, cs = m.coefficients()
        expected = [np.full(50, 0.000267), np.full(50, 0.0012)]
        # Handle eigenvalue ordering flips
        if cs[0, 0] > cs[1, 0]:
            assert_array_almost_equal(cs, expected[::-1])
        else:
            assert_array_almost_equal(cs, expected)

    def test_coefficients_lasso(self):
        m = REGRESSION_MODELS['lasso']['multi'](100, 'NIR', WAVE)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.train(DATA, self._variables)
        ws, cs = m.coefficients()
        # Handle eigenvalue ordering flips
        if ws[0][0] < ws[1][0]:
            ws = ws[::-1]
            cs = cs[::-1]
        assert_array_almost_equal(ws[0], [1000])
        assert_array_almost_equal(ws[1], [853.061224, 118.367347, 210.204082])
        assert_array_almost_equal(cs[0], [0.013281])
        assert_array_almost_equal(cs[1], [0.025562, 0.026936, 0.002654])

    def test_coefficients_lars(self):
        m = REGRESSION_MODELS['lars']['multi'](2, 'NIR', WAVE)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m.train(DATA, self._variables)
        ws, cs = m.coefficients()
        expected_ws = [[1000, 100], [1000, 853.061224]]
        expected_cs = [[0.020408, -0.007075], [-0.3075, 0.3675]]
        # Handle eigenvalue ordering flips
        if cs[0][0] < cs[1][0]:
            assert_array_almost_equal(ws, expected_ws[::-1])
            assert_array_almost_equal(cs, expected_cs[::-1])
        else:
            assert_array_almost_equal(ws, expected_ws)
            assert_array_almost_equal(cs, expected_cs)

    def test_crossval_pls(self):
        model_cls = REGRESSION_MODELS['pls']['multi']
        for labels in (None, GROUPS):
            cv_gen = model_cls.cross_validate(
                DATA, self._variables, comps=[
                    3, 4], num_folds=2, labels=labels)
            for name, x, y, yerr in cv_gen:
                assert_array_equal(x, [3, 4])
                self.assertEqual(y.shape, (2,))
                self.assertEqual(yerr.shape, (2,))
                self.assertTrue(np.isfinite(y).all())
                self.assertTrue(np.isfinite(yerr).all())

    def test_crossval_lars(self):
        model_cls = REGRESSION_MODELS['lars']['multi']
        cv_gen = model_cls.cross_validate(DATA, self._variables, chans=[2, 3],
                                          num_folds=2, labels=None)
        for name, x, y, yerr in cv_gen:
            assert_array_equal(x, [2, 3])
            self.assertEqual(y.shape, (2,))
            self.assertEqual(yerr.shape, (2,))
            self.assertTrue(np.isfinite(y).all())
            self.assertTrue(np.isfinite(yerr).all())

    def test_save_load(self):
        for kind, param in [('pls', 3), ('lasso', 0.1), ('lars', 2)]:
            model_cls = REGRESSION_MODELS[kind]['multi']
            m = model_cls(param, 'NIR', WAVE)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.train(DATA, self._variables)
            mpreds, mstats = m.predict(DATA, self._variables)

            buf = BytesIO()
            m.save(buf)
            buf.seek(0)

            x = GenericModel.load(buf)
            self.assertEqual(x.info_html(), m.info_html())
            self.assertEqual(x.var_names, m.var_names)
            self.assertEqual(x.var_keys, m.var_keys)
            xpreds, xstats = m.predict(DATA, self._variables)
            assert_array_equal(xpreds['foo'], mpreds['foo'])
            assert_array_equal(xpreds['bar'], mpreds['bar'])
            self.assertEqual(xstats, mstats)


if __name__ == '__main__':
    unittest.main()
