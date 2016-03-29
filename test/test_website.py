import numpy as np
import time
import unittest
from mock import Mock
from numpy.testing import assert_array_equal

# import this first to make sure we call matplotlib.use() right away
from server.mpl_server import MatplotlibServer
from server.handlers.base import BaseHandler
from server.handlers.baseline_handlers import BaselineHandler
from server.handlers.handlers import SelectHandler
from server.web_datasets import (
    DATASETS, WebTrajDataset, WebVectorDataset,
    LookupMetadata, NumericMetadata, BooleanMetadata, PrimaryKeyMetadata,
    CompositionMetadata
)


def _load_traj_set(ds):
  data = dict(a=np.arange(12).reshape((6,2)), b=np.ones((7,2)))
  names = ['a', 'b']
  comps = {
    'x': NumericMetadata([0.2, 0.7]),
    'y': NumericMetadata([-4, 123]),
  }
  ds.set_data(names, data, lookup=LookupMetadata(['Test A', 'Test B']),
              numeric=NumericMetadata([2.3, 3.2], display_name='#s'),
              boolean=BooleanMetadata([False, True]),
              comps=CompositionMetadata(comps))
  return True


def _load_vector_set(ds):
  data = np.random.random((3, 14))
  bands = np.arange(14)
  ds.set_data(bands, data, pkey=PrimaryKeyMetadata(list("abc")))
  return True


class TestDatasetLoading(unittest.TestCase):
  def _generic_checks(self, ds):
    for _ in range(20):
      if ds.name in DATASETS[ds.kind]:
        break
      time.sleep(0.1)
    else:
      self.fail('Loader thread timed out after 2 seconds')

    self.assertIs(ds, DATASETS[ds.kind][ds.name])

  def test_traj_dataset_load(self):
    ds = WebTrajDataset('Test Set', 'Raman', _load_traj_set)
    self._generic_checks(ds)
    self.assertEqual(ds.num_spectra(), 2)
    self.assertIsNone(ds.num_dimensions())

  def test_vector_dataset_load(self):
    ds = WebVectorDataset('Test 2', 'NIR', _load_vector_set)
    self._generic_checks(ds)
    self.assertEqual(ds.num_spectra(), 3)
    self.assertEqual(ds.num_dimensions(), 14)


class TestRoutes(unittest.TestCase):
  def setUp(self):
    WebTrajDataset('Test Set', 'Raman', _load_traj_set)
    WebVectorDataset('Test 2', 'NIR', _load_vector_set)

    self.app = MatplotlibServer(cookie_secret='foobar')

    # Python3 compat
    if not hasattr(self, 'assertRegex'):
      self.assertRegex = self.assertRegexpMatches

    for _ in range(20):
      if len(DATASETS['Raman']) == 1 and len(DATASETS['NIR']) == 1:
        break
      time.sleep(0.1)
    else:
      self.fail('Loader threads timed out after 2 seconds')

  def test_base(self):
    req = Mock()
    req.cookies = dict()
    h = BaseHandler(self.app, req)
    self.assertTrue(h.is_private)
    datasets = h.all_datasets()
    self.assertEqual(len(datasets), 2)

  def test_select(self):
    fignum = self.app.register_new_figure((1,1))
    req = Mock()
    req.cookies = dict()
    req.arguments = dict(fignum=[str(fignum)], ds_name=['Test Set'],
                         ds_kind=['Raman'], name=['a'])
    h = SelectHandler(self.app, req)
    h.write = Mock()
    h.post()
    self.assertEqual(len(h.write.call_args_list), 1)
    args, kwargs = h.write.call_args
    self.assertRegex(args[0], '\[-?\d+.\d+, \d+.\d+, \d+.\d+, \d+.\d+]')
    self.assertEqual(kwargs, {})

    fig_data = self.app.figure_data[fignum]
    self.assertEqual(fig_data.title, 'a')
    x = fig_data.get_trajectory('upload')
    assert_array_equal(x, np.arange(12).reshape((6,2)))

  def test_baseline(self):
    fignum = self.app.register_new_figure((1,1))
    fig_data = self.app.figure_data[fignum]
    fig = fig_data.figure
    fig.add_subplot(212, sharex=fig.add_subplot(211))

    fig_data.set_selected(DATASETS['Raman']['Test Set'].view(mask=[0]))

    req = Mock()
    req.cookies = dict()
    req.arguments = dict(fignum=[str(fignum)], blr_method=['median'])
    h = BaselineHandler(self.app, req)
    h.write = Mock()
    h.post()

if __name__ == '__main__':
  unittest.main()
