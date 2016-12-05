import numpy as np
import time
import unittest
from mock import Mock
from numpy.testing import assert_array_equal

from backend import MatplotlibServer, BaseHandler
from backend.handlers.baseline_handlers import BaselineHandler
from backend.handlers.handlers import SelectHandler
from backend.web_datasets import (
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


WebTrajDataset('Test Set', 'Raman', _load_traj_set)
WebVectorDataset('Test 2', 'NIR', _load_vector_set)
for _ in range(20):
  if len(DATASETS['Raman']) == 1 and len(DATASETS['NIR']) == 1:
    break
  time.sleep(0.25)
else:
  assert False


class RouteTester(unittest.TestCase):
  def setUp(self):
    self.app = MatplotlibServer(cookie_secret='foobar')

    # Python3 compat
    if not hasattr(self, 'assertRegex'):
      self.assertRegex = self.assertRegexpMatches


class TestGenericRoutes(RouteTester):
  def test_base(self):
    req = Mock(cookies=dict())
    h = BaseHandler(self.app, req)
    self.assertTrue(h.is_private)
    datasets = h.all_datasets()
    self.assertEqual(len(datasets), 2)

  def test_select(self):
    fignum = self.app.register_new_figure((1,1))
    req = Mock(cookies=dict())
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


class TestBaselineRoutes(RouteTester):
  def setUp(self):
    RouteTester.setUp(self)
    self.fignum = self.app.register_new_figure((1,1))
    fig_data = self.app.figure_data[self.fignum]
    fig_data.set_selected(DATASETS['Raman']['Test Set'].view(mask=[0]))

  def test_baseline_2axes(self):
    fig = self.app.figure_data[self.fignum].figure
    fig.add_subplot(212, sharex=fig.add_subplot(211))

    req = Mock(cookies=dict())
    req.arguments = dict(fignum=[str(self.fignum)], blr_method=['median'],
                         blr_window_=['3'])
    BaselineHandler(self.app, req).post()

  def test_baseline_1axis(self):
    req = Mock(cookies=dict())
    req.arguments = dict(fignum=[str(self.fignum)], blr_method=['polyfit'])
    BaselineHandler(self.app, req).post()

  def test_baseline_download(self):
    req = Mock()
    h = BaselineHandler(self.app, req)
    h.write = Mock()
    h.finish = Mock()
    h.get(str(self.fignum))

    self.assertEqual(len(h.write.call_args_list), 6)
    self.assertEqual(len(h.finish.call_args_list), 1)


if __name__ == '__main__':
  unittest.main()
