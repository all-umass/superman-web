import os.path
import unittest
from mock import Mock

from backend import MatplotlibServer
from backend.handlers.page_handlers import (
    MainPage, LoginPage, DatasetsPage, DataExplorerPage, BaselinePage,
    PeakFitPage, DatasetImportPage)


class TestNoDataPages(unittest.TestCase):
  def setUp(self):
    tpl = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates')
    self.app = MatplotlibServer([], cookie_secret='foobar', template_path=tpl)

  def test_main_page(self):
    req = Mock(cookies=dict(), headers=dict())
    h = MainPage(self.app, req)
    h.finish = Mock()
    h.get()
    self.assertEqual(len(h.finish.call_args_list), 1)

  def test_login_page(self):
    req = Mock(cookies=dict(), headers=dict(), arguments=dict(msg=['Message']))
    h = LoginPage(self.app, req)
    h.finish = Mock()
    h.get()
    self.assertEqual(len(h.finish.call_args_list), 1)

  def test_public_subpages(self):
    # these don't require datasets to render successfully
    req = Mock(cookies=dict(), headers=dict(), arguments=dict())
    for page_cls in (DatasetsPage, DataExplorerPage, BaselinePage,
                     PeakFitPage):
      h = page_cls(self.app, req)
      h.finish = Mock()
      h.get()
      self.assertEqual(len(h.finish.call_args_list), 1)

  def test_private_subpages(self):
    # these don't require datasets to render successfully
    req = Mock(headers=dict(), arguments=dict())
    for page_cls in (DatasetImportPage,):
      h = page_cls(self.app, req)
      h.is_private = False
      h.finish = Mock()
      h.get()
      self.assertEqual(len(h.finish.call_args_list), 1)


if __name__ == '__main__':
  unittest.main()
