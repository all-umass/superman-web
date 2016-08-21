from __future__ import absolute_import
import logging
import tornado.web

from ..web_datasets import DATASETS


class BaseHandler(tornado.web.RequestHandler):
  is_private = True

  def get_fig_data(self, fignum=None):
    if fignum is None:
      fignum = int(self.get_argument('fignum', 0))
    if fignum not in self.application.figure_data:
      logging.error('Invalid figure number: %d', fignum)
      return None
    return self.application.figure_data[fignum]

  def get_current_user(self):
    # Necessary for authentication,
    # see http://tornado.readthedocs.org/en/latest/guide/security.html
    if self.is_private:
      return self.get_secure_cookie('user')
    return True

  def _include_private_datasets(self):
    return self.is_private and self.get_current_user()

  def all_datasets(self):
    include_private = self._include_private_datasets()
    return [d for dd in DATASETS.values() for d in dd.values()
            if d.is_public or include_private]

  def get_dataset(self, ds_kind, ds_name):
    ds = DATASETS[ds_kind].get(ds_name, None)
    if ds is None or ds.is_public or self._include_private_datasets():
      return ds
    return None

  def dataset_kinds(self):
    return DATASETS.keys()
