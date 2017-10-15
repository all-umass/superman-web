from __future__ import absolute_import, print_function, division
import logging
import os
import tornado.web
import yaml
from tornado import gen
from threading import Thread

from .common import BaseHandler
from ..web_datasets import DATASETS


class RefreshHandler(BaseHandler):
  @tornado.web.authenticated
  def get(self):
    self.write('''<form action='refresh' method='POST'>
                  <input type='submit' value='Refresh Data'>
                  </form>''')

  @gen.coroutine
  def post(self):
    logging.info('Refreshing datasets')
    for ds in self.all_datasets():
      yield gen.Task(RefreshHandler._reload, ds)
    self.redirect('/datasets')

  @staticmethod
  def _reload(ds, callback=None):
    t = Thread(target=lambda: callback(ds.reload()))
    t.daemon = True
    t.start()


class RemovalHandler(BaseHandler):
  def post(self):
    ds = self.request_one_ds('kind', 'name')
    if not ds.user_added:
      return self.visible_error(403, 'Cannot remove this dataset.')
    logging.info('Removing user-added dataset: %s', ds)
    del DATASETS[ds.kind][ds.name]
    self.redirect('/datasets')
    # Remove the dataset from user-uploaded files.
    config_path = os.path.join(os.path.dirname(__file__),
                               '../../uploads/user_data.yml')
    if os.path.exists(config_path):
      config = yaml.safe_load(open(config_path))
      entry = config[ds.kind].pop(ds.name)
      os.remove(entry['file'])
      yaml.safe_dump(config, open(config_path, 'w'), allow_unicode=True)


routes = [
    (r'/_remove_dataset', RemovalHandler),
    (r'/refresh', RefreshHandler),
]
