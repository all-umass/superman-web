from __future__ import absolute_import
import logging
import tornado.web
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


routes = [
    (r'/_dataset_remover', RemovalHandler),
    (r'/refresh', RefreshHandler),
]
