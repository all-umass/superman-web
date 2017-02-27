#!/usr/bin/env python
import base64
import logging
import os.path
import shutil
import time
import tornado.web
import uuid
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from backend import MatplotlibServer, all_routes, BaseHandler
from backend.web_datasets import (
    DATASETS, WebLIBSDataset, WebVectorDataset, WebTrajDataset)
import backend.web_datasets
import dataset_loaders


def main():
  webserver_dir = os.path.dirname(__file__)
  ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
  ap.add_argument('--config', type=open,
                  default=os.path.join(webserver_dir, 'config.yml'),
                  help='YAML file with configuration options.')
  ap.add_argument('--debug', action='store_true',
                  help='Start an IPython shell instead of starting the server.')
  args = ap.parse_args()
  config = yaml.safe_load(args.config)

  if not args.debug:
    logfile = os.path.join(webserver_dir,
                           config.get('logfile', 'logs/server.log'))
    if os.path.isfile(logfile):
      shutil.move(logfile, '%s.%d' % (logfile, time.time()))
    logging.basicConfig(filename=logfile,
                        format='[%(asctime)s] %(levelname)s: %(message)s',
                        filemode='w',
                        level=logging.INFO)

  ds_config = config.get('datasets', 'datasets.yml')
  password = config.get('password', None)
  with open(os.path.join(webserver_dir, ds_config)) as datasets_fh:
    load_datasets(datasets_fh, public_only=(password is None))

  if args.debug:
    return debug()

  if password is None:
    BaseHandler.is_public = True

  cookie_secret = config.get('cookie_secret', None)
  if cookie_secret is None:
    cookie_secret = base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)
    logging.info('Using fresh cookie_secret: %s', cookie_secret)

  logging.info('Starting server...')
  routes = all_routes + [
      (r'/(\S+\.(?:png|gif|css|js|ico))', tornado.web.StaticFileHandler,
       dict(path=os.path.join(webserver_dir, 'frontend', 'static')))]
  server = MatplotlibServer(
      routes, password=password, login_url=r'/login',
      template_path=os.path.join(webserver_dir, 'frontend', 'templates'),
      cookie_secret=cookie_secret)
  server.run_forever(int(config.get('port', 54321)))


def debug():
  import IPython
  IPython.embed(header=('Note: Datasets are still loading asynchronously.\n'
                        'They will appear in DATASETS: %s' % DATASETS))


def load_datasets(config_fh, public_only=False):
  config = yaml.safe_load(config_fh)

  for kind, entries in config.items():
    for name, info in entries.items():
      # skip this entry if it shouldn't be included
      is_public = info.get('public', True)
      if public_only and not is_public:
        continue

      if 'files' in info:
        files = info['files']
      else:
        files = [info['file']]

      if 'loader' in info:
        # look up the loader function from the module namespace
        loader_fn = getattr(dataset_loaders, info['loader'])
      else:
        # construct a loader from the meta_mapping and the default template
        meta_mapping = [(k, getattr(backend.web_datasets, cls), mname)
                        for k, cls, mname in info.get('metadata', [])]
        if info.get('vector', False):
          loader_fn = dataset_loaders._generic_vector_loader(meta_mapping)
        else:
          loader_fn = dataset_loaders._generic_traj_loader(meta_mapping)

      if kind == 'LIBS':
        ds = WebLIBSDataset(name, loader_fn, *files)
      elif info.get('vector', False):
        ds = WebVectorDataset(name, kind, loader_fn, *files)
      else:
        ds = WebTrajDataset(name, kind, loader_fn, *files)

      if 'description' in info:
        ds.description = info['description']
      if 'urls' in info:
        ds.urls = info['urls']
      ds.is_public = is_public


if __name__ == '__main__':
  main()
