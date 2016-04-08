#!/usr/bin/env python
import logging
import os.path
import shutil
import time
import tornado.web
import yaml
from argparse import ArgumentParser

from server import MatplotlibServer, all_routes, BaseHandler
from server.web_datasets import (
    DATASETS, WebLIBSDataset, WebVectorDataset, WebTrajDataset)
import server.web_datasets
import dataset_loaders


def main():
  webserver_dir = os.path.dirname(__file__)
  ap = ArgumentParser()
  ap.add_argument('--port', type=int, default=54321,
                  help='Port to listen on. [%(default)s]')
  ap.add_argument('--datasets', type=open,
                  default=os.path.join(webserver_dir, 'datasets.yml'),
                  help='YAML file specifying datasets to load. [%(default)s]')
  ap.add_argument('--debug', action='store_true',
                  help='Start an IPython shell instead of starting the server.')
  ap.add_argument('--public-only', action='store_true',
                  help='Exclude any non-public datasets and disable login.')
  args = ap.parse_args()

  if not args.debug:
    logfile = os.path.join(webserver_dir, 'logs/server.log')
    if os.path.isfile(logfile):
      shutil.move(logfile, '%s.%d' % (logfile, time.time()))
    logging.basicConfig(filename=logfile,
                        format='[%(asctime)s] %(levelname)s: %(message)s',
                        filemode='w',
                        level=logging.INFO)

  load_datasets(args.datasets, public_only=args.public_only)

  if args.debug:
    return debug()

  if args.public_only:
    BaseHandler.is_public = True

  logging.info('Starting server...')
  routes = all_routes + [
      (r'/(\w+\.(?:png|gif|css|js|ico))', tornado.web.StaticFileHandler,
       dict(path=os.path.join(webserver_dir, 'static')))]
  server = MatplotlibServer(
      *routes,
      template_path=os.path.join(webserver_dir, 'templates'),
      login_url=r'/login',
      cookie_secret="sdfjnwp9483nzjafagq582bqd")
  server.run_forever(args.port)


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

      description = info.get('description', 'No description provided.')

      if 'files' in info:
        files = info['files']
      else:
        files = [info['file']]

      if 'loader' in info:
        # look up the loader function from the module namespace
        loader_fn = getattr(dataset_loaders, info['loader'])
      else:
        # construct a loader from the meta_mapping and the default template
        meta_mapping = [(k, getattr(server.web_datasets, cls), mname)
                        for k, cls, mname in info.get('metadata', [])]
        if info.get('vector', False):
          loader_fn = dataset_loaders._generic_vector_loader(meta_mapping)
        else:
          loader_fn = dataset_loaders._generic_traj_loader(meta_mapping)

      if kind == 'LIBS':
        ds = WebLIBSDataset(name, description, loader_fn, *files)
      elif info.get('vector', False):
        ds = WebVectorDataset(name, kind, description, loader_fn, *files)
      else:
        ds = WebTrajDataset(name, kind, description, loader_fn, *files)
      ds.is_public = is_public


if __name__ == '__main__':
  main()
