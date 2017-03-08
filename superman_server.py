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

from backend import MatplotlibServer, all_routes
from backend.handlers.common import BaseHandler
from backend.web_datasets import DATASETS
from backend.dataset_loaders import load_datasets

# User-supplied dataset loader functions
import custom_datasets


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
    load_datasets(datasets_fh, custom_datasets, public_only=(password is None))

  if args.debug:
    return debug()

  if password is None:
    BaseHandler.is_public = True

  cookie_secret = config.get('cookie_secret', None)
  if cookie_secret is None:
    cookie_secret = base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)
    logging.info('Using fresh cookie_secret: %s', cookie_secret)

  logging.info('Starting server...')
  server = MatplotlibServer(
      all_routes, password=password, login_url=r'/login',
      template_path=os.path.join(webserver_dir, 'frontend', 'templates'),
      static_path=os.path.join(webserver_dir, 'frontend', 'static'),
      cookie_secret=cookie_secret)
  server.run_forever(int(config.get('port', 54321)))


def debug():
  import IPython
  IPython.embed(header=('Note: Datasets are still loading asynchronously.\n'
                        'They will appear in DATASETS: %s' % DATASETS))


if __name__ == '__main__':
  main()
