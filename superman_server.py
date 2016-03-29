#!/usr/bin/env python
import logging
import os.path
import shutil
import time
import tornado.web
from argparse import ArgumentParser

from server import MatplotlibServer, load_datasets, all_routes, BaseHandler


def main():
  webserver_dir = os.path.dirname(__file__)
  darby_dir = os.path.join(webserver_dir, '..', 'darby_projects')
  msl_ccs_dir = '/srv/nfs/common/msl_data/ccs_data'
  ap = ArgumentParser()
  ap.add_argument('--port', type=int, default=54321, help='Port to listen on')
  ap.add_argument('--darby-dir', type=str, default=darby_dir,
                  help='Path to the darby_projects root (%(default)s)')
  ap.add_argument('--msl-ccs-dir', type=str, default=msl_ccs_dir,
                  help='Path to MSL CCS LIBS data (%(default)s)')
  ap.add_argument('--debug', action='store_true',
                  help='Start an IPython shell instead of starting the server.')
  ap.add_argument('--public', action='store_true',
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

  load_datasets(args.darby_dir, args.msl_ccs_dir, public=args.public)

  if args.debug:
    return debug()

  if args.public:
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
  from server.web_datasets import DATASETS
  import IPython
  IPython.embed(header='Note: Datasets are still loading asynchronously.')


if __name__ == '__main__':
  main()
