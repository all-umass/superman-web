from __future__ import absolute_import
import ast
import logging
import numpy as np
from superman.file_io import parse_spectrum
from superman.preprocess import preprocess
from tornado.escape import json_encode

from .base import BaseHandler
from ..web_datasets import UploadedDataset, BytesIO


class SelectHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return
    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    ds = self.get_dataset(ds_kind, ds_name)
    if ds is None:
      logging.error("Failed to look up dataset: %s [%s]" % (ds_name, ds_kind))
      self.set_status(404)
      return
    name = self.get_argument('name', None)
    if name is None:
      idx = int(self.get_argument('idx'))
      if not (0 <= idx < ds.num_spectra()):
        logging.info('Index %d out of bounds in dataset %s', idx, ds)
        self.set_status(403)
        return
      name = 'Spectrum #%d' % idx
    else:
      idx = ds.pkey.key2index(name)
    fig_data.set_selected(ds.view(mask=[idx]), title=name)
    axlimits = fig_data.plot()
    return self.write(json_encode(axlimits))


class FilterHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return
    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    ds = self.get_dataset(ds_kind, ds_name)
    if ds is None:
      logging.error("Failed to look up dataset: %s [%s]" % (ds_name, ds_kind))
      self.set_status(404)
      return

    params = {k: ast.literal_eval(self.get_argument(k)) for k in ds.metadata}
    if ds.pkey is not None:
      params['pkey'] = ast.literal_eval(self.get_argument('pkey'))
    logging.info('Filtering %s with args: %s', ds, params)

    mask = ds.filter_metadata(params)
    fig_data.filter_mask = mask
    num_spectra = np.count_nonzero(mask)
    logging.info('Filtered to %d spectra', num_spectra)
    return self.write(str(num_spectra))


class UploadHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return
    if not self.request.files:
      logging.error('UploadHandler: no file uploaded')
      self.set_status(403)
      return
    f = self.request.files['query'][0]
    fname = f['filename']
    logging.info('Parsing file: %s', fname)
    fh = BytesIO(f['body'])
    try:
      query = parse_spectrum(fh)
    except Exception as e:
      logging.error('Failed to parse uploaded file: %s', e.message)
      self.set_status(415)
      return
    ds = UploadedDataset(fname, query.astype(np.float32, order='C'))
    fig_data.set_selected(ds.view(), title=fname)
    axlimits = fig_data.plot()
    return self.write(json_encode(axlimits))


class PreprocessHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return
    pp = self.get_argument('pp')
    fig_data.add_transform('pp', pp=pp)
    axlimits = fig_data.plot('pp')
    return self.write(json_encode(axlimits))


class ZoomFigureHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return
    xmin = float(self.get_argument('xmin'))
    xmax = float(self.get_argument('xmax'))
    ymin = float(self.get_argument('ymin'))
    ymax = float(self.get_argument('ymax'))
    ax = fig_data.figure.gca()
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    fig_data.manager.canvas.draw()


routes = [
    # For selecting a single spectrum from a dataset
    (r'/_select', SelectHandler),
    # For selecting >1 spectra from a dataset
    (r'/_filter', FilterHandler),
    (r'/_upload', UploadHandler),
    (r'/_pp', PreprocessHandler),
    (r'/_zoom', ZoomFigureHandler),
]
