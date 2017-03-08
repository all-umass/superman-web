from __future__ import absolute_import
import ast
import logging
import numpy as np
from io import BytesIO, StringIO
from superman.file_io import parse_spectrum

from .common import BaseHandler
from ..web_datasets import UploadedSpectrumDataset


class SelectHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    ds = self.get_dataset(ds_kind, ds_name)
    if ds is None:
      msg = "Can't find dataset: %s [%s]" % (ds_name, ds_kind)
      return self.visible_error(404, msg)

    name = self.get_argument('name', None)
    if name is None:
      idx = int(self.get_argument('idx'))
      if not (0 <= idx < ds.num_spectra()):
        return self.visible_error(403, 'Invalid spectrum number.',
                                  'Index %d out of bounds in dataset %s',
                                  idx, ds)
      name = 'Spectrum %d' % idx
    else:
      # XXX: hack to match dtype of pkey
      name = np.array(name, dtype=ds.pkey.keys.dtype).item()
      idx = ds.pkey.key2index(name)

    fig_data.set_selected(ds.view(mask=[idx]), title=str(name))
    axlimits = fig_data.plot()
    return self.write_json(axlimits)


class FilterHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    ds = self.get_dataset(ds_kind, ds_name)
    if ds is None:
      msg = "Can't find dataset: %s [%s]" % (ds_name, ds_kind)
      return self.visible_error(404, msg)

    params = {k: ast.literal_eval(self.get_argument(k)) for k in ds.metadata}
    if ds.pkey is not None:
      params['pkey'] = ast.literal_eval(self.get_argument('pkey'))
    logging.info('Filtering %s with args: %s', ds, params)

    mask = ds.filter_metadata(params)
    fig_data.filter_mask[ds] = mask
    num_spectra = np.count_nonzero(mask)
    logging.info('Filtered to %d spectra', num_spectra)

    # blow away any cached explorer data
    fig_data.clear_explorer_cache()

    return self.write(str(num_spectra))


class UploadHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    if not self.request.files:
      return self.visible_error(403, 'No file uploaded.')

    f = self.request.files['query'][0]
    fname = f['filename']
    logging.info('Parsing file: %s', fname)
    fh = BytesIO(f['body'])
    try:
      query = parse_spectrum(fh)
    except Exception:
      try:
        fh = StringIO(f['body'].decode('utf-8', 'ignore'), newline=None)
        query = parse_spectrum(fh)
      except Exception:
        logging.exception('Spectrum parse failed.')
        # XXX: save failed uploads for debugging purposes
        open('logs/badupload-'+fname, 'w').write(f['body'])
        return self.visible_error(415, 'Spectrum upload failed.')
    ds = UploadedSpectrumDataset(fname, query)
    fig_data.set_selected(ds.view(), title=fname)
    axlimits = fig_data.plot()
    return self.write_json(axlimits)


class PreprocessHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    pp = self.get_argument('pp')
    fig_data.add_transform('pp', pp=pp)
    axlimits = fig_data.plot('pp')
    return self.write_json(axlimits)


class ZoomFigureHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    xmin = float(self.get_argument('xmin'))
    xmax = float(self.get_argument('xmax'))
    ymin = float(self.get_argument('ymin'))
    ymax = float(self.get_argument('ymax'))
    ax = fig_data.figure.axes[0]
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
