from __future__ import absolute_import
import logging
import numpy as np
from io import BytesIO
from superman.file_io import parse_spectrum
from tornado.escape import url_escape
from zipfile import is_zipfile, ZipFile

from .base import BaseHandler
from ..web_datasets import (
    WebTrajDataset, WebVectorDataset, WebLIBSDataset, DATASETS,
    PrimaryKeyMetadata, NumericMetadata, BooleanMetadata, LookupMetadata)


class DatasetImportHandler(BaseHandler):
  def post(self):
    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')

    if ds_kind not in DATASETS:
      return self._raise_error(400, 'Invalid dataset kind.',
                               'Invalid ds_kind: %r' % ds_kind)

    if ds_name in DATASETS[ds_kind]:
      return self._raise_error(
          403, 'Dataset already exists.',
          'ds import would clobber existing: %s [%s]' % (ds_name, ds_kind))

    if not self.request.files or 'spectra' not in self.request.files:
      return self._raise_error(400, 'No spectrum data uploaded.')

    meta_kwargs, meta_pkeys = self._load_metadata_csv()
    if meta_kwargs is None:
      return

    f, = self.request.files['spectra']
    fh = BytesIO(f['body'])
    if is_zipfile(fh):
      # interpret this as a ZIP of csv files
      success = self._traj_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys)
    else:
      # this is one single csv file with all spectra in it
      success = self._vector_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys)

    if success:
      self.write('/explorer?ds_kind=%s&ds_name=%s' % (
        ds_kind, url_escape(ds_name, plus=False)))

  def _raise_error(self, status_code, user_msg, log_msg=None):
    if log_msg is None:
      logging.error(user_msg)
    else:
      logging.error(log_msg)
    self.set_status(status_code)
    self.finish(user_msg)
    return False

  def _traj_ds(self, fh, ds_name, ds_kind, meta_kwargs, meta_pkeys):
    zf = ZipFile(fh)
    traj_data = {}
    for subfile in zf.infolist():
      if subfile.file_size <= 0:
        continue
      fname = subfile.filename
      # read and wrap, because the ZipExtFile object isn't seekable
      sub_fh = BytesIO(zf.open(subfile).read())
      try:
        traj_data[fname] = parse_spectrum(sub_fh)
      except Exception as e:
        return self._raise_error(
            415, 'Unable to parse spectrum file: %s' % fname,
            'bad spectrum subfile (%s): %s' % (fname, e))

    if len(meta_pkeys) > 0 and len(meta_pkeys) != len(traj_data):
      return self._raise_error(415, 'Spectrum and metadata names mismatch.',
                               'wrong number of meta_pkeys for traj')
    for pkey in meta_pkeys:
      if pkey not in traj_data:
        return self._raise_error(415, 'Spectrum and metadata names mismatch.',
                                 'extra meta_pkey not in traj_data: %r' % pkey)
    if len(meta_pkeys) == 0:
      meta_pkeys = traj_data.keys()

    def _load(ds):
      ds.set_data(meta_pkeys, traj_data, **meta_kwargs)
      return True

    WebTrajDataset(ds_name, ds_kind, _load)
    return True

  def _vector_ds(self, fh, ds_name, ds_kind, meta_kwargs, meta_pkeys):
    try:
      data = np.genfromtxt(fh, dtype=np.float32, delimiter=',', names=True)
      wave = data[data.dtype.names[0]]
      spectra = data.view((np.float32, len(data.dtype.names))).T[1:]
    except Exception as e:
      return self._raise_error(415, 'Unable to parse spectrum data CSV.',
                               'bad spectra file: %s' % e)

    if ds_kind == 'LIBS' and wave.shape != (6144,):
      return self._raise_error(415, 'Wrong number of channels for LIBS data.')

    pkey = np.array(data.dtype.names[1:])
    if len(meta_pkeys) > 0 and not np.array_equal(meta_pkeys, pkey):
      return self._raise_error(415, 'Spectrum and metadata names mismatch.')

    # async loading machinery automatically registers us with DATASETS
    def _load(ds):
      ds.set_data(wave, spectra, pkey=PrimaryKeyMetadata(pkey), **meta_kwargs)
      return True

    if ds_kind == 'LIBS':
      WebLIBSDataset(ds_name, _load)
    else:
      WebVectorDataset(ds_name, ds_kind, _load)
    return True

  def _load_metadata_csv(self):
    # metadata is optional
    if 'metadata' not in self.request.files:
      return {}, []

    f, = self.request.files['metadata']
    fh = BytesIO(f['body'])
    try:
      meta = np.genfromtxt(fh, dtype=None, delimiter=',', names=True)
      meta_pkeys = np.array(meta[meta.dtype.names[0]])
    except Exception as e:
      self._raise_error(415, 'Unable to parse metadata CSV.',
                        'bad metadata file: %s' % e)
      return None, None

    meta_kwargs = {}
    for meta_name in meta.dtype.names[1:]:
      x = meta[meta_name]
      if np.issubdtype(x.dtype, np.bool_):
        m = BooleanMetadata(x)
      elif np.issubdtype(x.dtype, np.number):
        m = NumericMetadata(x)
      else:
        m = LookupMetadata(x)
      meta_kwargs[meta_name] = m
    return meta_kwargs, meta_pkeys

routes = [
    (r'/_import', DatasetImportHandler),
]
