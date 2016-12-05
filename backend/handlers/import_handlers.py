from __future__ import absolute_import
import numpy as np
import os
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
    description = self.get_argument('desc')

    resample = (self.get_argument('lb', ''), self.get_argument('ub', ''),
                self.get_argument('step', ''))
    if not any(resample):
      resample = None

    if ds_kind not in DATASETS:
      return self.write_error(400, 'Invalid dataset kind.',
                              'Invalid ds_kind: %r', ds_kind)

    if ds_name in DATASETS[ds_kind]:
      return self.write_error(403, 'Dataset already exists.',
                              'ds import would clobber existing: %s [%s]',
                              ds_name, ds_kind)

    if not self.request.files or 'spectra' not in self.request.files:
      return self.write_error(400, 'No spectrum data uploaded.')

    meta_kwargs, meta_pkeys = self._load_metadata_csv()
    if meta_kwargs is None:
      return

    f, = self.request.files['spectra']
    fh = BytesIO(f['body'])
    if is_zipfile(fh):
      # interpret this as a ZIP of csv files
      fh.seek(0)
      success = self._traj_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys,
                              resample, description)
    else:
      # this is one single csv file with all spectra in it
      fh.seek(0)
      success = self._vector_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys,
                                resample, description)

    if success:
      self.write('/explorer?ds_kind=%s&ds_name=%s' % (
        ds_kind, url_escape(ds_name, plus=False)))

  def _traj_ds(self, fh, ds_name, ds_kind, meta_kwargs, meta_pkeys, resample,
               description):
    zf = ZipFile(fh)
    traj_data = {}
    for subfile in zf.infolist():
      if subfile.file_size <= 0:
        continue
      # ignore directory prefixes
      fname = os.path.basename(subfile.filename)
      # ignore hidden files
      if fname.startswith('.'):
        continue
      # read and wrap, because the ZipExtFile object isn't seekable
      sub_fh = BytesIO(zf.open(subfile).read())
      try:
        # TODO: ensure each traj has wavelengths in increasing order
        traj_data[fname] = parse_spectrum(sub_fh)
      except Exception as e:
        self.write_error(415, 'Unable to parse spectrum file: %s' % fname,
                         'bad spectrum subfile (%s): %s', fname, e)
        return False

    num_meta = len(meta_pkeys)
    num_traj = len(traj_data)

    if num_meta == 0:
      meta_pkeys = traj_data.keys()
    elif num_meta != num_traj:
      self.write_error(415, 'Failed: %d metadata entries for %d spectra' % (
          num_meta, num_traj))
      return False
    else:
      for pkey in meta_pkeys:
        if pkey not in traj_data:
          self.write_error(415, 'Failed: %r not in spectra.' % pkey)
          return False

    if resample is None:
      _load = _make_loader_function(description, meta_pkeys, traj_data,
                                    **meta_kwargs)
      WebTrajDataset(ds_name, ds_kind, _load)
    else:
      lb, ub, step = map(_maybe_float, resample)
      waves = [t[:,0] for t in traj_data.values()]
      if lb is None:
        lb = max(w[0] for w in waves)
      if ub is None:
        ub = min(w[-1] for w in waves)
      if step is None:
        step = min(np.diff(w).min() for w in waves)

      wave = np.arange(lb, ub + step/2, step, dtype=waves[0].dtype)
      spectra = np.zeros((len(waves), len(wave)), dtype=wave.dtype)
      for i, key in enumerate(meta_pkeys):
        traj = traj_data[key]
        spectra[i] = np.interp(wave, traj[:,0], traj[:,1])
      pkey = PrimaryKeyMetadata(meta_pkeys)

      _load = _make_loader_function(description, wave, spectra, pkey=pkey,
                                    **meta_kwargs)
      WebVectorDataset(ds_name, ds_kind, _load)

    return True

  def _vector_ds(self, fh, ds_name, ds_kind, meta_kwargs, meta_pkeys, resample,
                 description):
    try:
      pkey = np.array(next(fh).strip().split(',')[1:])
      data = np.genfromtxt(fh, dtype=np.float32, delimiter=',', unpack=True)
      wave = data[0]
      spectra = data[1:]
    except Exception as e:
      self.write_error(415, 'Unable to parse spectrum data CSV.',
                       'bad spectra file: %s', e)
      return False

    # cut out empty rows (where wave is NaN)
    mask = np.isfinite(wave)
    if np.count_nonzero(mask) != len(mask):
      wave = wave[mask]
      spectra = spectra[:, mask]

    if ds_kind == 'LIBS' and wave.shape != (6144,):
      self.write_error(415, 'Wrong number of channels for LIBS data.')
      return False

    if len(meta_pkeys) > 0 and not np.array_equal(meta_pkeys, pkey):
      if len(meta_pkeys) != len(pkey):
        self.write_error(415, 'Spectrum and metadata names mismatch.',
                         'wrong number of meta_pkeys for vector data')
        return False
      meta_order = np.argsort(meta_pkeys)
      data_order = np.argsort(pkey)
      if not np.array_equal(meta_pkeys[meta_order], pkey[data_order]):
        self.write_error(415, 'Spectrum and metadata names mismatch.')
        return False
      # convert data to meta order
      order = np.zeros_like(data_order)
      order[data_order[meta_order]] = np.arange(len(order))
      data = data[order]
      assert np.array_equal(meta_pkeys, pkey[order])

    try:
      pkey = PrimaryKeyMetadata(pkey)
    except AssertionError:  # XXX: convert this to a real error
      self.write_error(415, 'Primary keys not unique.')
      return False

    # make sure wave is in increasing order
    order = np.argsort(wave)
    if not np.array_equal(order, np.arange(len(wave))):
      wave = wave[order]
      spectra = spectra[:, order]

    if resample is not None:
      lb, ub, step = resample
      lb = _maybe_float(lb, wave[0])
      ub = _maybe_float(ub, wave[-1])
      step = _maybe_float(step)
      if step is not None:
        new_wave = np.arange(lb, ub + step/2, step, dtype=wave.dtype)
        new_spectra = np.zeros((len(spectra), len(new_wave)),
                               dtype=spectra.dtype)
        for i, y in enumerate(spectra):
          new_spectra[i] = np.interp(new_wave, wave, y)
        wave = new_wave
        spectra = new_spectra
      else:
        lb_idx = np.searchsorted(wave, lb)
        ub_idx = np.searchsorted(wave, ub, side='right')
        spectra = spectra[:, lb_idx:ub_idx]
        wave = wave[lb_idx:ub_idx]

    # async loading machinery automatically registers us with DATASETS
    _load = _make_loader_function(description, wave, spectra, pkey=pkey,
                                  **meta_kwargs)
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
      self.write_error(415, 'Unable to parse metadata CSV.',
                       'bad metadata file: %s', e)
      return None, None

    # get the actual meta names (dtype names are mangled)
    fh.seek(0)
    meta_names = next(fh).strip().split(',')[1:]
    meta_keys = meta.dtype.names[1:]

    meta_kwargs = {}
    for key, name in zip(meta_keys, meta_names):
      x = meta[key]
      if np.issubdtype(x.dtype, np.bool_):
        m = BooleanMetadata(x, display_name=name)
      elif np.issubdtype(x.dtype, np.number):
        m = NumericMetadata(x, display_name=name)
      else:
        m = LookupMetadata(x, display_name=name)
      meta_kwargs[key] = m
    return meta_kwargs, meta_pkeys


def _maybe_float(x, default=None):
  try:
    return float(x)
  except ValueError:
    return default


def _make_loader_function(desc, *args, **kwargs):
  def _load(ds):
    ds.set_data(*args, **kwargs)
    ds.is_public = False
    ds.user_added = True
    ds.description = desc
    return True
  return _load

routes = [
    (r'/_import', DatasetImportHandler),
]
