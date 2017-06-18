from __future__ import absolute_import
import h5py
import logging
import numpy as np
import os
import pandas as pd
import time
import yaml
from io import BytesIO, StringIO
from six.moves import xrange
from superman.file_io import parse_spectrum
from threading import Thread
from tornado import gen
from tornado.escape import url_escape
from zipfile import is_zipfile, ZipFile

from .common import BaseHandler
from ..web_datasets import (
    UploadedSpectrumDataset,
    WebTrajDataset, WebVectorDataset, WebLIBSDataset, DATASETS,
    PrimaryKeyMetadata, NumericMetadata, BooleanMetadata, LookupMetadata)


class SpectrumUploadHandler(BaseHandler):
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


class DatasetUploadHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    description = self.get_argument('desc')

    resample = (self.get_argument('lb', ''), self.get_argument('ub', ''),
                self.get_argument('step', ''))
    if not any(resample):
      resample = None

    if ds_kind not in DATASETS:
      self.visible_error(400, 'Invalid dataset kind.', 'Invalid ds_kind: %r',
                         ds_kind)
      return

    if ds_name in DATASETS[ds_kind]:
      self.visible_error(403, 'Dataset already exists.',
                         'ds import would clobber existing: %s [%s]',
                         ds_name, ds_kind)
      return

    if not self.request.files or 'spectra' not in self.request.files:
      self.visible_error(400, 'No spectrum data uploaded.')
      return

    meta_file, = self.request.files.get('metadata', [None])
    spectra_file, = self.request.files['spectra']
    err = yield gen.Task(_async_ds_upload, meta_file, spectra_file, ds_name,
                         ds_kind, resample, description)
    if err:
      self.visible_error(*err)
      return

    # Return a link to the new dataset to signal the upload succeeded.
    self.write('/explorer?ds_kind=%s&ds_name=%s' % (
      ds_kind, url_escape(ds_name, plus=False)))

    # Kick off a background thread to save this new dataset to disk.
    t = Thread(target=_save_ds, args=(ds_kind, ds_name))
    t.daemon = True
    t.start()


def _async_ds_upload(meta_file, spectra_file, ds_name, ds_kind, resample,
                     description, callback=None):
  def helper():
    meta_kwargs, meta_pkeys, err = _load_metadata_csv(meta_file)
    if err is None:
      fh = BytesIO(spectra_file['body'])
      if is_zipfile(fh):
        # interpret this as a ZIP of csv files
        fh.seek(0)
        err = _traj_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys,
                       resample, description)
      else:
        # this is one single csv file with all spectra in it
        fh.seek(0)
        err = _vector_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys,
                         resample, description)
    callback(err)

  t = Thread(target=helper)
  t.daemon = True
  t.start()


def _load_metadata_csv(f=None):
  # metadata is optional
  if f is None:
    return {}, [], None

  fh = BytesIO(f['body'])
  try:
    meta = pd.read_csv(fh)
  except Exception:
    logging.exception('Bad metadata file')
    return None, None, (415, 'Unable to parse metadata CSV.')

  if meta.columns[0] != 'pkey':
    return None, None, (415, 'Metadata CSV must start with "pkey" column.')

  meta_kwargs = {}
  for i, name in enumerate(meta.columns[1:]):
    x = meta[name].values
    if np.issubdtype(x.dtype, np.bool_):
      m = BooleanMetadata(x, display_name=name)
    elif np.issubdtype(x.dtype, np.number):
      m = NumericMetadata(x, display_name=name)
    else:
      m = LookupMetadata(x, display_name=name)
    # use a JS-friendly string key
    meta_kwargs['k%d' % i] = m

  # make sure there's no whitespace sticking to the pkeys
  meta_pkeys = np.array(meta.pkey.values, dtype='U', copy=False)
  meta_pkeys = np.char.strip(meta_pkeys)
  return meta_kwargs, meta_pkeys, None


def _traj_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys, resample,
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
    except Exception:
      logging.exception('bad spectrum subfile: ' + fname)
      return (415, 'Unable to parse spectrum file: %s' % fname)

  num_meta = len(meta_pkeys)
  num_traj = len(traj_data)

  if num_meta == 0:
    meta_pkeys = traj_data.keys()
  elif num_meta != num_traj:
    return (415, 'Failed: %d metadata entries for %d spectra' % (num_meta,
                                                                 num_traj))
  else:
    for pkey in meta_pkeys:
      if pkey not in traj_data:
        return (415, 'Failed: %r not in spectra.' % pkey)

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

  return None


def _vector_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys, resample,
               description):
  try:
    pkey = np.array(next(fh).strip().split(b',')[1:])
    data = np.genfromtxt(fh, dtype=np.float32, delimiter=b',', unpack=True)
    wave = data[0]
    spectra = data[1:]
  except Exception:
    logging.exception('Bad spectra file.')
    return visible_error(415, 'Unable to parse spectrum data CSV.')

  # cut out empty columns (where pkey is '')
  mask = pkey != b''
  if not mask.all():
    pkey = pkey[mask]
    spectra = spectra[mask]

  # cut out empty rows (where wave is NaN)
  mask = np.isfinite(wave)
  if not mask.all():
    wave = wave[mask]
    spectra = spectra[:, mask]

  if ds_kind == 'LIBS' and wave.shape[0] not in (6144, 6143, 5485):
    return (415, 'Wrong number of channels for LIBS data: %d.' % wave.shape[0])

  # make sure there's no whitespace sticking to the pkeys
  pkey = np.char.strip(pkey)

  if len(meta_pkeys) > 0 and not np.array_equal(meta_pkeys, pkey):
    if len(meta_pkeys) != len(pkey):
      return (415, 'Spectrum and metadata names mismatch.',
                   'wrong number of meta_pkeys for vector data')
    meta_order = np.argsort(meta_pkeys)
    data_order = np.argsort(pkey)
    if not np.array_equal(meta_pkeys[meta_order], pkey[data_order]):
      return (415, 'Spectrum and metadata names mismatch.')
    # convert data to meta order
    order = np.zeros_like(data_order)
    order[data_order[meta_order]] = np.arange(len(order))
    data = data[order]
    assert np.array_equal(meta_pkeys, pkey[order])

  try:
    pkey = PrimaryKeyMetadata(pkey)
  except AssertionError:  # XXX: convert this to a real error
    return (415, 'Primary keys not unique.')

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
  return None


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


def _save_ds(ds_kind, ds_name):
  # Wait for the new dataset to finish registering.
  time.sleep(1)
  for _ in xrange(60):
    if ds_name in DATASETS[ds_kind]:
      break
    logging.info('Waiting for %s [%s] to register...', ds_name, ds_kind)
    time.sleep(1)

  # Save the new dataset to disk as a canonical hdf5.
  ds = DATASETS[ds_kind][ds_name]
  # XXX: this path manipulation is pretty hacky
  outdir = os.path.normpath(os.path.join(os.path.dirname(__file__),
                                         '../../logs'))
  outname = os.path.join(outdir, '%s_%s.hdf5' % (ds_kind,
                                                 ds_name.replace(' ', '_')))
  logging.info('Writing %s to disk: %s', ds, outname)

  entry = dict(vector=isinstance(ds, WebTrajDataset),
               file=os.path.abspath(outname),
               description=ds.description,
               public=ds.is_public,
               metadata=[])
  # TODO: move this logic to superman.dataset
  with h5py.File(outname, 'w') as fh:
    if isinstance(ds, WebTrajDataset):
      for key, traj in ds.traj.items():
        fh['/spectra/'+key] = traj
    else:
      fh['/spectra'] = ds.intensities
      fh['/meta/waves'] = ds.bands
    if ds.pkey is not None:
      fh['/meta/pkey'] = ds.pkey.keys
      entry['metadata'].append(('pkey', 'PrimaryKeyMetadata', None))
    for key, m in ds.metadata.items():
      try:
        arr = m.get_array()
      except:
        logging.exception('Failed to get_array for %s /meta/%s', ds, key)
      else:
        fh['/meta/'+key] = np.array(arr)
        entry['metadata'].append((key, str(type(m)), m.display_name(key)))
  # Clean up if no metadata was added.
  if not entry['metadata']:
    del entry['metadata']

  # Update the user-uploaded dataset config with the new dataset.
  config_path = os.path.join(outdir, 'user_uploads.yml')
  if os.path.exists(config_path):
    config = yaml.safe_load(config_path)
  else:
    config = {}
  if ds_kind not in config:
    config[ds_kind] = {ds_name: entry}
  else:
    config[ds_kind][ds_name] = entry
  with open(config_path, 'w') as fh:
    yaml.dump(config, fh)


routes = [
    (r'/_upload_spectrum', SpectrumUploadHandler),
    (r'/_upload_dataset', DatasetUploadHandler),
]
