# -*- coding: utf-8 -*-
import h5py
import logging
import pandas as pd
import numpy as np
import os.path
import re
from six.moves import xrange

from superman.dataset import (
    NumericMetadata, BooleanMetadata, PrimaryKeyMetadata, LookupMetadata,
    CompositionMetadata, TagMetadata)


def _generic_traj_loader(meta_mapping):
  """Creates a loader function for a standard HDF5 file representing a
  trajectory dataset. The HDF5 structure is expected to be:
   - /meta/pkey : an array of keys, used to address individual spectra
   - /spectra/[pkey] : a (n,2) trajectory spectrum
   - /meta/foobar : (optional) metadata, specified by the meta_mapping
  """
  def _load(ds, filepath):
    data = _try_load(filepath, str(ds))
    if data is None:
      return False
    meta = data['/meta']
    kwargs = {}
    for key, cls, display_name in meta_mapping:
      if key not in meta:
        continue
      safe_key = re.sub(r'[^a-z0-9_-]', '', key, flags=re.I)
      kwargs[safe_key] = cls(meta[key], display_name)
    ds.set_data(meta['pkey'], data['/spectra'], **kwargs)
    return True
  return _load


def _generic_vector_loader(meta_mapping):
  """Creates a loader function for a standard HDF5 file representing a
  vector dataset. The HDF5 structure is expected to be:
   - /meta/waves : length-d array of wavelengths
   - /spectra : (n,d) array of spectra
   - /meta/foobar : (optional) metadata, specified by the meta_mapping
   - /composition/[name] : (optional) composition metadata
  """
  def _load(ds, filepath):
    data = _try_load(filepath, str(ds))
    if data is None:
      return False
    meta = data['/meta']
    kwargs = {}
    for key, cls, display_name in meta_mapping:
      if key not in meta:
        continue
      if cls is PrimaryKeyMetadata:
        kwargs['pkey'] = cls(meta[key])
      else:
        kwargs[key] = cls(meta[key], display_name=display_name)
    if '/composition' in data:
      comp_meta = {name: NumericMetadata(arr, display_name=name) for name, arr
                   in data['/composition'].items()}
      kwargs['Composition'] = CompositionMetadata(comp_meta)
    ds.set_data(meta['waves'], data['/spectra'], **kwargs)
    return True
  return _load


def load_mhc_multipower(ds, filepath, with_blr=False):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  kind = 'preprocessed' if with_blr else 'formatted'
  s = data['/spectra/'+kind]
  spectra = np.vstack((s['low_power'], s['med_power'], s['high_power']))
  comps = data['/composition']
  meta = data['/meta']
  names = np.asarray(meta['names'])
  idx = np.argsort(names)
  powers = ('3.2', '5', '7')
  pkey = ['%s @ %s%%' % (n, p) for p in powers for n in names]
  comp_meta = {k: NumericMetadata(comps[k], display_name=k, repeats=3)
               for k in comps.keys()}
  ds.set_data(meta['waves'], spectra,
              pkey=PrimaryKeyMetadata(pkey),
              in_lanl=BooleanMetadata(meta['in_lanl'], 'In LANL', repeats=3),
              name=LookupMetadata(names[idx], 'Name', labels=np.tile(idx, 3)),
              power=LookupMetadata(powers, 'Laser Power',
                                   labels=np.repeat([0,1,2], len(names))),
              comp=CompositionMetadata(comp_meta, 'Composition'))
  return True


def load_mars_big(ds, msl_ccs_dir, pred_file, mixed_pred_file,
                  moc_pred_file, dust_pred_file):
  logging.info('Loading Mars (big) LIBS data...')
  file_pattern = os.path.join(msl_ccs_dir, 'ccs.%03d.hdf5')
  meta_file = os.path.join(msl_ccs_dir, 'ccs_meta.npz')
  chan_file = os.path.join(msl_ccs_dir, 'ccs_channels.npy')

  try:
    import dask.array as da
  except ImportError as e:
    logging.warning('Failed to load Mars (big) LIBS data!')
    logging.warning(str(e))
    return False

  try:
    f = h5py.File(file_pattern, mode='r', driver='family', libver='latest')
    spectra = da.from_array(f['/spectra'], chunks=10000, name='mars_big')
    meta = np.load(meta_file)
    bands = np.load(chan_file)
    comps = np.load(pred_file)
    mixed_comps = np.load(mixed_pred_file)
    moc_comps = np.load(moc_pred_file)
    # TODO: store this in a better way
    is_dust = np.load(dust_pred_file)['Is dust?'].astype(bool)
  except IOError as e:
    logging.warning('Failed to load Mars (big) LIBS data!')
    logging.warning(str(e))
    return False

  logging.info('Making Mars (big) metadata...')
  # Predicted compositions
  pred_comps = {elem: NumericMetadata(comps[elem], display_name=elem)
                for elem in comps.files}
  mixed_pred_comps = {elem: NumericMetadata(mixed_comps[elem],display_name=elem)
                      for elem in mixed_comps.files}
  moc_pred_comps = {elem: NumericMetadata(moc_comps[elem], display_name=elem)
                    for elem in moc_comps.files}

  # Cal target uct_mask
  uniq_targets, target_labels = np.unique(meta['names'], return_inverse=True)
  uct_mask = np.array([n.startswith('Cal Target') or n.startswith('Cal_Target')
                       for n in uniq_targets], dtype=bool)
  ct_mask = np.in1d(target_labels, uct_mask.nonzero()[0])

  # Location numbers (per target)
  uniq_ids, id_labels = np.unique(meta['ids'], return_inverse=True)
  locations = np.zeros_like(target_labels)
  for tl in xrange(len(uniq_targets)):
    mask = target_labels == tl
    locations[mask] = 1 + np.unique(id_labels[mask], return_inverse=True)[1]

  ds.set_data(
      bands, spectra,
      target=LookupMetadata(uniq_targets, 'Target Name', labels=target_labels),
      dist=NumericMetadata(meta['distances'], 0.1, 'Standoff Distance'),
      sol=NumericMetadata(meta['sols'], 1, 'Sol #'),
      shot=NumericMetadata(meta['numbers'], 1, 'Shot #'),
      Autofocus=BooleanMetadata(meta['foci']),
      id=LookupMetadata(uniq_ids, 'Instrument ID', labels=id_labels),
      umass_pred=CompositionMetadata(pred_comps, 'LANL PLS'),
      mixed_pred=CompositionMetadata(mixed_pred_comps, 'MHC/LANL/Mars PLS'),
      moc_pred=CompositionMetadata(moc_pred_comps, 'MOC (ChemCam Team)'),
      dust=BooleanMetadata(is_dust, 'Dust? (predicted)'),
      caltarget=BooleanMetadata(ct_mask, 'Cal Target?'),
      loc=NumericMetadata(locations, 1, 'Location #'))

  logging.info('Finished Mars (big) setup.')
  return True


def load_usda(ds, filepath):
  usda = _try_load(filepath, str(ds))
  if usda is None:
    return False
  filenames, keys = usda['key'].T
  bands = usda['data_names']
  # XXX: There are 183 different compositions here, but I'm just looking at
  #      the first three. If we want to add more, do so here.
  comp_names = [n.split(',',1)[0] for n in usda['target_names'][:3]]
  comp_vals = usda['target'][:,:3].T
  comp_meta = {name: NumericMetadata(arr, display_name=name) for name, arr
               in zip(comp_names, comp_vals)}
  ds.set_data(bands, usda['data'], pkey=PrimaryKeyMetadata(filenames),
              key=LookupMetadata(keys, 'Natural Key'),
              Composition=CompositionMetadata(comp_meta))
  return True


def load_corn(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  # band info given by: http://www.eigenvector.com/data/Corn/index.html
  bands = np.arange(1100, 2500, 2).astype(float)
  instrument_names = ['m5', 'mp5', 'mp6']
  spectra = np.vstack([data['/spectra/' + n] for n in instrument_names])
  metadata = {key: NumericMetadata(val, repeats=3)
              for key, val in data['/meta'].items()}
  metadata['inst'] = LookupMetadata(instrument_names,
                                    display_name='Spectrometer',
                                    labels=np.repeat(np.arange(3), 80))
  ds.set_data(bands, spectra, **metadata)
  return True


def load_mhc_hydrogen(ds, filepath):
  hdf5 = _try_load(filepath, str(ds))
  if hdf5 is None:
    return False
  powers = hdf5['/meta/powers']
  names = hdf5['/meta/names']
  pkey = ['%s - %s%%' % (name, power) for name, power in zip(names, powers)]
  bands = hdf5['/meta/waves']
  comp = {'H2O': NumericMetadata(hdf5['/composition/H2O'], display_name='H2O')}
  ds.set_data(bands, hdf5['/spectra'],
              pkey=PrimaryKeyMetadata(pkey),
              Composition=CompositionMetadata(comp),
              names=LookupMetadata(names, 'Sample Name'),
              powers=LookupMetadata(powers, 'Laser Power'))
  return True


def load_mhc_libs(ds, data_dir, master_file):
  logging.info('Loading MHC LIBS data...')
  data_file = os.path.join(data_dir, 'prepro_no_blr.%03d.hdf5')
  chan_file = os.path.join(data_dir, 'prepro_channels.npy')
  try:
    hdf5 = h5py.File(data_file, driver='family', mode='r')
    meta = np.load(master_file)
    bands = np.load(chan_file)
  except IOError as e:
    logging.warning('Failed to load data in %s!' % data_dir)
    logging.warning(str(e))
    return None
  logging.info('Making MHC LIBS metadata...')
  projects = [set(filter(None, p.split(','))) for p in meta['Projects']]
  compositions = {}
  for key in meta.files:
    if key.startswith('e_'):
      vals = np.array(meta[key], dtype=float, copy=False)
      if np.nanmin(vals) < np.nanmax(vals):
        elem = key.lstrip('e_').replace('*', '')
        compositions[elem] = NumericMetadata(vals, display_name=elem)
  # date = np.array(pd.to_datetime(meta['Date']), dtype=np.datetime64)
  ds.set_data(bands, hdf5['/spectra'],
              Composition=CompositionMetadata(compositions),
              samples=LookupMetadata(meta['Sample'], 'Sample Name'),
              carousels=LookupMetadata(meta['Carousel'], 'Carousel #'),
              locations=LookupMetadata(meta['Location'], 'Location #'),
              shots=NumericMetadata(meta['Number'], 1, 'Shot #'),
              targets=LookupMetadata(meta['Target'], 'Target Name'),
              powers=LookupMetadata(meta['LaserAttenuation'], 'Laser Power'),
              projects=TagMetadata(projects, 'Project'),
              # TODO: make a DateTimeMetadata to handle this
              # date=NumericMetadata(date, 'Acquisition Time'),
              # NOTE: These have only one unique value for now.
              # atmospheres=LookupMetadata(meta['Atmosphere'], 'Atmosphere'),
              # dists=LookupMetadata(meta['DistToTarget'],'Distance to Target')
              )
  logging.info('Finished MHC LIBS setup.')
  return True


def load_mhc_raman(ds, data_dir, meta_file):
  logging.info('Loading MHC Raman data...')
  data_file = os.path.join(data_dir, 'raman.%03d.hdf5')
  try:
    hdf5 = h5py.File(data_file, driver='family', mode='r')
    meta = np.load(meta_file)
  except IOError as e:
    logging.warning('Failed to load data in %s!' % data_dir)
    logging.warning(str(e))
    return None
  pkey = meta['spectrum_number']
  ds.set_data(pkey, hdf5['/spectra'],
              vial=LookupMetadata(meta['vial_name'], 'Vial Name'),
              Instrument=LookupMetadata(meta['instrument']),
              Project=LookupMetadata(meta['project'])
              )
  return True


def _try_load(filepath, data_name):
  logging.info('Loading %s data...' % data_name)
  if not os.path.exists(filepath):
    logging.warning('Data file for %s not found.' % data_name)
    return None

  load_fn = (np.load if filepath.endswith('.npz') else
             lambda f: h5py.File(f, mode='r'))
  try:
    return load_fn(filepath)
  except IOError as e:
    logging.warning('Failed to load %s data!' % data_name)
    logging.warning(str(e))
    return None
