# -*- coding: utf-8 -*-
import h5py
import logging
import numpy as np
import os.path
from six.moves import xrange

from server.web_datasets import (
    LookupMetadata, NumericMetadata, BooleanMetadata, PrimaryKeyMetadata,
    CompositionMetadata
)


def _generic_traj_loader(*meta_mapping):
  def _load(ds, filepath):
    data = _try_load(filepath, str(ds))
    if data is None:
      return False
    meta = data['/meta']
    kwargs = {}
    for key, cls, display_name in meta_mapping:
      kwargs[key] = cls(meta[key], display_name)
    ds.set_data(meta['pkey'], data['/spectra'], **kwargs)
    return True
  return _load

load_no_meta = _generic_traj_loader()
load_names = _generic_traj_loader(('names', LookupMetadata, 'Name'))
load_usgs = _generic_traj_loader(('ids', LookupMetadata, 'Sample ID'))
load_ucl = _generic_traj_loader(
    ('color', LookupMetadata, 'Color'),
    ('power', LookupMetadata, 'Laser Power'),
    ('wavelength', LookupMetadata, 'Laser Frequency'))
load_irug = _generic_traj_loader(
    ('ids', LookupMetadata, 'ID'),
    ('names', LookupMetadata, 'Name'),
    ('materials', LookupMetadata, 'Material'))
load_synth_pyrox = _generic_traj_loader(
    ('mineral', LookupMetadata, 'Mineral'),
    ('formula', LookupMetadata, 'Formula'),
    ('dana', LookupMetadata, 'Dana Number'),
    ('source', LookupMetadata, 'Source'))
load_dyar96 = _generic_traj_loader(
    ('laser', LookupMetadata, 'Laser Type'),
    ('id', LookupMetadata, 'ID'),
    ('mineral', LookupMetadata, 'Mineral'))
load_mineral_mixes = _generic_traj_loader(
    ('sample', LookupMetadata, 'Sample ID'),
    ('laser', LookupMetadata, 'Laser Type'),
    ('mineral_1', LookupMetadata, 'Mineral A'),
    ('mineral_2', LookupMetadata, 'Mineral B'),
    ('ratio', LookupMetadata, 'Mix Ratio (A:B)'),
    ('grain_size', LookupMetadata, 'Grain Size (µm)'))


def load_rruff(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  meta = data['/meta']
  names = meta['sample']
  if 'species' in meta:
    species = meta['species']
  else:
    species = [n.rsplit('-',1)[0] for n in names]
  metadata = dict(
      minerals=LookupMetadata(species, 'Species'),
      rruff_ids=LookupMetadata(meta['rruff_id'], 'RRUFF ID'))
  if 'laser' in meta:
    metadata['lasers'] = LookupMetadata(meta['laser'], 'Laser')
  if 'dana' in meta:
    metadata['dana'] = LookupMetadata(meta['dana/species'], 'Dana Number')
  ds.set_data(names, data['/spectra'], **metadata)
  return True


def load_lanl_generic(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False

  comp_meta = {name: NumericMetadata(arr, display_name=name) for name, arr
               in data['/composition'].items()}
  ds.set_data(data['/meta/waves'], data['/spectra'],
              pkey=PrimaryKeyMetadata(data['/meta/names']),
              Composition=CompositionMetadata(comp_meta))
  return True


def load_lanl_caltargets(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False

  ds.set_data(data['/meta/waves'], data['/spectra'],
              names=LookupMetadata(data['/meta/targets'], 'Target Names'))
  return True


def load_mhc_caltargets(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  meta = data['meta']
  ds.set_data(meta['waves'], data['/spectra'],
              name=LookupMetadata(meta['names'], 'Mineral Name'),
              laser=LookupMetadata(meta['powers'], 'Laser Power'),
              time=LookupMetadata(meta['integration_times'],
                                  'Integration Time'))
  return True


def load_doped(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False

  comps = data['/composition']
  meta = data['/meta']
  kwargs = {
      'power': LookupMetadata(meta['power'], 'Power'),
      'matrix': LookupMetadata(meta['matrix'], 'Matrix'),
      'mix': LookupMetadata(meta['mix_no'], 'Mix #'),
      'comp': CompositionMetadata({k: NumericMetadata(comps[k], display_name=k)
                                   for k in comps.keys()}, 'Compositions'),
  }
  # The doped_full dataset has some extra metadata
  if 'pkey' in meta:
    kwargs['pkey'] = PrimaryKeyMetadata(meta['pkey'])
  if 'target' in meta:
    kwargs['target'] = LookupMetadata(meta['target'], 'Target')
  if 'concentration' in meta:
    kwargs['conc'] = LookupMetadata(meta['concentration'], 'Concentration')
  if 'dopant' in meta:
    kwargs['dopant'] = LookupMetadata(meta['dopant'], 'Dopant')

  ds.set_data(meta['waves'], data['/spectra'], **kwargs)
  return True


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
              power=LookupMetadata(powers, 'Power',
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


def load_silicate_glass(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['keys'], data,
              fe3=NumericMetadata(data['fe3'], display_name='% Fe3+'),
              Formula=LookupMetadata(data['formula']))
  return True


def load_amphibole(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  orient = data['orientation']
  names = data['names']
  pkey = ['-'.join(filter(None,x)) for x in zip(names, orient)]
  ds.set_data(data['bands'], data['spectra'],
              pkey=PrimaryKeyMetadata(pkey),
              fe3=NumericMetadata(data['fe3'], display_name='% Fe3+'),
              Orientation=LookupMetadata(orient),
              name=LookupMetadata(names, 'Sample Name'))
  return True


def load_garnet(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['bands'], data['spectra'],
              pkey=PrimaryKeyMetadata(data['names']),
              fe3=NumericMetadata(data['fe3'], display_name='% Fe3+'))
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
