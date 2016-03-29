# -*- coding: utf-8 -*-
from __future__ import absolute_import
import h5py
import logging
import numpy as np
import os.path
from glob import glob
from six.moves import xrange

from .web_datasets import (
    WebTrajDataset, WebVectorDataset, WebLIBSDataset,
    LookupMetadata, NumericMetadata, BooleanMetadata, PrimaryKeyMetadata,
    CompositionMetadata
)


def load_datasets(darby_dir, msl_ccs_dir, public=False):
  raman_dir = os.path.join(darby_dir, 'raman', 'data')
  ftir_dir = os.path.join(darby_dir, 'ftir', 'data')
  mars_dir = os.path.join(darby_dir, 'mars', 'data')
  xray_dir = os.path.join(darby_dir, 'xray', 'data')
  libs_dir = os.path.join(darby_dir, 'libs', 'data')

  WebTrajDataset('RRUFF', 'Raman', _load_rruff,
                 os.path.join(raman_dir, 'rruff-spectra.hdf5'))
  WebTrajDataset('RRUFF (raw)', 'Raman', _load_rruff,
                 os.path.join(raman_dir, 'rruff-spectra-raw.hdf5'))
  WebTrajDataset('RRUFF', 'FTIR', _load_rruff,
                 os.path.join(ftir_dir, 'rruff-spectra.hdf5'))
  WebTrajDataset('RRUFF', 'XRD', _load_rruff,
                 os.path.join(xray_dir, 'rruff-spectra.hdf5'))
  WebTrajDataset('RRUFF (raw)', 'XRD', _load_rruff,
                 os.path.join(xray_dir, 'rruff-spectra-raw.hdf5'))
  WebTrajDataset('IRUG', 'Raman', _load_irug,
                 os.path.join(raman_dir, 'irug.npz'))
  WebTrajDataset('IRUG', 'FTIR', _load_irug, os.path.join(ftir_dir, 'irug.npz'))
  WebTrajDataset('USGS', 'FTIR', _load_usgs, os.path.join(ftir_dir, 'USGS.npz'))
  WebTrajDataset('Parma', 'Raman', _load_no_metadata,
                 os.path.join(raman_dir, 'parma.npz'))
  WebTrajDataset('Lyon (LST)', 'Raman', _load_keys_names,
                 os.path.join(raman_dir, 'LST.npz'))
  WebTrajDataset('RDRS', 'Raman', _load_keys_names,
                 os.path.join(raman_dir, 'RDRS.npz'))
  WebTrajDataset('UCL', 'Raman', _load_ucl,
                 os.path.join(raman_dir, 'UCL.npz'))
  WebTrajDataset('Dyar 96', 'Raman', _load_dyar96,
                 os.path.join(raman_dir, 'dyar96.hdf5'))
  WebTrajDataset('Mineral Mixtures', 'Raman', _load_mineral_mixes,
                 os.path.join(raman_dir, 'mineral_mixtures.hdf5'))
  WebTrajDataset('Synthetic Pyroxenes', 'Raman', _load_synth_pyrox,
                 os.path.join(raman_dir, 'synth_pyroxenes.npz'))
  WebTrajDataset('Silicate Glass', 'XAS', _load_silicate_glass,
                 os.path.join(xray_dir, 'glass_full_noSA.npz'))
  WebTrajDataset('Silicate Glass (SA-corrected)', 'XAS', _load_silicate_glass,
                 os.path.join(xray_dir, 'glass_full_withSA.npz'))
  WebVectorDataset('USDA', 'FTIR', _load_usda,
                   os.path.join(ftir_dir, 'usda_soil.npz'))
  WebVectorDataset('Amphibole', 'XAS', _load_amphibole,
                   os.path.join(xray_dir, 'amphiboleFe3.npz'))
  WebVectorDataset('Garnet', 'XAS', _load_garnet,
                   os.path.join(xray_dir, 'garnetFe3.npz'))
  WebVectorDataset('Corn', 'NIR', _load_corn,
                   os.path.join(ftir_dir, 'corn.hdf5'))

  if not public:
    mars_preds_dir = os.path.join(os.path.dirname(msl_ccs_dir),
                                  'models_predictions')
    WebLIBSDataset('Mars (big)', _load_mars_big, msl_ccs_dir,
                   os.path.join(mars_preds_dir,'msl_model_mars_preds.npz'),
                   os.path.join(mars_preds_dir,'mixed_model_mars_preds.npz'),
                   os.path.join(mars_preds_dir,'moc_model_mars_preds.npz'),
                   os.path.join(mars_preds_dir,'dust_classifier_mars_preds.npz')
                   ).is_public = False

    WebLIBSDataset('MHC multipower (no BLR)', _load_mhc_multipower,
                   os.path.join(libs_dir, 'mhc_multi_power.hdf5'), False
                   ).is_public = False
    WebLIBSDataset('MHC multipower', _load_mhc_multipower,
                   os.path.join(libs_dir, 'mhc_multi_power.hdf5'), True
                   ).is_public = False

  try:
    bands = np.loadtxt(os.path.join(libs_dir, 'prepro_wavelengths.csv'))
    raw_bands = np.loadtxt(os.path.join(libs_dir, 'raw_wavelengths.csv'))
  except IOError as e:
    logging.warning('Failed to load LIBS wavelength data!')
    logging.warning(str(e))
    return  # The rest depend on band info, so bail out now.

  WebLIBSDataset('LANL New', _load_new_lanl,
                 os.path.join(mars_dir, 'new_lanl.hdf5'), bands)
  WebLIBSDataset('LANL New (raw)', _load_new_lanl_raw,
                 os.path.join(mars_dir, 'raw_big_new_lanl.npz'), raw_bands)
  WebLIBSDataset('LANL Cleanroom', _load_cleanroom,
                 os.path.join(mars_dir, 'cleanroom.hdf5'), bands)
  WebLIBSDataset('LANL New caltargets', _load_lanl_caltargets,
                 os.path.join(mars_dir, 'new_lanl_ccct.npz'), bands)

  if not public:
    WebLIBSDataset('MHC caltargets', _load_mhc_caltargets,
                   os.path.join(mars_dir, 'mhc_caltargets.hdf5'),
                   bands).is_public = False
    WebLIBSDataset('MHC Doping', _load_doped,
                   os.path.join(libs_dir, 'doped_full.hdf5'),
                   bands).is_public = False
    WebLIBSDataset('MHC Doping (no BLR)', _load_doped,
                   os.path.join(libs_dir, 'doped_processed.hdf5'),
                   bands).is_public = False


def _load_rruff(ds, filepath):
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
    metadata['danaC'] = LookupMetadata(meta['dana/class'], 'Dana Class')
    metadata['danaT'] = LookupMetadata(meta['dana/type'], 'Dana Type')
    metadata['danaG'] = LookupMetadata(meta['dana/group'], 'Dana Group')
  ds.set_data(names, data['/spectra'], **metadata)
  return True


def _load_no_metadata(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['names'], data)
  return True


def _load_keys_names(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['keys'], data, Name=LookupMetadata(data['names']))
  return True


def _load_ucl(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['name'], data,
              Color=LookupMetadata(data['color']),
              power=LookupMetadata(data['power'], 'Laser Power'),
              wave=LookupMetadata(data['wavelength'], 'Laser Frequency'))
  return True


def _load_new_lanl(ds, filepath, bands):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  names = PrimaryKeyMetadata(data['/meta/names'])
  comp_meta = {name: NumericMetadata(arr, display_name=name) for name, arr
               in data['/composition'].items()}
  ds.set_data(bands, data['/spectra'], pkey=names,
              Composition=CompositionMetadata(comp_meta))
  return True


def _load_new_lanl_raw(ds, filepath, bands):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  names = PrimaryKeyMetadata(data['names'])
  comp_keys = [k for k in data.files if k not in ('names', 'spectra')]
  comp_meta = {k: NumericMetadata(data[k], display_name=k) for k in comp_keys}
  ds.set_data(bands, data['spectra'], pkey=names,
              Composition=CompositionMetadata(comp_meta))
  return True


def _load_cleanroom(ds, filepath, bands):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  names = PrimaryKeyMetadata(data['/meta/names'])
  comp_meta = {name: NumericMetadata(arr, display_name=name) for name, arr
               in data['/composition'].items()}
  ds.set_data(bands, data['/spectra'], pkey=names,
              Composition=CompositionMetadata(comp_meta))
  return True


def _load_lanl_caltargets(ds, filepath, bands):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False

  mineral_data = []
  mineral_names = []
  for mineral in data.files:
    d = data[mineral]
    mineral_data.append(d)
    mineral_names.extend([mineral] * d.shape[0])
  mineral_data = np.vstack(mineral_data)
  ds.set_data(bands, mineral_data,
              names=LookupMetadata(mineral_names, 'Target Names'))
  return True


def _load_mhc_caltargets(ds, filepath, bands):
  mhc_ct = _try_load(filepath, str(ds))
  if mhc_ct is None:
    return False
  ds.set_data(
      bands, mhc_ct['/spectra'],
      name=LookupMetadata(mhc_ct['/meta/names'], 'Mineral Name'),
      laser=LookupMetadata(mhc_ct['/meta/powers'], 'Laser Power'),
      time=LookupMetadata(mhc_ct['/meta/integration_times'],
                          'Integration Time'))
  return True


def _load_doped(ds, filepath, bands):
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
  if 'pkey' in meta:
    kwargs['pkey'] = PrimaryKeyMetadata(meta['pkey'])
  if 'target' in meta:
    kwargs['target'] = LookupMetadata(meta['target'], 'Target')
  if 'concentration' in meta:
    kwargs['conc'] = LookupMetadata(meta['concentration'], 'Concentration')
  if 'dopant' in meta:
    kwargs['dopant'] = LookupMetadata(meta['dopant'], 'Dopant')
  if 'waves' in meta:
    bands = meta['waves']
  ds.set_data(bands, data['/spectra'], **kwargs)
  return True


def _load_mhc_multipower(ds, filepath, with_blr=False):
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


def _load_mars_big(ds, msl_ccs_dir, pred_file, mixed_pred_file,
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


def _load_irug(ds, filepath):
  irug = _try_load(filepath, str(ds))
  if irug is None:
    return False
  ds.set_data(irug['keys'], irug,
              IDs=LookupMetadata(irug['ids']),
              Names=LookupMetadata(irug['names']),
              Materials=LookupMetadata(irug['materials']))
  return True


def _load_usgs(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['names'], data,
              ids=LookupMetadata(data['sample_ids'], 'Sample IDs'))
  return True


def _load_usda(ds, filepath):
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


def _load_dyar96(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  meta = data['/meta']
  ds.set_data(meta['pkey'], data['/spectra'],
              laser=LookupMetadata(meta['laser'], 'Laser Type'),
              ID=LookupMetadata(meta['id']),
              Mineral=LookupMetadata(meta['mineral']))
  return True


def _load_mineral_mixes(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  spectra = data['/spectra']
  meta = data['/meta']
  ds.set_data(meta['pkey'], spectra,
              sample=LookupMetadata(meta['sample'], 'Sample ID'),
              laser=LookupMetadata(meta['laser'], 'Laser Type'),
              minA=LookupMetadata(meta['mineral_1'], 'Mineral A'),
              minB=LookupMetadata(meta['mineral_2'], 'Mineral B'),
              ratio=LookupMetadata(meta['ratio'], 'Mix Ratio (A:B)'),
              grainsize=LookupMetadata(meta['grain_size'], 'Grain Size (µm)'))
  return True


def _load_synth_pyrox(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['pkey'], data,
              Mineral=LookupMetadata(data['mineral']),
              Formula=LookupMetadata(data['formula']),
              dana=LookupMetadata(data['dana'], 'Dana Number'),
              Source=LookupMetadata(data['source']))
  return True


def _load_silicate_glass(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['keys'], data,
              fe3=NumericMetadata(data['fe3'], display_name='% Fe3+'),
              Formula=LookupMetadata(data['formula']))
  return True


def _load_amphibole(ds, filepath):
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


def _load_garnet(ds, filepath):
  data = _try_load(filepath, str(ds))
  if data is None:
    return False
  ds.set_data(data['bands'], data['spectra'],
              pkey=PrimaryKeyMetadata(data['names']),
              fe3=NumericMetadata(data['fe3'], display_name='% Fe3+'))
  return True


def _load_corn(ds, filepath):
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
  load_fn = (np.load if filepath.endswith('.npz') else
             lambda f: h5py.File(f, mode='r'))
  try:
    return load_fn(filepath)
  except IOError as e:
    logging.warning('Failed to load %s data!' % data_name)
    logging.warning(str(e))
    return None
