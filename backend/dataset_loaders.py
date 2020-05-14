from __future__ import absolute_import, print_function, division
import h5py
import logging
import numpy as np
import os.path
import pandas as pd
import re
import yaml

from superman.dataset import (
    NumericMetadata, PrimaryKeyMetadata, CompositionMetadata, DateMetadata)

from . import web_datasets
from .web_datasets import WebLIBSDataset, WebVectorDataset, WebTrajDataset


def load_datasets(config_fh, custom_loaders,
                  public_only=False, user_added=False):
    config = yaml.safe_load(config_fh)

    for kind, entries in config.items():
        for name, info in entries.items():
            # skip this entry if it shouldn't be included
            is_public = info.get('public', True)
            if public_only and not is_public:
                continue

            if 'files' in info:
                files = info['files']
            else:
                files = [info['file']]

            if 'loader' in info:
                # look up the loader function from the module namespace
                loader_fn = getattr(custom_loaders, info['loader'])
            else:
                # construct a loader from the meta_mapping and the default
                # template
                meta_mapping = [(k, getattr(web_datasets, cls), mname)
                                for k, cls, mname in info.get('metadata', [])]
                if info.get('vector', False):
                    loader_fn = _generic_vector_loader(meta_mapping)
                else:
                    loader_fn = _generic_traj_loader(meta_mapping)

            # MHC SuperLIBS 5120 is trajectory, not vector.
            if kind == 'LIBS' and info.get('vector', True):
                ds = WebLIBSDataset(name, loader_fn, *files)
            elif info.get('vector', False):
                ds = WebVectorDataset(name, kind, loader_fn, *files)
            else:
                ds = WebTrajDataset(name, kind, loader_fn, *files)

            if 'description' in info:
                ds.description = info['description']
            if 'urls' in info:
                ds.urls = info['urls']
            ds.is_public = is_public
            ds.user_added = user_added


def try_load(filepath, data_name):
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


def _generic_traj_loader(meta_mapping):
    """Creates a loader function for a standard HDF5 file representing a
    trajectory dataset. The HDF5 structure is expected to be:
     - /meta/pkey : an array of keys, used to address individual spectra
     - /spectra/[pkey] : a (n,2) trajectory spectrum
     - /meta/foobar : (optional) metadata, specified by the meta_mapping
    """
    def _load(ds, filepath):
        data = try_load(filepath, str(ds))
        if data is None:
            return False
        meta = data['/meta']
        kwargs = {}
        for key, cls, display_name in meta_mapping:
            if key not in meta:
                continue
            m = meta[key]
            if cls is DateMetadata:
                m = pd.to_datetime(np.array(m).astype(str, copy=False))
            elif cls is PrimaryKeyMetadata:
                assert key == 'pkey'
                continue
            safe_key = re.sub(r'[^a-z0-9_-]', '', key, flags=re.I)
            kwargs[safe_key] = cls(m, display_name=display_name)
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
        data = try_load(filepath, str(ds))
        if data is None:
            return False
        meta = data['/meta']
        kwargs = {}
        for key, cls, display_name in meta_mapping:
            if key not in meta:
                continue
            if cls is PrimaryKeyMetadata:
                kwargs['pkey'] = cls(meta[key])
            elif cls is DateMetadata:
                kwargs[key] = cls(pd.to_datetime(np.array(meta[key])),
                                  display_name=display_name)
            else:
                kwargs[key] = cls(meta[key], display_name=display_name)
        if '/composition' in data:
            comp_meta = {
                name: NumericMetadata(arr, display_name=name)
                for name, arr in data['/composition'].items()
            }
            kwargs['Composition'] = CompositionMetadata(comp_meta)
        ds.set_data(meta['waves'], data['/spectra'], **kwargs)
        return True
    return _load
