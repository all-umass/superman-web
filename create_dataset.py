#!/usr/bin/env python3
from argparse import ArgumentParser
import logging
import sys
from zipfile import is_zipfile

import numpy as np
import pandas as pd
from superman.dataset.metadata import (
    BooleanMetadata, NumericMetadata, LookupMetadata)

from backend.handlers.upload import _save_ds, _traj_ds, _vector_ds

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')


# TODO: Deduplicate with the version in backend.handlers.upload
def _load_metadata_csv(fh):
  meta = pd.read_csv(fh)

  if meta.columns[0] != 'pkey':
    raise ValueError('Metadata CSV must start with "pkey" column.')

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
  return meta_kwargs, meta_pkeys


# TODO: Deduplicate with the version in backend.handlers.upload
def ds_upload(meta_file, fh, ds_name, ds_kind, description):
  if not meta_file:
    meta_kwargs = {}
    meta_pkeys = []
  else:
    meta_kwargs, meta_pkeys = _load_metadata_csv(meta_file)
  print('Metadata loaded:', len(meta_pkeys), 'primary keys.')

  if is_zipfile(fh):
    print('Reading ZIP spectra file...')
    # interpret this as a ZIP of csv files
    fh.seek(0)
    err = _traj_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys,
                   None, description)
  else:
    print('Reading CSV spectra file...')
    # this is one single csv file with all spectra in it
    fh.seek(0)
    err = _vector_ds(fh, ds_name, ds_kind, meta_kwargs, meta_pkeys,
                     None, description)

  if err is not None:
    raise ValueError(err[1])
  print('Finished loading spectra.')


def binary_file(f):
  return open(f, 'rb')


def main():
  ap = ArgumentParser()
  ap.add_argument('spectra', type=binary_file, help='Path to a spectra file.')
  ap.add_argument('--meta', type=binary_file,
                  help='Optional path to a metadata CSV.')
  ap.add_argument('--ds_kind',
                  help='Superman dataset kind (LIBS, FTIR, Raman, etc.)')
  ap.add_argument('--ds_name', help='Dataset name.')
  ap.add_argument('--description', help='Optional description of the dataset.')
  args = ap.parse_args()
  if not args.ds_kind or not args.ds_name:
    ap.error('--ds_kind and --ds_name must both be provided')

  # Updates the global DATASETS dict.
  ds_upload(args.meta, args.spectra, args.ds_name, args.ds_kind,
            args.description)

  # TODO: use a more standalone version of this function
  _save_ds(args.ds_kind, args.ds_name)
  print('Updated uploads/user_data.yml')



if __name__ == '__main__':
  main()
