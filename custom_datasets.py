# -*- coding: utf-8 -*-
import numpy as np
from superman.dataset import (
    NumericMetadata, BooleanMetadata, PrimaryKeyMetadata, LookupMetadata,
    CompositionMetadata, TagMetadata, DateMetadata)

# Helper function for loading HDF5 or NPZ files.
from backend.dataset_loaders import try_load


def load_corn(ds, filepath):
  data = try_load(filepath, str(ds))
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

