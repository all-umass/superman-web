# These loader functions should be moved to custom_datasets.py if/when we have
# permanent datasets that need them.


def load_mhc_hydrogen(ds, filepath):
    hdf5 = try_load(filepath, str(ds))
    if hdf5 is None:
        return False
    powers = hdf5['/meta/powers']
    names = hdf5['/meta/names']
    pkey = ['%s - %s%%' % (name, power) for name, power in zip(names, powers)]
    bands = hdf5['/meta/waves']
    comp = {
        'H2O': NumericMetadata(hdf5['/composition/H2O'], display_name='H2O')
    }
    ds.set_data(
        bands,
        hdf5['/spectra'],
        pkey=PrimaryKeyMetadata(pkey),
        Composition=CompositionMetadata(comp),
        names=LookupMetadata(names, 'Sample Name'),
        powers=LookupMetadata(powers, 'Laser Power'))
    return True


def load_mhc_multipower(ds, filepath, with_blr=False):
    data = try_load(filepath, str(ds))
    if data is None:
        return False
    kind = 'preprocessed' if with_blr else 'formatted'
    s = data['/spectra/' + kind]
    spectra = np.vstack((s['low_power'], s['med_power'], s['high_power']))
    comps = data['/composition']
    meta = data['/meta']
    names = np.asarray(meta['names'])
    idx = np.argsort(names)
    powers = ('3.2', '5', '7')
    pkey = ['%s @ %s%%' % (n, p) for p in powers for n in names]
    comp_meta = {
        k: NumericMetadata(comps[k], display_name=k, repeats=3)
        for k in comps.keys()
    }
    ds.set_data(
        meta['waves'],
        spectra,
        pkey=PrimaryKeyMetadata(pkey),
        in_lanl=BooleanMetadata(meta['in_lanl'], 'In LANL', repeats=3),
        name=LookupMetadata(names[idx], 'Name', labels=np.tile(idx, 3)),
        power=LookupMetadata(
            powers, 'Laser Power', labels=np.repeat([0, 1, 2], len(names))),
        comp=CompositionMetadata(comp_meta, 'Composition'))
    return True


def load_mhc_xrf(ds, data_file, meta_file):
    log_prefix = 'MHC XRF loader for {0}:'.format(ds.name)
    logging.info('%s Starting...', log_prefix)
    try:
        hdf5 = h5py.File(data_file, mode='r')
        meta = np.load(meta_file, allow_pickle=True)
    except IOError as e:
        logging.error('%s Failed to load data in %s: %s', log_prefix, data_dir,
                      str(e))
        return None

    logging.info('%s Collecting metadata...', log_prefix)
    pkey = np.array(meta['spectrum_number'], dtype=bytes)
    compositions = import_comps(meta, log_prefix)

    logging.info('%s Loading data...', log_prefix)
    try:
        ds.set_data(
            pkey,
            hdf5['/spectra'],
            Composition=CompositionMetadata(compositions),
            Instrument=TagMetadata([tags.split() for tags in meta['Project']]),
            Pellet=LookupMetadata(meta['pellet_name']),
            Filter=LookupMetadata(meta['Filter']),
            Dopant=LookupMetadata(meta['instrument']),
            Matrix=LookupMetadata(meta['Vacuum']),
            Duration=NumericMetadata(
                meta['Duration Time'], display_name='Duration Time'),
        )
    except Exception as e:
        e_type = type(e).__name__
        logging.error('%s Exception: %s "%s"', log_prefix, e_type, str(e))
        raise
    logging.info('%s Finished setup.', log_prefix)
    return True
