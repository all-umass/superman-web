from __future__ import absolute_import, print_function, division
import ast
import logging
import numpy as np

from .common import BaseHandler, BLR_KWARGS
from ..web_datasets import (
    DATASETS, CompositionMetadata, NumericMetadata, LookupMetadata,
    BooleanMetadata
)
from six.moves import map


class FilterBoxHandler(BaseHandler):
    def post(self):
        ds = self.request_one_ds('kind', 'name')
        if ds is None:
            return self.visible_error(404, 'Dataset not found.')
        logging.info('Generating filter HTML for dataset: %s', ds)

        if ds.pkey is None and not ds.metadata:
            return self.write('<span>No metadata to filter on</span>')

        html_parts, init_js, collect_js = ds.filter_ui()
        return self.render(
            '_dataset_filters.html',
            ds=ds,
            html_parts=html_parts,
            init_js=init_js,
            collect_js=collect_js)


class FilterHandler(BaseHandler):
    def post(self):
        fig_data = self.get_fig_data()
        if fig_data is None:
            return self.visible_error(403, 'Broken connection to server.')

        ds_name = self.get_argument('ds_name')
        ds_kind = self.get_argument('ds_kind')
        ds = self.get_dataset(ds_kind, ds_name)
        if ds is None:
            msg = "Can't find dataset: %s [%s]" % (ds_name, ds_kind)
            return self.visible_error(404, msg)

        params = {k: ast.literal_eval(self.get_argument(k))
                  for k in ds.metadata}
        if ds.pkey is not None:
            pkey_arg = self.get_argument('pkey', None)
            if pkey_arg:
                params['pkey'] = ast.literal_eval(pkey_arg)
        logging.info('Filtering %s with args: %s', ds, params)

        mask = ds.filter_metadata(params)
        fig_data.filter_mask[ds] = mask
        num_spectra = np.count_nonzero(mask)
        logging.info('Filtered to %d spectra', num_spectra)

        # blow away any cached explorer data
        fig_data.clear_explorer_cache()

        return self.write(str(num_spectra))


class PlotOptionsHandler(BaseHandler):
    def post(self):
        all_ds = self.request_many_ds('kind[]', 'name[]')
        logging.info('Generating plot options HTML for: %s',
                     list(map(str, all_ds)))
        is_libs = any(ds.kind == 'LIBS' for ds in all_ds)
        meta_names = sorted(set.intersection(*[set(ds.metadata_names())
                                               for ds in all_ds]))
        if len(all_ds) > 1:
            # Allow coloring by dataset origin in multi-ds case
            meta_names.append(('_ds', 'Dataset'))
        return self.render('_filterplots.html', is_libs=is_libs,
                           metadata_names=meta_names)


class CompositionOptionsHandler(BaseHandler):
    def post(self):
        all_ds = self.request_many_ds('kind[]', 'name[]')
        if len(all_ds) != 1:
            return self.visible_error(
                403, 'This only works for 1 dataset at a time.')
        ds, = all_ds
        logging.info('Generating composition options HTML for: %s', ds)

        # Get composition and numeric metadata (key, display_name) pairs
        comp_pairs = sorted(ds.metadata_names((CompositionMetadata,)))
        if comp_pairs:
            num_pairs = sorted(ds.metadata_names((NumericMetadata,)))
            self.render('_compositions.html', ds=ds,
                        comp_pairs=comp_pairs, num_pairs=num_pairs)


class PredictionOptionsHandler(BaseHandler):
    def post(self):
        all_ds = self.request_many_ds('kind[]', 'name[]')
        logging.info('Generating prediction options HTML for: %s',
                     list(map(str, all_ds)))

        pred_pairs, strat_pairs = [], []
        for ds in all_ds:
            pred_pairs.append(
                set(ds.metadata_names((NumericMetadata, CompositionMetadata))))
            strat_pairs.append(
                set(ds.metadata_names((LookupMetadata, BooleanMetadata))))

        # only use common metadata across all datasets
        pred_pairs = sorted(set.intersection(*pred_pairs))
        if pred_pairs:
            strat_pairs = sorted(set.intersection(*strat_pairs))
            self.render('_predictions.html', pred_pairs=pred_pairs,
                        strat_pairs=strat_pairs)


class ClassificationOptionsHandler(BaseHandler):
    def post(self):
        all_ds = self.request_many_ds('kind[]', 'name[]')
        logging.info('Generating prediction options HTML for: %s',
                     list(map(str, all_ds)))

        pairs = [set(ds.metadata_names((LookupMetadata, BooleanMetadata)))
                 for ds in all_ds]
        pairs = sorted(set.intersection(*pairs))
        if pairs:
            return self.render('_classifications.html', meta_pairs=pairs)


class MatchOptionsHandler(BaseHandler):
    def post(self):
        ds_kinds = set(self.get_arguments('kind[]'))
        if len(ds_kinds) != 1:
            return self.visible_error(403, 'Only one dataset kind is allowed.')

        ds_kind, = ds_kinds
        logging.info('Generating matching options HTML for %s', ds_kind)
        ds_names = sorted(DATASETS[ds_kind])
        return self.render(
            '_matching.html',
            ds_kind=ds_kind,
            ds_names=ds_names,
            **BLR_KWARGS)


routes = [
    (r'/_filter', FilterHandler),
    (r'/_dataset_filterer', FilterBoxHandler),
    (r'/_plot_options', PlotOptionsHandler),
    (r'/_comp_options', CompositionOptionsHandler),
    (r'/_pred_options', PredictionOptionsHandler),
    (r'/_classify_options', ClassificationOptionsHandler),
    (r'/_matching_options', MatchOptionsHandler),
]
