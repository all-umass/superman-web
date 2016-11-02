from __future__ import absolute_import
import logging
import tornado.web
from matplotlib import cm, rcParams

from .base import BaseHandler
from ..web_datasets import DATASETS, CompositionMetadata, NumericMetadata

bad_cmaps = set(('gist_gray', 'gist_yarg', 'binary'))
cmaps = sorted(
    [m for m in cm.cmap_d if not m.endswith('_r') and m not in bad_cmaps],
    key=lambda x: x.lower())


class RefreshHandler(BaseHandler):
  @tornado.web.authenticated
  def get(self):
    self.write('''<form action='refresh' method='POST'>
                  <input type='submit' value='Refresh Data'>
                  </form>''')

  def post(self):
    logging.info('Refreshing datasets')
    for ds in self.all_datasets():
      ds.reload()
    self.redirect('/datasets')


class DatasetSelectorHandler(BaseHandler):
  def post(self):
    ds = self.request_one_ds('kind', 'name')
    logging.info('Generating selector for dataset: %s', ds)
    return self.render('_spectrum_selector.html', ds=ds)


class DatasetFiltererHandler(BaseHandler):
  def post(self):
    ds = self.request_one_ds('kind', 'name')
    logging.info('Generating filter HTML for dataset: %s', ds)

    if ds.pkey is None and not ds.metadata:
      return self.write('<span>No metadata to filter on</span>')

    html_parts, init_js, collect_js = ds.filter_ui()
    return self.render('_filters_table.html', ds=ds, html_parts=html_parts,
                       init_js=init_js, collect_js=collect_js)


class DatasetPlotOptionsHandler(BaseHandler):
  def post(self):
    all_ds = self.request_many_ds('kind[]', 'name[]')
    logging.info('Generating plot options HTML for: %s', map(str, all_ds))
    is_libs = any(ds.kind == 'LIBS' for ds in all_ds)
    meta_names = sorted(set.intersection(*[set(ds.metadata_names())
                                           for ds in all_ds]))
    if len(all_ds) > 1:
      # Allow coloring by dataset origin in multi-ds case
      meta_names.append(('_ds', 'Dataset'))
    return self.render('_plot_options_table.html', is_libs=is_libs,
                       metadata_names=meta_names, cmaps=cmaps,
                       default_cmap=rcParams['image.cmap'],
                       default_lw=rcParams['lines.linewidth'])


class DatasetCompositionOptionsHandler(BaseHandler):
  def post(self):
    all_ds = self.request_many_ds('kind[]', 'name[]')
    if len(all_ds) != 1:
      return self.write('This only works for 1 dataset at a time.')
    ds, = all_ds
    logging.info('Generating composition options HTML for: %s', ds)

    # Get composition and numeric metadata (key, display_name) pairs
    comp_pairs = sorted(ds.metadata_names((CompositionMetadata,)))
    num_pairs = sorted(ds.metadata_names((NumericMetadata,)))
    return self.render('_compositions.html', ds=ds,
                       comp_pairs=comp_pairs, num_pairs=num_pairs)


class DatasetPredictionOptionsHandler(BaseHandler):
  def post(self):
    all_ds = self.request_many_ds('kind[]', 'name[]')
    logging.info('Generating prediction options HTML for: %s', map(str,all_ds))

    # TODO: exclude boolean metadata from this
    pairs = [set(ds.metadata_names((NumericMetadata, CompositionMetadata)))
             for ds in all_ds]
    pairs = sorted(set.intersection(*pairs))
    return self.render('_predictions.html', meta_pairs=pairs)


class DatasetRemovalHandler(BaseHandler):
  def post(self):
    ds = self.request_one_ds('kind', 'name')
    if not ds.user_added:
      return self.write('Cannot remove this dataset.')
    logging.info('Removing user-added dataset: %s', ds)
    del DATASETS[ds.kind][ds.name]
    self.redirect('/datasets')


routes = [
    (r'/_dataset_selector', DatasetSelectorHandler),
    (r'/_dataset_filterer', DatasetFiltererHandler),
    (r'/_dataset_plot_options', DatasetPlotOptionsHandler),
    (r'/_dataset_comp_options', DatasetCompositionOptionsHandler),
    (r'/_dataset_pred_options', DatasetPredictionOptionsHandler),
    (r'/_dataset_remover', DatasetRemovalHandler),
    (r'/refresh', RefreshHandler),
]
