from __future__ import absolute_import
import logging
import tornado.web

from .base import BaseHandler


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


class DatasetHandler(BaseHandler):
  def get_ds(self):
    return self.get_dataset(self.get_argument('kind'),
                            self.get_argument('name'))


class DatasetSelectorHandler(DatasetHandler):
  def post(self):
    ds = self.get_ds()
    logging.info('Generating selector for dataset: %s', ds)
    return self.render('_spectrum_selector.html', ds=ds)


class DatasetFiltererHandler(DatasetHandler):
  def post(self):
    ds = self.get_ds()
    logging.info('Generating filter HTML for dataset: %s', ds)

    if ds.pkey is None and not ds.metadata:
      return self.write('<span>No metadata to filter on</span>')

    html_parts, init_js, collect_js = ds.filter_ui(num_cols=2)
    return self.render('_filters_table.html', ds=ds, html_parts=html_parts,
                       init_js=init_js, collect_js=collect_js, num_cols=2)


class DatasetPlotOptionsHandler(DatasetHandler):
  def post(self):
    ds = self.get_ds()
    logging.info('Generating plot options HTML for dataset: %s', ds)
    return self.render('_plot_options_table.html', ds_kind=ds.kind,
                       metadata_names=sorted(ds.metadata_names()))


routes = [
    (r'/_dataset_selector', DatasetSelectorHandler),
    (r'/_dataset_filterer', DatasetFiltererHandler),
    (r'/_dataset_plot_options', DatasetPlotOptionsHandler),
    (r'/refresh', RefreshHandler),
]
