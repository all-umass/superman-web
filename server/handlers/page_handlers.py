from __future__ import absolute_import
import datetime
import matplotlib
import numpy as np
import os
import tornado.web
from collections import defaultdict
from superman.baseline import BL_CLASSES
from matplotlib import cm, rcParams

from .base import BaseHandler
from ..web_datasets import (
    CompositionMetadata, NumericMetadata, WebVectorDataset)

MPL_JS = sorted(os.listdir(os.path.join(matplotlib.__path__[0],
                                        'backends/web_backend/jquery/js')))
bad_cmaps = set(('gist_gray', 'gist_yarg', 'binary'))
cmaps = sorted(
    [m for m in cm.cmap_d if not m.endswith('_r') and m not in bad_cmaps],
    key=lambda x: x.lower())


def compute_step(lb, ub, kind):
  if kind == 'integer':
    return 1
  if kind == 'log':
    lb, ub = np.log10((lb, ub))
  return (ub - lb) / 100.

blr_kwargs = dict(
    bl_classes=sorted((key, bl()) for key, bl in BL_CLASSES.items()),
    compute_step=compute_step, log10=np.log10)


class MainPage(BaseHandler):
  def get(self):
    logged_in = self.current_user is not None
    # tuples of (title, relative_url, description) for subpages
    subpage_info = [(p.title, link[1:], p.description) for link, p in routes
                    if issubclass(p, Subpage) and (logged_in or p.public)]
    self.render('index.html', page_title='Project Superman: Web Interface',
                subpage_info=subpage_info, mpl_js=MPL_JS, logged_in=logged_in)


class LoginPage(BaseHandler):
  def get(self):
    if self.get_argument('logout', False):
      self.clear_cookie('user')
      self.redirect('/')
    else:
      self.render('login.html',
                  message=self.get_argument('msg', ''),
                  next=self.get_argument('next', '/'))

  def post(self):
    password = self.get_argument('pw')
    if password == 'superman':  # Elite security
      # Doesn't matter what value it has, just needs to be there.
      self.set_secure_cookie('user', 'authenticated')
      self.redirect(self.get_argument('next', '/'))
    else:
      self.clear_cookie('user')
      self.redirect('/login?msg=Login%20failed')


class Subpage(BaseHandler):
  '''Base class which renders the common template arguments for subpages.
  Requires that subclasses define the following properties:
   * template    - name of template html file
   * title       - title of the page
   * description - sentence describing the page
   * figsize     - figure size in inches, as a tuple of (width, height)
  '''
  public = True

  def render(self, **kwargs):
    if 'fig_id' not in kwargs:
      kwargs['fig_id'] = self.application.register_new_figure(self.figsize)
    kwargs['ws_uri'] = "ws://{req.host}/".format(req=self.request)
    kwargs['page_title'] = self.title
    kwargs['mpl_js'] = MPL_JS
    return BaseHandler.render(self, self.template, **kwargs)


class DatasetsPage(Subpage):
  template = 'datasets.html'
  title = 'Datasets'
  description = 'Browse all spectroscopy datasets.'

  def get(self):
    self.render(fig_id=0,  # No figure on this page
                dt=datetime.datetime.fromtimestamp,
                datasets=self.all_datasets())


class DataExplorerPage(Subpage):
  template = 'explorer.html'
  title = 'Dataset Explorer'
  description = 'Filter and plot datasets.'
  figsize = (14, 6)

  def get(self):
    ds_tree = defaultdict(list)
    for ds in self.all_datasets():
      ds_tree[ds.kind].append(ds.name)
    self.render(datasets=ds_tree, ds_kind=self.get_argument('ds_kind', ''),
                ds_name=self.get_argument('ds_name', ''),
                cmaps=cmaps, default_cmap=rcParams['image.cmap'],
                default_lw=rcParams['lines.linewidth'], **blr_kwargs)


class BaselinePage(Subpage):
  template = 'baseline.html'
  title = 'Baseline Correction'
  description = 'Explore baseline correction algorithms.'
  figsize = (8, 8)

  def get(self):
    # Do the figure creation manually, due to subplot shenanigans
    fignum = self.application.register_new_figure(self.figsize)
    fig_data = self.application.figure_data[fignum]
    fig = fig_data.figure
    ax1 = fig.add_subplot(211)
    fig.add_subplot(212, sharex=ax1)
    self.render(datasets=self.all_datasets(), fig_id=fignum, **blr_kwargs)


class SearcherPage(Subpage):
  template = 'search.html'
  title = 'Spectrum Matching'
  description = 'Match a spectrum against a target dataset.'
  figsize = (8, 4)

  def get(self):
    pkey_ds = [d for d in self.all_datasets() if d.pkey is not None]
    self.render(datasets=pkey_ds, **blr_kwargs)


class CompositionsPage(Subpage):
  template = 'composition.html'
  title = 'Mars Compositions'
  description = 'Plot predicted compositions of Mars LIBS shots.'
  figsize = (10, 6)

  def get(self):
    ds = self.get_dataset('LIBS', 'MSL ChemCam')
    if ds is None:
      page = '''
      <body style="text-align:center;padding-top:25vh;font-size:larger">
      Mars data is not available.<br /><a href="/">Go back</a></body>'''
      return self.write(page)

    # Get composition and numeric metadata (key, display_name) pairs
    comp_pairs = sorted(ds.metadata_names((CompositionMetadata,)))
    num_pairs = sorted(ds.metadata_names((NumericMetadata,)))

    html_parts, init_js, collect_js = ds.filter_ui(num_cols=2)
    self.render(ds=ds, comp_pairs=comp_pairs, num_pairs=num_pairs,
                html_parts=html_parts, init_js=init_js,
                collect_js=collect_js, num_cols=2)


class PeakFitPage(Subpage):
  template = 'peakfit.html'
  title = 'Peak Fitting'
  description = 'Explore peak fitting algorithms.'
  figsize = (8, 4)

  def get(self):
    self.render(datasets=self.all_datasets(), **blr_kwargs)


class PredictionPage(Subpage):
  template = 'predict.html'
  title = 'PLS Prediction'
  description = 'Run PLS regression to build predictive models.'
  figsize = (8, 6)
  public = False

  @tornado.web.authenticated
  def get(self):
    vector_ds = [d for d in self.all_datasets()
                 if isinstance(d, WebVectorDataset)]
    self.render(datasets=vector_ds, **blr_kwargs)


class DebugPage(BaseHandler):
  @tornado.web.authenticated
  def get(self):
    self.render('debug.html', page_title='Debug View', mpl_js=[],
                figure_data=self.application.figure_data)


# Define the routes for each page.
routes = [
    (r'/', MainPage),
    (r'/datasets', DatasetsPage),
    (r'/explorer', DataExplorerPage),
    (r'/baseline', BaselinePage),
    (r'/search', SearcherPage),
    (r'/compositions', CompositionsPage),
    (r'/peakfit', PeakFitPage),
    (r'/login', LoginPage),
    (r'/predict', PredictionPage),
    (r'/debug', DebugPage),
]
