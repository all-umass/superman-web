from __future__ import absolute_import
import datetime
import matplotlib
import os
import tornado.web
from collections import defaultdict
from matplotlib import cm, rcParams

from .common import BaseHandler, BLR_KWARGS

MPL_JS = sorted(os.listdir(os.path.join(matplotlib.__path__[0],
                                        'backends/web_backend/jquery/js')))


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
    if password == self.application.login_password:
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
  figsize = None
  public = True

  def render(self, **kwargs):
    kwargs['page_title'] = self.title
    kwargs['mpl_js'] = MPL_JS
    if self.figsize is not None:
      kwargs['ws_uri'] = "ws://{req.host}/".format(req=self.request)
      if 'fig_id' not in kwargs:
        kwargs['fig_id'] = self.application.register_new_figure(self.figsize)
    return BaseHandler.render(self, self.template, **kwargs)


class DatasetsPage(Subpage):
  template = 'datasets.html'
  title = 'Datasets'
  description = 'Browse all spectroscopy datasets.'

  def get(self):
    self.render(dt=datetime.datetime.fromtimestamp,
                datasets=self.all_datasets())


class DataExplorerPage(Subpage):
  template = 'explorer.html'
  title = 'Dataset Explorer'
  description = 'Analyze, process, and plot spectra.'
  figsize = (8, 4)

  def initialize(self):
    bad_cmaps = set(('gist_gray', 'gist_yarg', 'binary'))
    self.cmaps = sorted(
        [m for m in cm.cmap_d if not m.endswith('_r') and m not in bad_cmaps],
        key=lambda x: x.lower())

  def get(self):
    ds_tree = defaultdict(dict)
    for ds in self.all_datasets():
      is_traj = ds.num_dimensions() is None
      ds_tree[ds.kind][ds.name] = (hash(ds), is_traj)
    self.render(ds_tree=ds_tree, logged_in=(self.current_user is not None),
                ds_kind=self.get_argument('ds_kind', ''),
                ds_name=self.get_argument('ds_name', ''),
                cmaps=self.cmaps, default_cmap=rcParams['image.cmap'],
                default_lw=rcParams['lines.linewidth'], **BLR_KWARGS)


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
    self.render(datasets=self.all_datasets(), fig_id=fignum, **BLR_KWARGS)


class PeakFitPage(Subpage):
  template = 'peakfit.html'
  title = 'Peak Fitting'
  description = 'Explore peak fitting algorithms.'
  figsize = (8, 4)

  def get(self):
    self.render(datasets=self.all_datasets(), **BLR_KWARGS)


class DatasetImportPage(Subpage):
  template = 'import.html'
  title = 'Dataset Import'
  description = 'Upload new datasets to Superman.'
  public = False

  @tornado.web.authenticated
  def get(self):
    self.render(ds_kinds=self.dataset_kinds())


class SearchPage(Subpage):
  template = 'search.html'
  title = 'Spectrum Search'
  description = 'Find data by searching across datasets.'

  def get(self):
    self.render(ds_kinds=self.dataset_kinds())


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
    (r'/peakfit', PeakFitPage),
    (r'/login', LoginPage),
    (r'/import', DatasetImportPage),
    (r'/search', SearchPage),
    (r'/debug', DebugPage),
]
