import logging
import numpy as np
import os
import time
from base64 import b64encode
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
from matplotlib.backends.backend_agg import FigureCanvasAgg
from threading import Thread
from tornado.escape import xhtml_escape
from six import BytesIO, string_types

from superman.dataset import (
    VectorDataset, TrajDataset, NumericMetadata, BooleanMetadata,
    PrimaryKeyMetadata, LookupMetadata, CompositionMetadata)

__all__ = [
    'WebTrajDataset', 'WebVectorDataset', 'WebLIBSDataset',
    'NumericMetadata', 'BooleanMetadata', 'PrimaryKeyMetadata',
    'LookupMetadata', 'CompositionMetadata', 'UploadedDataset'
]

# Global structure for all loaded datasets.
DATASETS = dict(Raman={}, LIBS={}, FTIR={}, NIR={}, XAS={}, XRD={})


class _ReloadableMixin(object):
  def init_load(self, loader_fn, loader_args):
    # Machinery for on-demand data refresh
    self.load_time = -1
    self.loader_fn = loader_fn
    self.loader_args = loader_args
    Thread(target=self.reload, name='%s loader thread' % self).start()

  def reload(self):
    if self.loader_args:
      mtime = max(map(_try_get_mtime, self.loader_args))
    else:
      mtime = 0
    if mtime < self.load_time:
      return
    if not self.loader_fn(self, *self.loader_args):
      return
    self.load_time = mtime if mtime > 0 else time.time()
    # register with the global dataset manager
    DATASETS[self.kind][self.name] = self
    logging.info('Successfully registered %s', self)

  def x_axis_units(self):
    if self.kind in ('LIBS', 'NIR'):
      return 'Wavelength (nm)'
    if self.kind in ('Raman', 'FTIR'):
      return 'Wavenumber (1/cm)'
    if self.kind == 'XAS':
      return 'Energy (eV)'
    if self.kind == 'XRD':
      return '2 Theta'
    return 'Unknown units'

  def filter_ui(self, num_cols=2):
    # get a unique string for this dataset
    ds_key = 'ds%d' % hash(str(self))
    # Get HTML+JS for filters
    metas = sorted(self.metadata.items(), key=lambda t: (str(type(t[1])), t[0]))
    if self.pkey is not None:
      metas.append(('pkey', self.pkey))
    # Collect all the fragments
    init_js, collect_js, tds, comp_tds = [], [], [], []
    for key, m in metas:
      full_key = ds_key + '_' + key
      ijs, cjs = _get_filter_js(m, full_key)
      init_js.append(ijs)
      collect_js.append((key, cjs))
      if isinstance(m, CompositionMetadata):
        comp_tds.extend(_get_composition_filter_html(m, key, full_key))
      else:
        tds.append(('', _get_filter_html(m, key, full_key)))
    tds += comp_tds
    # reshape tds
    html_parts = [[]]
    for td in tds:
      if len(html_parts[-1]) == num_cols:
        html_parts.append([td])
      else:
         html_parts[-1].append(td)
    last_row_pad = num_cols - len(html_parts[-1])
    html_parts[-1].extend([('','')]*last_row_pad)
    return html_parts, init_js, collect_js

  def metadata_names(self, allowed_baseclasses=(object,)):
    for key, m in self.metadata.items():
      if not isinstance(m, allowed_baseclasses):
        continue
      if isinstance(m, CompositionMetadata):
        ck = key + '$'
        cn = m.display_name(key) + ': '
        for k, mm in m.comps.items():
          yield ck+k, cn+mm.display_name(k)
      else:
        yield key, m.display_name(key)


class WebTrajDataset(TrajDataset, _ReloadableMixin):
  def __init__(self, name, spec_kind, loader_fn, *loader_args):
    TrajDataset.__init__(self, name, spec_kind)
    self.description = 'No description provided.'
    self.urls = []
    self.is_public = True
    self.user_added = False
    self.init_load(loader_fn, loader_args)


class WebVectorDataset(VectorDataset, _ReloadableMixin):
  def __init__(self, name, spec_kind, loader_fn, *loader_args):
    VectorDataset.__init__(self, name, spec_kind)
    self.description = 'No description provided.'
    self.urls = []
    self.is_public = True
    self.user_added = False
    self.init_load(loader_fn, loader_args)


class WebLIBSDataset(WebVectorDataset):
  def __init__(self, name, *args, **kwargs):
    WebVectorDataset.__init__(self, name, 'LIBS', *args, **kwargs)

  def set_data(self, bands, spectra, pkey=None, **metadata):
    if 'si' not in metadata:
      # Compute the Si ratio as a proxy for temperature
      chan_ranges = (288., 288.5, 633., 635.5)
      den_lo, den_hi, num_lo, num_hi = np.searchsorted(bands, chan_ranges)
      si_ratio = np.asarray(spectra[:,num_lo:num_hi].max(axis=1) /
                            spectra[:,den_lo:den_hi].max(axis=1))
      np.maximum(si_ratio, 0, out=si_ratio)
      metadata['si'] = NumericMetadata(si_ratio, display_name='Si Ratio')

    # Set data as usual, with the Si ratio added
    VectorDataset.set_data(self, bands, spectra, pkey=pkey, **metadata)

  def view(self, **kwargs):
    if 'nan_gap' not in kwargs:
      # default to inserting NaNs for LIBS data
      kwargs['nan_gap'] = 1
    return VectorDataset.view(self, **kwargs)


class UploadedDataset(TrajDataset):
  def __init__(self, name, traj):
    TrajDataset.__init__(self, name, '<unknown>')
    # do some cleanup on the spectrum
    if traj[0,0] > traj[1,0]:
      traj = traj[::-1]
    data = {name: traj.astype(np.float32, order='C')}
    self.set_data([name], data)


def _try_get_mtime(filepath):
  if not isinstance(filepath, string_types):
    return -1
  try:
    return os.path.getmtime(filepath)
  except OSError:
    return -1


def _generate_histogram(m):
  # Make a 350px by 32px image for a slider background
  fig = Figure(figsize=(3.5,0.32), dpi=100, tight_layout=False)
  # This is required, even though we don't explicitly use the canvas.
  canvas = FigureCanvasAgg(fig)
  ax = fig.add_subplot(1,1,1)
  vrange = float(m.bounds[1] - m.bounds[0])
  num_bins = np.ceil(vrange / m.step) + 1
  if np.isnan(num_bins):
    ax.hist(m.arr)
  elif num_bins > 300:
    ax.hist(m.arr, 300, range=m.bounds)
  else:
    ax.hist(m.arr, int(num_bins), range=m.bounds)
  ax.axis('off')
  fig.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0,wspace=0)
  ax.margins(0, 0)
  ax.xaxis.set_major_locator(NullLocator())
  ax.yaxis.set_major_locator(NullLocator())
  # save it as a base64 encoded string
  img_data = BytesIO()
  fig.savefig(img_data, format='png', bbox_inches='tight', pad_inches=0)
  return b64encode(img_data.getvalue())


def _get_filter_js(m, full_key):
  if isinstance(m, NumericMetadata):
    elt = '$("#%s")' % full_key
    if isinstance(m, BooleanMetadata):
      return '', elt + '.val()'
    lb, ub = m.bounds
    init_js = ('%s.slider({min: %.17g, max: %.17g, step: %.17g, range: true, '
               'values: [%.17g,%.17g], slide: function(e,ui){'
               '$("#%s_label").text(ui.values[0]+" to "+ui.values[1]);}});') % (
                   elt, lb, ub, m.step, lb, ub, full_key)
    collect_js = elt + '.slider("values")'
    return init_js, collect_js
  if isinstance(m, CompositionMetadata):
    init_parts, collect_parts = [], []
    for k, mm in m.comps.items():
      ijs, cjs = _get_filter_js(mm, full_key + '-' + k)
      init_parts.append(ijs)
      collect_parts.append('%s: %s' % (k, cjs))
    collect_js = '{' + ','.join(collect_parts) + '}'
    init_js = '$("#%s_toggle").click();\n%s' % (full_key, '\n'.join(init_parts))
    return init_js, collect_js
  # only LookupMetadata and PrimaryKeyMetadata remain
  jq = '$("#%s_chooser")' % full_key
  # initialize the chosen dropdown, adding some width for the scrollbar
  init_js = jq + ".css('width', '+=20').chosen({search_contains: true});"
  prefix = ''
  dtype = (m.uniques if isinstance(m,LookupMetadata) else m.keys).dtype
  if np.issubdtype(dtype, np.number):
    prefix = '+'  # convert JS string to number
  collect_js = jq + ('.next().find(".search-choice").map(function(){'
                     'return %s($(this).text())}).toArray()') % prefix
  return init_js, collect_js


def _get_filter_html(m, key, full_key):
  disp = m.display_name(key)
  if isinstance(m, BooleanMetadata):
    return ('%s: <select id="%s"><option value=both>Both</option>'
            '<option value=yes>Yes</option><option value=no>No</option>'
            '</select>') % (disp, full_key)
  if isinstance(m, NumericMetadata):
    lb, ub = m.bounds
    # lazy load histogram
    if not hasattr(m, 'hist_image'):
      m.hist_image = _generate_histogram(m)
    return ('%s: <span id="%s_label">%s to %s</span><br />'
            '<div class="slider" id="%s" style="background-image: '
            'url(data:img/png;base64,%s);"></div>') % (
                disp, full_key, lb, ub, full_key, m.hist_image)
  # only LookupMetadata and PrimaryKeyMetadata remain
  uniques = sorted(m.keys) if isinstance(m, PrimaryKeyMetadata) else m.uniques
  html = ('%s:<br /><select id="%s_chooser" data-placeholder="All" '
          'class="chosen-select" multiple>\n') % (disp, full_key)
  html += '\n'.join(
      '<option value="%s">%s</option>' % (x, xhtml_escape(str(x)))
      for x in uniques)
  html += '</select>'
  return html


def _get_composition_filter_html(m, key, full_key):
  disp = m.display_name(key)
  # CSS class for our filters, and CSS ID for our button
  css = full_key + '_toggle'
  td = ('%s: <button onclick="$(\'.%s\').toggle()" '
        'id="%s">Show/Hide</button>') % (disp, css, css)
  html = [('', td)]
  for k, m in m.comps.items():
    html.append((css, _get_filter_html(m, key + '-' + k, full_key + '-' + k)))
  return html
