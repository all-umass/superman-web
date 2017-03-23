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
    VectorDataset, TrajDataset, NumericMetadata, BooleanMetadata, DateMetadata,
    PrimaryKeyMetadata, LookupMetadata, CompositionMetadata, TagMetadata)

__all__ = [
    'WebTrajDataset', 'WebVectorDataset', 'WebLIBSDataset',
    'UploadedSpectrumDataset'
]

# Global structure for all loaded datasets.
DATASETS = dict(
    Raman={}, LIBS={}, FTIR={}, NIR={}, XAS={}, XRD={}, Mossbauer={}
)

# Ordering for filters of various metadata types.
FILTER_ORDER = {
    PrimaryKeyMetadata: 0,
    LookupMetadata: 1,
    BooleanMetadata: 2,
    TagMetadata: 3,
    DateMetadata: 4,
    NumericMetadata: 5,
    CompositionMetadata: 999  # should always come last
}


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
    # release any existing data first
    self.clear_data()
    if not self.loader_fn(self, *self.loader_args):
      # loader failed, remove ourselves from the registry
      DATASETS[self.kind].pop(self.name, None)
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
    if self.kind == 'Mossbauer':
      return 'Velocity (mm/s)'
    return 'Unknown units'

  def filter_ui(self):
    # get a unique string for this dataset
    ds_key = 'ds%d' % hash(str(self))
    # Get HTML+JS for filters
    metas = sorted(self.metadata.items(),
                   key=lambda t: (FILTER_ORDER[type(t[1])], t[0]))
    if self.pkey is not None:
      metas.insert(0, ('pkey', self.pkey))
    # Collect all the fragments
    init_js, collect_js, filter_htmls = [], [], []
    for key, m in metas:
      full_key = ds_key + '_' + key
      ijs, cjs = _get_filter_js(m, full_key)
      init_js.append(ijs)
      collect_js.append((key, cjs))
      if isinstance(m, CompositionMetadata):
        filter_htmls.extend(_get_composition_filter_html(m, key, full_key))
      else:
        filter_htmls.append(_get_filter_html(m, key, full_key))
    return filter_htmls, init_js, collect_js

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


class UploadedSpectrumDataset(TrajDataset):
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
  if isinstance(m, BooleanMetadata):
    return '', '$("#%s").val()' % full_key
  if isinstance(m, NumericMetadata):
    elt = '$("#%s")' % full_key
    lb, ub = m.bounds
    init_js = ('%s.slider({min: %.17g, max: %.17g, step: %.17g, range: true, '
               'values: [%.17g,%.17g], slide: function(e,ui){'
               '$("#%s_label").text(ui.values[0]+" to "+ui.values[1]);}});') % (
                   elt, lb, ub, m.step, lb, ub, full_key)
    collect_js = elt + '.slider("values")'
    return init_js, collect_js
  if isinstance(m, DateMetadata):
    return '', '[$("#%s_lb").val(),$("#%s_ub").val()]' % (full_key, full_key)
  if isinstance(m, CompositionMetadata):
    init_parts, collect_parts = [], []
    for k, mm in m.comps.items():
      ijs, cjs = _get_filter_js(mm, full_key + '-' + k)
      init_parts.append(ijs)
      collect_parts.append('"%s": %s' % (k, cjs))
    collect_js = '{' + ','.join(collect_parts) + '}'
    return '\n'.join(init_parts), collect_js
  # only chosen selects remain (Lookup/PrimaryKey/Tag)
  # initialize the chosen dropdown, adding some width for the scrollbar
  init_js = '$("#%s_chooser").css("width", "+=20")' % full_key
  init_js += '.chosen({search_contains: true});'
  collect_js = 'multi_val($("#%s_chooser option:selected"))' % full_key
  if isinstance(m, (LookupMetadata, PrimaryKeyMetadata)):
    search_js = '$("#%s_search").val()' % full_key
    collect_js = '{select: %s, search: %s}' % (collect_js, search_js)
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
    return ('<div>%s: <span id="%s_label">%s to %s</span></div>'
            '<div class="slider" id="%s" style="background-image: '
            'url(data:img/png;base64,%s);"></div>') % (
                disp, full_key, lb, ub, full_key, m.hist_image)
  if isinstance(m, DateMetadata):
    lb, ub = map(str, np.array(m.bounds, dtype='datetime64[D]'))
    lb_input = '<input type="date" id="%s_lb" value="%s">' % (full_key, lb)
    ub_input = '<input type="date" id="%s_ub" value="%s">' % (full_key, ub)
    return '%s:<div>%s to %s</div>' % (disp, lb_input, ub_input)
  # only chosen selects remain (Lookup/PrimaryKey/Tag)
  html = u'%s:<select id="%s_chooser" data-placeholder="All" multiple>' % (
      disp, full_key)
  if isinstance(m, PrimaryKeyMetadata):
    uniques = sorted(m.keys)
  elif isinstance(m, TagMetadata):
    uniques = sorted(m.tags)
  else:
    uniques = m.uniques
  lines = (u'\n<option value="%s">%s</option>' % (x, xhtml_escape(x))
           for x in uniques)
  html += u''.join(lines) + u'\n</select>'
  if isinstance(m, (LookupMetadata, PrimaryKeyMetadata)):
    html += '\n<input type="text" placeholder="search" id="%s_search">' % (
        full_key)
  return html


def _get_composition_filter_html(m, key, full_key):
  comp_name = m.display_name(key)
  html_parts = []
  for k, m in m.comps.items():
    html = _get_filter_html(m, key + '-' + k, full_key + '-' + k)
    # HACK: insert composition name just inside the existing <div>
    html_parts.append('%s%s %s' % (html[:5], comp_name, html[5:]))
  return html_parts
