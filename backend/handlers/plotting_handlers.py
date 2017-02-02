from __future__ import absolute_import
import ast
import logging
import numpy as np
import os
from collections import namedtuple
from itertools import cycle, islice
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from six.moves import xrange
from threading import Thread
from tornado import gen

from .base import MultiDatasetHandler

# old matplotlib used a different key
if 'axes.prop_cycle' not in rcParams:
  COLOR_CYCLE = rcParams['axes.color_cycle']
else:
  ccycle = rcParams['axes.prop_cycle']
  # newest matplotlib has a convenience wrapper
  if hasattr(ccycle, 'by_key'):
    COLOR_CYCLE = ccycle.by_key()['color']
  else:
    COLOR_CYCLE = [c['color'] for c in ccycle]

# helper for storing validated/processed inputs
AxisInfo = namedtuple('AxisInfo', ('type', 'argument'))
# helpers for storing computed data
ColorData = namedtuple('ColorData', ('color', 'label', 'names', 'needs_cbar',
                                     'is_categorical'))
PlotData = namedtuple('PlotData', ('trajs', 'xlabel', 'ylabel', 'xticks',
                                   'yticks', 'scatter'))


class FilterPlotHandler(MultiDatasetHandler):
  def get(self, fignum, download_type):
    '''Downloads plot data as text.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return
    if fig_data.explorer_data is None or fig_data.last_plot != 'filterplot':
      self.write('No plotted data to download.')
      return
    all_ds = self.request_many_ds()
    if not all_ds:
      self.write('No datasets selected.')
      return
    all_ds_views = [ds.view(mask=fig_data.filter_mask[ds]) for ds in all_ds]

    pkeys = []
    for dv in all_ds_views:
      pkeys.extend(map(_sanitize_csv, dv.get_primary_keys()))

    as_matrix = bool(int(self.get_argument('as_matrix', '0')))

    if download_type == 'metadata':
      meta_rows = self._prep_metadata(pkeys, all_ds_views)
    elif download_type == 'spectra':
      if as_matrix:
        try:
          bands, ints = self._prep_vector_spectra(fig_data)
        except ValueError as e:
          self.write(e.message)
          return
      else:
        lines, xlabel, ylabel = self._prep_traj_spectra(fig_data)
    else:
      self.write('Unknown download_type: %s' % download_type)
      return

    # set up for writing the response file
    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition', 'attachment; filename='+fname)

    if download_type == 'metadata':
      for row in meta_rows:
        self.write(','.join(row))
        self.write('\n')
    elif as_matrix:
      self.write('wave,%s\n' % ','.join(pkeys))
      for i, row in enumerate(ints.T):
        self.write('%g,%s\n' % (bands[i], ','.join('%g' % y for y in row)))
    else:
      self.write('pkey,axis')
      for i, traj in enumerate(lines):
        self.write('\n%s,%s,' % (pkeys[i], xlabel))
        self.write(','.join('%g' % x for x in traj[:,0]))
        self.write('\n,%s,' % ylabel)
        self.write(','.join('%g' % y for y in traj[:,1]))
      self.write('\n')
    self.finish()

  def _prep_traj_spectra(self, fig_data):
    ax = fig_data.figure.gca()
    xlabel = _sanitize_csv(ax.get_xlabel()) or 'x'
    ylabel = _sanitize_csv(ax.get_ylabel()) or 'y'
    return fig_data.explorer_data.trajs, xlabel, ylabel

  def _prep_vector_spectra(self, fig_data):
    # convert to vector shape
    bands, ints = None, []
    for traj in fig_data.explorer_data.trajs:
      x, y = traj.T
      if bands is None:
        bands = x
      else:
        if x.shape != bands.shape or not np.allclose(x, bands):
          raise ValueError('Mismatching wavelength data.')
      ints.append(y)
    ints = np.vstack(ints)
    return bands, ints

  def _prep_metadata(self, pkeys, all_ds_views):
    header = ['pkey']
    meta_columns = [pkeys]

    # make a dataset column, if there are multiple datasets
    if len(all_ds_views) > 1:
      header.append('Dataset')
      ds_names = [str(dv.ds) for dv in all_ds_views]
      counts = [np.count_nonzero(dv.mask) for dv in all_ds_views]
      meta_columns.append(np.repeat(ds_names, counts))

    # collect the requested meta info
    for meta_key in self.get_arguments('meta_keys[]'):
      data = []
      for dv in all_ds_views:
        x, label = dv.get_metadata(meta_key)
        data.append(x)
      header.append(label)
      meta_columns.append(np.concatenate(data))

    # transpose into rows
    rows = [header]
    for i in xrange(len(pkeys)):
      rows.append([str(col[i]) for col in meta_columns])
    return rows

  @gen.coroutine
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.visible_error(403, "Broken connection to server.")
      return

    # parse plot information from request arguments
    xaxis = self._get_axis_info('x')
    yaxis = self._get_axis_info('y')
    caxis = self._get_color_info()
    plot_kwargs = dict(
        cmap=self.get_argument('cmap'),
        alpha=float(self.get_argument('alpha')),
        lw=float(self.get_argument('line_width')),
    )

    all_ds_views, num_spectra = self.prepare_ds_views(fig_data)
    if all_ds_views is None:
      # not an error, just nothing to do
      self.write('{}')
      return

    # check to see if anything changed since the last view we had
    view_params = [(k, self.get_argument(k)) for k in self.request.arguments
                   if k.startswith('blr_')]
    trans = all_ds_views[0].transformations
    for k in ['chan_mask', 'pp', 'crop']:
      view_params.append((k, trans[k]))
    view_params = tuple(sorted(view_params))
    # if the view changed, invalidate the cache
    if view_params != fig_data.explorer_view_params:
      fig_data.clear_explorer_cache()
      fig_data.explorer_view_params = view_params

    # get individual line/point labels (TODO: cache this)
    if len(all_ds_views) == 1:
      pkeys = all_ds_views[0].get_primary_keys()
    else:
      pkeys = []
      for dv in all_ds_views:
        dv_name = dv.ds.name
        for k in dv.get_primary_keys():
          pkeys.append('%s: %s' % (dv_name, k))

    # get the plot data (trajs or scatter points), possibly cached
    if xaxis != fig_data.explorer_xaxis or yaxis != fig_data.explorer_yaxis:
      plot_data = self._get_plot_data(all_ds_views, xaxis, yaxis)
      if plot_data is None:
        # self.visible_error has already been called by _get_plot_data
        return
      fig_data.explorer_xaxis = xaxis
      fig_data.explorer_yaxis = yaxis
      fig_data.explorer_data = plot_data
    else:
      plot_data = fig_data.explorer_data

    # get the color data, possibly cached
    if caxis != fig_data.explorer_caxis:
      color_data = _get_color_data(all_ds_views, caxis)
      fig_data.explorer_caxis = caxis
      fig_data.explorer_color = color_data
    else:
      color_data = fig_data.explorer_color

    # prepare the figure
    fig_data.figure.clf(keep_observers=True)

    # generate, decorate, and draw the plot
    do_legend = bool(int(self.get_argument('legend')))
    ax = yield gen.Task(_async_draw_plot, fig_data, plot_data, color_data,
                        plot_kwargs, do_legend, pkeys)
    # return the axis limits
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    self.write_json([xmin,xmax,ymin,ymax])

  def _get_axis_info(self, ax_char):
    atype = self.get_argument(ax_char + 'axis')
    if atype == 'metadata':
      argument = self.get_argument(ax_char + '_metadata')
    elif atype == 'line_ratio':
      argument = ast.literal_eval(self.get_argument(ax_char + '_line_ratio'))
    elif atype in ('default', 'cardinal'):
      argument = None
    elif atype == 'computed':
      argument = self.get_argument(ax_char + '_computed')
    else:
      raise ValueError('Invalid axis type: %s' % atype)
    return AxisInfo(type=atype, argument=argument)

  def _get_color_info(self):
    ctype = self.get_argument('color')
    if ctype == 'default':
      argument = self.get_argument('fixed_color')
    elif ctype == 'cycled':
      argument = None
    elif ctype == 'metadata':
      argument = self.get_argument('color_by')
    elif ctype == 'line_ratio':
      argument = ast.literal_eval(self.get_argument('color_line_ratio'))
    elif ctype == 'computed':
      argument = self.get_argument('color_computed')
    else:
      raise ValueError('Invalid color type: %s' % ctype)
    return AxisInfo(type=ctype, argument=argument)

  def _get_plot_data(self, all_ds_views, xaxis, yaxis):
    logging.info('Getting new plot data: %s, %s', xaxis, yaxis)
    if xaxis.type == 'default' and yaxis.type == 'default':
      trajs, xlabels = [], set()
      for dv in all_ds_views:
        xlabels.add(dv.ds.x_axis_units())
        try:
          tt = dv.get_trajectories()
        except ValueError as e:
          logging.exception('Failed to get data from %s', dv.ds)
          self.visible_error(400, e.message)
          return None
        else:
          trajs.extend(tt)
      return PlotData(scatter=False, trajs=trajs, xlabel=' & '.join(xlabels),
                      ylabel='Intensity', xticks=None, yticks=None)
    # scatter
    try:
      xdata, xlabel, xticks = _get_axis_data(all_ds_views, xaxis)
      ydata, ylabel, yticks = _get_axis_data(all_ds_views, yaxis)
    except Exception as e:
      logging.exception('Failed to get axis data.')
      self.visible_error(400, e.message)
      return None

    if xdata is None or ydata is None:
      self.visible_error(400, 'Bad axis type combination.',
                         'Invalid axis combo: %s vs %s', xaxis, yaxis)
      return None

    trajs = [np.column_stack((xdata, ydata))]
    return PlotData(scatter=True, trajs=trajs, xlabel=xlabel, ylabel=ylabel,
                    xticks=xticks, yticks=yticks)


def _async_draw_plot(fig_data, plot_data, color_data, plot_kwargs, do_legend,
                     pkeys, callback=None):
  fig = fig_data.figure

  def helper():
    ax = fig.gca()
    artist = _add_plot(fig, ax, plot_data, color_data, pkeys, **plot_kwargs)
    _decorate_plot(fig, ax, artist, plot_data, color_data,
                   do_legend, plot_kwargs['cmap'])
    fig_data.manager.canvas.draw()
    fig_data.last_plot = 'filterplot'
    callback(ax)

  t = Thread(target=helper)
  t.daemon = True
  t.start()


def _get_color_data(all_ds_views, caxis):
  logging.info('Getting new color data: %s', caxis)
  color, label, names = _get_axis_data(all_ds_views, caxis)
  return ColorData(color=color, label=label, names=names,
                   needs_cbar=(names is None and label is not None),
                   is_categorical=(names is not None and label is not None))


def _add_plot(fig, ax, plot_data, color_data, pkeys, lw=1, cmap='_auto',
              alpha=1.0):
  colors = color_data.color
  if cmap == '_auto':
    cmap = None
    if color_data.is_categorical:
      colors = np.take(COLOR_CYCLE, colors, mode='wrap')

  if plot_data.scatter:
    data, = plot_data.trajs
    artist = ax.scatter(*data.T, marker='o', c=colors, edgecolor='none',
                        s=lw*20, cmap=cmap, alpha=alpha, picker=5)
  else:
    # trajectory plot
    if hasattr(colors, 'dtype') and np.issubdtype(colors.dtype, np.number):
      # delete lines with NaN colors
      mask = np.isfinite(colors)
      if mask.all():
        trajs = plot_data.trajs
      else:
        trajs = [t for i,t in enumerate(plot_data.trajs) if mask[i]]
        colors = colors[mask]
      artist = LineCollection(trajs, linewidths=lw, cmap=cmap)
      artist.set_array(colors)
    else:
      artist = LineCollection(plot_data.trajs, linewidths=lw, cmap=cmap)
      artist.set_color(colors)
    artist.set_alpha(alpha)
    artist.set_picker(True)
    artist.set_pickradius(5)
    ax.add_collection(artist, autolim=True)
    ax.autoscale_view()
    # Force ymin -> 0
    ax.set_ylim((0, ax.get_ylim()[1]))

  def on_pick(event):
    if event.artist is not artist:
      return
    label = pkeys[event.ind[0]]
    ax.set_title(label)
    fig.canvas.draw_idle()

  # XXX: hack, make this more official
  if hasattr(fig, '_superman_cb'):
    fig.canvas.mpl_disconnect(fig._superman_cb[0])
  cb_id = fig.canvas.mpl_connect('pick_event', on_pick)
  fig._superman_cb = (cb_id, on_pick)
  return artist


def _decorate_plot(fig, ax, artist, plot_data, color_data, legend, cmap):
  ax.set_xlabel(plot_data.xlabel)
  ax.set_ylabel(plot_data.ylabel)

  if plot_data.xticks is not None:
    ax.set_xticks(np.arange(len(plot_data.xticks)))
    ax.set_xticklabels(plot_data.xticks)
  if plot_data.yticks is not None:
    ax.set_yticks(np.arange(len(plot_data.yticks)))
    ax.set_yticklabels(plot_data.yticks)

  if color_data.needs_cbar:
    cbar = fig.colorbar(artist)
    cbar.set_label(color_data.label)
    return

  if legend and color_data.names is not None and len(color_data.names) <= 20:
    # using trick from http://stackoverflow.com/a/19881647/10601
    num_colors = len(color_data.names)
    if cmap == '_auto':
      colors = islice(cycle(COLOR_CYCLE), num_colors)
    else:
      colors = artist.cmap(np.linspace(0, 1, num_colors))
    proxies = [Patch(color=c, label=k) for c,k in zip(colors, color_data.names)]

    # Make a new skinny subplot, then cover it with the legend.
    # This allows an out-of-axis legend without it getting cut off.
    # Uses the trick from http://stackoverflow.com/a/22885327/10601
    gs = GridSpec(1, 2, width_ratios=(3, 1), wspace=0.02)
    ax.set_position(gs[0].get_position(fig))
    ax.set_subplotspec(gs[0])
    lx = fig.add_subplot(gs[1])
    lx.legend(handles=proxies, title=color_data.label, loc='upper left',
              mode='expand', borderaxespad=0, fontsize='small', frameon=False)
    lx.axis('off')
    gs.tight_layout(fig, w_pad=0)


def _sanitize_csv(s):
  if ',' in str(s):
    return '"%s"' % s
  return s


def _get_axis_data(all_ds_views, axis):
  label, tick_names = None, None
  if axis.type == 'default':
    data = axis.argument
  elif axis.type == 'metadata':
    data, label, tick_names = _get_all_metadata(all_ds_views, axis.argument)
  elif axis.type == 'line_ratio':
    data, label = _compute_line_ratios(all_ds_views, axis.argument)
  elif axis.type == 'cardinal':
    n = sum(np.count_nonzero(dv.mask) for dv in all_ds_views)
    data = np.arange(n)
    label = ''  # TODO: find a reasonable label for this
  elif axis.type == 'computed':
    data, label = _compute_expr(all_ds_views, axis.argument)
  elif axis.type == 'cycled':
    # use the default color cycle from matplotlib
    data = COLOR_CYCLE
    if all(dv.ds.pkey is not None for dv in all_ds_views):
      tick_names = np.concatenate([dv.ds.pkey.index2key(dv.mask)
                                   for dv in all_ds_views])
  return data, label, tick_names


def _get_all_metadata(all_ds_views, meta_key):
  # semi-hack: this is hard-coded for the "color-by-dataset" case
  if meta_key == '_ds':
    tick_names = [dv.ds.name for dv in all_ds_views]
    counts = [np.count_nonzero(dv.mask) for dv in all_ds_views]
    data = np.repeat(np.arange(len(counts)), counts)
    label = 'Dataset'
    return data, 'Dataset', tick_names
  # normal metadata lookup
  data = []
  for dv in all_ds_views:
    x, label = dv.get_metadata(meta_key)
    data.append(x)
  data = np.concatenate(data)
  if not np.issubdtype(data.dtype, np.number):
    # Categorical case
    tick_names, data = np.unique(data, return_inverse=True)
  else:
    tick_names = None
  return data, label, tick_names


def _compute_line_ratios(all_ds_views, lines):
  # TODO: use existing plot data here, instead of recomputing it
  data = np.concatenate([dv.compute_line(lines) for dv in all_ds_views])
  if len(lines) == 4:
    label = '(%g to %g) / (%g to %g)' % tuple(lines)
  else:
    label = '%g to %g' % tuple(lines)
  return data, label


def _compute_expr(all_ds_views, expr):
  # get all the trajectories without NaN gapping
  trajs = []
  for dv in all_ds_views:
    nan_gap = dv.transformations.get('nan_gap', None)
    dv.transformations['nan_gap'] = None
    trajs.extend(dv.get_trajectories())
    dv.transformations['nan_gap'] = nan_gap

  logging.info('Compiling user expression: %r', expr)
  expr_code = compile(expr, '<string>', 'eval')

  def _expr_eval(t):
    res = eval(expr_code, np.__dict__, dict(x=t[:,0],y=t[:,1]))
    return float(res)
  return np.array(list(map(_expr_eval, trajs))), expr


routes = [
    (r'/_filterplot', FilterPlotHandler),
    (r'/([0-9]+)/(spectra|metadata)\.csv', FilterPlotHandler),
]
