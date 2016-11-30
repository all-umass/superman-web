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
from threading import Thread
from tornado import gen
from tornado.escape import json_encode
from six.moves import xrange

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
  def get(self, fignum):
    '''Downloads plot data as text.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return
    if fig_data.explorer_data is None or fig_data.last_plot != 'filterplot':
      self.write('No plotted data to download.')
      return

    lines = fig_data.explorer_data.trajs
    ax = fig_data.figure.gca()
    xlabel = _sanitize_csv(ax.get_xlabel()) or 'x'
    ylabel = _sanitize_csv(ax.get_ylabel()) or 'y'

    all_ds = self.request_many_ds()
    if not all_ds:
      self.write('No datasets selected.')
      return
    all_ds_views = [ds.view(mask=fig_data.filter_mask[ds]) for ds in all_ds]

    # make the UID (line names) column
    header = ['UID']
    line_names = []
    for ds, dv in zip(all_ds, all_ds_views):
      if ds.pkey is None:
        n = np.count_nonzero(dv.mask)
        line_names.extend(['_line%d' % i for i in xrange(n)])
      else:
        names = ds.pkey.index2key(dv.mask)
        line_names.extend(list(map(_sanitize_csv, names)))
    meta_data = [line_names]

    # make a dataset column, if there are multiple datasets
    if len(all_ds_views) > 1:
      header.append('Dataset')
      ds_names = map(str, all_ds)
      counts = [np.count_nonzero(dv.mask) for dv in all_ds_views]
      meta_data.append(np.repeat(ds_names, counts))

    # collect the requested meta info
    for meta_key in self.get_arguments('meta_keys[]'):
      data = []
      for dv in all_ds_views:
        x, label = dv.get_metadata(meta_key)
        data.append(x)
      header.append(label)
      meta_data.append(np.concatenate(data))

    # set up for writing the response file
    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    # write the header line
    self.write('%s,Axis\n' % ','.join(header))
    # write each spectrum with associated metadata
    for i, traj in enumerate(lines):
      self.write(','.join(str(m[i]) for m in meta_data))
      self.write(',%s,' % xlabel)
      self.write(','.join('%g' % x for x in traj[:,0]))
      self.write('\n%s%s,' % (',' * len(header), ylabel))
      self.write(','.join('%g' % y for y in traj[:,1]))
      self.write('\n')
    self.finish()

  @gen.coroutine
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.set_status(403)
      return

    all_ds_views, num_spectra = self.prepare_ds_views(fig_data)
    if all_ds_views is None:
      self.write('{}')
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

    # get the plot data (trajs or scatter points), possibly cached
    if xaxis != fig_data.explorer_xaxis or yaxis != fig_data.explorer_yaxis:
      plot_data = _get_plot_data(all_ds_views, xaxis, yaxis)
      if plot_data is None:
        logging.error('Invalid axis combo: %s vs %s', xaxis, yaxis)
        self.set_status(400)  # bad request
        self.write('Bad axis type combination.')
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
    fig = fig_data.figure
    if bool(int(self.get_argument('clear'))):
      fig.clf(keep_observers=True)

    # generate, decorate, and draw the plot
    do_legend = bool(int(self.get_argument('legend')))
    ax = yield gen.Task(_async_draw_plot, fig_data, plot_data, color_data,
                        plot_kwargs, do_legend)

    # return the axis limits
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    self.write(json_encode([xmin,xmax,ymin,ymax]))

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


def _async_draw_plot(fig_data, plot_data, color_data, plot_kwargs, do_legend,
                     callback=None):
  fig = fig_data.figure

  def helper():
    ax = fig.gca()
    artist = _add_plot(fig, ax, plot_data, color_data, **plot_kwargs)
    _decorate_plot(fig, ax, artist, plot_data, color_data,
                   do_legend, plot_kwargs['cmap'])
    fig_data.manager.canvas.draw()
    fig_data.last_plot = 'filterplot'
    callback(ax)

  t = Thread(target=helper)
  t.daemon = True
  t.start()


def _get_plot_data(all_ds_views, xaxis, yaxis):
  logging.info('Getting new plot data: %s, %s', xaxis, yaxis)
  if xaxis.type == 'default' and yaxis.type == 'default':
    trajs, xlabels = [], set()
    for dv in all_ds_views:
      trajs.extend(dv.get_trajectories())
      xlabels.add(dv.ds.x_axis_units())
    return PlotData(trajs=trajs, xlabel=' & '.join(xlabels), ylabel='Intensity',
                    xticks=None, yticks=None, scatter=False)
  # scatter
  xdata, xlabel, xticks = _get_axis_data(all_ds_views, xaxis)
  ydata, ylabel, yticks = _get_axis_data(all_ds_views, yaxis)
  if xdata is None or ydata is None:
    return None
  return PlotData(trajs=[np.column_stack((xdata, ydata))], xlabel=xlabel,
                  ylabel=ylabel, xticks=xticks, yticks=yticks, scatter=True)


def _get_color_data(all_ds_views, caxis):
  logging.info('Getting new color data: %s', caxis)
  color, label, names = _get_axis_data(all_ds_views, caxis)
  return ColorData(color=color, label=label, names=names,
                   needs_cbar=(names is None and label is not None),
                   is_categorical=(names is not None and label is not None))


def _add_plot(fig, ax, plot_data, color_data, lw=1, cmap='_auto', alpha=1):
  colors = color_data.color
  if cmap == '_auto':
    cmap = None
    if color_data.is_categorical:
      colors = np.take(COLOR_CYCLE, colors, mode='wrap')

  if plot_data.scatter:
    data, = plot_data.trajs
    return ax.scatter(*data.T, marker='o', c=colors, edgecolor='none',
                      s=lw*20, cmap=cmap, alpha=alpha)
  # trajectory plot
  if hasattr(colors, 'dtype') and np.issubdtype(colors.dtype, np.number):
    # delete lines with NaN colors
    mask = np.isfinite(colors)
    if mask.all():
      trajs = plot_data.trajs
    else:
      trajs = [t for i,t in enumerate(plot_data.trajs) if mask[i]]
      colors = colors[mask]
    lc = LineCollection(trajs, linewidths=lw, cmap=cmap)
    lc.set_array(colors)
  else:
    lc = LineCollection(plot_data.trajs, linewidths=lw, cmap=cmap)
    lc.set_color(colors)
  ax.add_collection(lc, autolim=True)
  lc.set_alpha(alpha)
  ax.autoscale_view()
  # Force ymin -> 0
  ax.set_ylim((0, ax.get_ylim()[1]))
  return lc


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
    (r'/([0-9]+)/spectra\.csv', FilterPlotHandler),
]
