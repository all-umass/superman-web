from __future__ import absolute_import
import ast
import logging
import numpy as np
import os
from itertools import cycle
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from tornado.escape import json_encode
from six.moves import xrange

from .base import BaseHandler
from .baseline_handlers import setup_blr_object

if 'axes.prop_cycle' in rcParams:
  COLOR_CYCLE = rcParams['axes.prop_cycle'].by_key()['color']
else:
  COLOR_CYCLE = rcParams['axes.color_cycle']


class FilterPlotHandler(BaseHandler):
  def get(self, fignum):
    '''Downloads plot data as text.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    lines = fig_data.explorer_data
    ax = fig_data.figure.gca()
    xlabel = _sanitize_csv(ax.get_xlabel()) or 'x'
    ylabel = _sanitize_csv(ax.get_ylabel()) or 'y'

    ds = self.get_dataset(self.get_argument('ds_kind'),
                          self.get_argument('ds_name'))
    ds_view = ds.view(mask=fig_data.filter_mask)

    # make the UID column
    header = ['UID']
    if ds.pkey is None:
      meta_data = [['_line%d' % i for i in xrange(len(lines))]]
    else:
      names = ds.pkey.index2key(fig_data.filter_mask)
      meta_data = [list(map(_sanitize_csv, names))]

    # collect the requested meta info
    meta_keys = filter(None, self.get_argument('meta_keys').split(','))
    for meta_key in meta_keys:
      try:
        data, label, names = _masked_metadata(ds_view, meta_key)
      except KeyError:
        logging.warn('Invalid meta_key: %r', meta_key)
        continue
      header.append(label)
      if names is not None:
        data = names[data]
      meta_data.append(data)

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

  def post(self):
    fig_data = self.get_fig_data()
    ds = self.get_dataset(self.get_argument('ds_kind'),
                          self.get_argument('ds_name'))
    if fig_data is None or ds is None:
      return

    mask = fig_data.filter_mask
    num_spectra = np.count_nonzero(mask)

    logging.info('FilterPlot request for %d spectra', num_spectra)
    if not 0 < num_spectra <= 99999:
      logging.warning('Data not plottable, %d spectra chosen', num_spectra)
      return

    bl_obj, segmented, lb, ub, _ = setup_blr_object(self)
    chan_mask = bool(int(self.get_argument('chan_mask', 0)))
    ds_view = ds.view(mask=mask, pp=self.get_argument('pp', ''),
                      blr_obj=bl_obj, blr_segmented=segmented,
                      crop=(lb, ub), chan_mask=chan_mask)

    color, color_label, color_names = _get_color(
        ds_view, self.get_argument('color'),
        self.get_argument('fixed_color'),
        self.get_argument('color_by'),
        ast.literal_eval(self.get_argument('color_line_ratio')),
        self.get_argument('color_computed'))

    xaxis_type = self.get_argument('xaxis')
    yaxis_type = self.get_argument('yaxis')

    lw = float(self.get_argument('line_width'))
    plot_kwargs = dict(
        cmap=self.get_argument('cmap'),
        alpha=float(self.get_argument('alpha')),
    )

    if bool(int(self.get_argument('clear'))):
      fig_data.figure.clf(keep_observers=True)
    ax = fig_data.figure.gca()
    legend = bool(int(self.get_argument('legend')))

    if xaxis_type == 'default' and yaxis_type == 'default':
      plot_kwargs['legend'] = legend
      trajs = ds_view.get_trajectories()
      _plot_trajs(fig_data.figure, ax, trajs, color,
                  color_label, color_names, lw=lw, **plot_kwargs)
      xlabel = ds.x_axis_units()
      ylabel = 'Intensity'
      fig_data.explorer_data = trajs
    else:
      xdata, xlabel, xticks = _get_axis_data(
          ds_view, xaxis_type, self.get_argument('x_metadata'),
          ast.literal_eval(self.get_argument('x_line_ratio')))
      ydata, ylabel, yticks = _get_axis_data(
          ds_view, yaxis_type, self.get_argument('y_metadata'),
          ast.literal_eval(self.get_argument('y_line_ratio')))
      if xdata is None or ydata is None:
        logging.error('Invalid axis type combo: xaxis=%s, yaxis=%s',
                      xaxis_type, yaxis_type)
        self.set_status(403)
        return self.finish(dict(message='Invalid axis type combination'))

      logging.info('Scatter plot: %d x, %d y', len(xdata), len(ydata))
      sc = ax.scatter(xdata, ydata, marker='o', c=color, edgecolor='none',
                      s=lw*20, **plot_kwargs)
      fig_data.explorer_data = [np.column_stack((xdata, ydata))]

      if xticks is not None:
        ax.set_xticks(np.arange(len(xticks)))
        ax.set_xticklabels(xticks)
      if yticks is not None:
        ax.set_yticks(np.arange(len(yticks)))
        ax.set_yticklabels(yticks)
      _apply_plot_labels(fig_data.figure, ax, sc, color_label, color_names,
                         legend=legend)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig_data.manager.canvas.draw()

    # return the axis limits
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    return self.write(json_encode([xmin,xmax,ymin,ymax]))


def _sanitize_csv(s):
  if ',' in s:
    return '"%s"' % s
  return s


def _get_axis_data(ds_view, axis_type, key, lines):
  ticks = None
  if axis_type == 'metadata':
    data, label, ticks = _masked_metadata(ds_view, key)
    label = label.decode('utf8', 'ignore')
  elif axis_type == 'line_ratio':
    data, label = _compute_line_ratios(ds_view, lines)
  elif axis_type == 'cardinal':
    data = np.arange(np.count_nonzero(ds_view.mask))
    label = ''
  else:
    data, label = None, None
  return data, label, ticks


def _get_color(ds_view, ctype, fixed, key, lines, expr):
  names, label = None, None
  if ctype == 'default':
    color = fixed
  elif ctype == 'cycled':
    # use the default color cycle from matplotlib
    color = COLOR_CYCLE
    if ds_view.ds.pkey is not None:
      names = ds_view.ds.pkey.index2key(ds_view.mask)
  elif ctype == 'metadata':
    color, label, names = _masked_metadata(ds_view, key)
  elif ctype == 'line_ratio':
    color, label = _compute_line_ratios(ds_view, lines)
  elif ctype == 'computed':
    color = _compute_expr(ds_view, expr)
    label = expr
  else:
    raise ValueError('Invalid color type: %s' % ctype)
  return color, label, names


def _plot_trajs(fig, ax, trajs, color, color_label, color_names,
                lw=1, cmap='jet', alpha=1, legend=True):
  do_legend = (legend and color_names is not None and
               len(trajs) < 20 and len(color_names) == len(trajs) and
               not isinstance(color, np.ndarray))
  if do_legend:
    # plot manually to avoid proxy lines for legend
    # see http://stackoverflow.com/q/19877666/10601
    for t,c,l in zip(trajs, cycle(color), color_names):
      ax.plot(t[:,0], t[:,1], color=c, label=l, lw=lw, alpha=alpha)
    ax.legend()
  else:
    # no legend, or too many lines, so use a line collection
    lc = LineCollection(trajs, linewidths=lw, cmap=cmap)
    if color_label is None:
      lc.set_color(color)
    else:
      lc.set_array(color)
    ax.add_collection(lc, autolim=True)
    _apply_plot_labels(fig, ax, lc, color_label, color_names, legend=legend)
    lc.set_alpha(alpha)
    ax.autoscale_view()
  # Force ymin -> 0
  ax.set_ylim((0, ax.get_ylim()[1]))


def _masked_metadata(ds_view, meta_key):
  data, label = ds_view.get_metadata(meta_key)
  if not np.issubdtype(data.dtype, np.number):
    # Categorical case
    names, data = np.unique(data, return_inverse=True)
    return data, label, names
  return data, label, None


def _compute_line_ratios(ds_view, lines):
  data = ds_view.compute_line(lines)
  if len(lines) == 4:
    label = '(%g to %g) / (%g to %g)' % tuple(lines)
  else:
    label = '%g to %g' % tuple(lines)
  return data, label


def _compute_expr(ds_view, expr):
  trajs = ds_view.get_trajectories()
  logging.info('Compiling user expression: %r', expr)
  expr_code = compile(expr, '<string>', 'eval')

  def _expr_eval(t):
    res = eval(expr_code, np.__dict__, dict(x=t[:,0],y=t[:,1]))
    return float(res)
  return np.array(list(map(_expr_eval, trajs)))


def _apply_plot_labels(fig, ax, artist, label, names, legend=True):
  if names is None:
    if label is not None:
      cbar = fig.colorbar(artist)
      cbar.set_label(label)
    return
  if not legend or len(names) > 20:
    return
  # using trick from http://stackoverflow.com/a/19881647/10601
  proxies = [Rectangle((0,0), 1, 1, fc=c)
             for c in artist.cmap(np.linspace(0, 1, len(names)))]
  ax.legend(proxies, names)


routes = [
    (r'/_filterplot', FilterPlotHandler),
    (r'/([0-9]+)/spectra\.csv', FilterPlotHandler),
]
