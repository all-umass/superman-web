from __future__ import absolute_import
import os
import logging
import numpy as np
from scipy.stats import linregress
from tornado.escape import json_encode

from ..web_datasets import DATASETS
from .base import BaseHandler


class CompositionPlotHandler(BaseHandler):
  def get(self, fignum):
    '''Downloads plot data as text.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    ax = fig_data.figure.gca()
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    if ax.collections:
      pc, = ax.collections
      xy_data = pc.get_offsets()
      kind = 'xy'
    else:
      xy_data = fig_data.hist_data
      kind = 'x' if xlabel else 'y'

    title = ax.get_title()
    if title:
      xlabel = title + ': ' + xlabel
      ylabel = title + ': ' + ylabel

    ds = DATASETS['LIBS']['Mars (big)']
    mask = fig_data.filter_mask[ds]
    sols = ds.metadata['sol'].get_array(mask)
    locs = ds.metadata['loc'].get_array(mask)
    shots = ds.metadata['shot'].get_array(mask)
    targets = ds.metadata['target'].get_array(mask)

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    if kind == 'xy':
      self.write('Target Name,Sol,Location,Shot,%s,%s\n' % (xlabel, ylabel))
      for i, (x, y) in enumerate(xy_data):
        self.write('%s,%s,%s,%s,%g,%g\n' % (targets[i], sols[i], locs[i],
                                            shots[i], x, y))
    else:
      labels = [t.get_text() for t in ax.legend_.texts]
      self.write('Target Name,Sol,Location,Shot,%s\n' % ','.join(labels))
      for i, t in enumerate(targets):
        vals = ','.join('%g' % data[i] for data in xy_data)
        self.write('%s,%s,%s,%s,%s\n' % (t, sols[i], locs[i], shots[i], vals))
    self.finish()

  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return
    ds = DATASETS['LIBS']['Mars (big)']
    do_fit = bool(int(self.get_argument('do_fit')))
    x_input = self.get_argument('x_comps')
    y_input = self.get_argument('y_comps')
    if (not x_input) and (not y_input):
      return
    elif (not x_input) or (not y_input):
      do_fit = False
    # input has the form: 'comp:element+comp:element'
    x_keys = [k.split('$',1) for k in x_input.split('+')] if x_input else []
    y_keys = [k.split('$',1) for k in y_input.split('+')] if y_input else []
    use_group_name = len(set(k[0] for k in (x_keys + y_keys))) > 1
    x_comps, x_labels = comps_with_labels(ds, x_keys, use_group_name)
    y_comps, y_labels = comps_with_labels(ds, y_keys, use_group_name)

    mask = fig_data.filter_mask[ds]

    if x_comps and y_comps:
      x_data = sum(m.get_array(mask) for m in x_comps)
      y_data = sum(m.get_array(mask) for m in y_comps)
    else:
      x_data = [m.get_array(mask) for m in x_comps]
      y_data = [m.get_array(mask) for m in y_comps]

    if do_fit:
      # handle NaNs
      notnan_mask = ~(np.isnan(x_data) | np.isnan(y_data))
      x_data = x_data[notnan_mask]
      y_data = y_data[notnan_mask]
      mask[mask] = notnan_mask

      logging.info('Running linregress on %d pairs of compositions: %s vs %s',
                   x_data.shape[0], x_labels, y_labels)

      # compute line of best fit
      slope, yint, rval, _, stderr = linregress(x_data, y_data)
      xint = -yint / slope
      results = dict(slope=slope,xint=xint,yint=yint,rval=rval**2,stderr=stderr)

      # make coordinates for the fit line
      x_min, x_max = x_data.min(), x_data.max()
      x_range = x_max - x_min
      fit_x = np.array([x_min - 0.25 * x_range, x_max + 0.25 * x_range])
      fit_y = fit_x * slope + yint
    else:
      results = {}

    fig_data.figure.clf(keep_observers=True)
    ax = fig_data.figure.gca()

    # set plot title (if needed)
    if not use_group_name:
      key, _ = (x_keys if x_keys else y_keys)[0]
      ax.set_title(ds.metadata[key].display_name(key))

    if x_comps and y_comps:
      # scatter plot: x vs y
      color_key = self.get_argument('color_by')
      color_meta = ds.metadata.get(color_key, None)
      if color_meta is not None:
        colors = color_meta.get_array(mask)
        sc = ax.scatter(x_data, y_data, c=colors)
        cbar = fig_data.figure.colorbar(sc)
        cbar.set_label(color_meta.display_name(color_key))
      else:
        ax.scatter(x_data, y_data)
      ax.set_xlabel(' + '.join(x_labels))
      ax.set_ylabel(' + '.join(y_labels))
    elif x_comps:
      # histogram along x
      ax.hist(x_data, bins='auto', orientation='vertical', label=x_labels)
      fig_data.hist_data = x_data
      ax.legend()
    else:
      # histogram along y
      ax.hist(y_data, bins='auto', orientation='horizontal', label=y_labels)
      fig_data.hist_data = y_data
      ax.legend()

    # get the plot bounds
    ax.autoscale_view()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    # plot the best fit line (optionally)
    if do_fit:
      ax.plot(fit_x, fit_y, 'k--')
      ax.set_xlim(xlim)
      ax.set_ylim(ylim)

    # draw!
    fig_data.manager.canvas.draw()

    # respond with fit parameters and zoom info
    results['zoom'] = (xlim[0], xlim[1], ylim[0], ylim[1])
    return self.write(json_encode(results))


def comps_with_labels(ds, comp_keys, use_group_name):
  comps, labels = [], []
  for k1, k2 in comp_keys:
    m1 = ds.metadata[k1]
    m2 = m1.comps[k2]
    comps.append(m2)
    name = m2.display_name(k2)
    if use_group_name:
      labels.append('(%s: %s)' % (m1.display_name(k1), name))
    else:
      labels.append(name)
  return comps, labels


routes = [
    (r'/_plot_compositions', CompositionPlotHandler),
    (r'/([0-9]+)/compositions\.csv', CompositionPlotHandler),
]
