from __future__ import absolute_import
import logging
import numpy as np
import os
import scipy.integrate
from superman.peaks.bump_fit import fit_single_peak
from tornado.escape import json_encode

from .base import BaseHandler


class PeakHandler(BaseHandler):
  def get(self, fignum):
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return
    ds = self.get_dataset(self.get_argument('ds_kind'),
                          self.get_argument('ds_name'))
    if ds is None or ds.pkey is None:
      logging.warn("Can't do batch peak area calc without a pkey")
      return

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)

    alg = self.get_argument('alg', 'manual')
    if alg == 'manual':
      base_type = self.get_argument('basetype')
      lb = float(self.get_argument('lb'))
      ub = float(self.get_argument('ub'))
      bounds = sorted((lb,ub))
      header = ['Name', 'Area', 'Height', 'X min', 'X max']
      peak_keys = ('area', 'height', 'xmin', 'xmax')

      def peak_stats(t):
        return _manual_peak_area(t, bounds, base_type)[-1]

    elif alg == 'fit':
      kind = self.get_argument('fitkind')
      loc = float(self.get_argument('fitloc'))
      xres = float(self.get_argument('xres'))
      loc_fixed = bool(int(self.get_argument('locfixed')))
      header = ['Name', 'Area', 'Height', 'Center', 'FWHM', 'X min', 'X max',
                'Area stdv', 'Center stdv', 'FWHM stdv']
      peak_keys = ('area', 'height', 'center', 'fwhm', 'xmin', 'xmax',
                   'area_std', 'center_std', 'fwhm_std')

      def peak_stats(t):
        return fit_single_peak(t[:,0], t[:,1], loc, fit_kind=kind,
                               log_fn=logging.info, band_resolution=xres,
                               loc_fixed=loc_fixed)[-1]

    mask = fig_data.filter_mask[ds]
    trans = fig_data.get_trans()
    trans['nan_gap'] = None  # make sure we're not inserting NaNs anywhere
    ds_view = ds.view(mask=mask, **trans)
    trajs, names = ds_view.get_trajectories(return_keys=True)

    meta_data = []
    if bool(int(self.get_argument('include_metadata'))):
      for meta_key, _ in ds.metadata_names():
        x, label = ds_view.get_metadata(meta_key)
        header.append(label)
        meta_data.append(x)

    self.write(','.join(header))
    self.write('\n')
    for i, traj in enumerate(trajs):
      peak = peak_stats(traj)
      self.write('%s,' % names[i])
      self.write(','.join(str(peak[k]) for k in peak_keys))
      self.write(''.join(',%s' % m[i] for m in meta_data))
      self.write('\n')
    self.finish()

  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return
    spectrum = fig_data.get_trajectory()

    # set up for plotted overlay(s)
    ax = fig_data.figure.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.cla()
    ax.plot(spectrum[:,0], spectrum[:,1], '-')
    ax.set_title(fig_data.title)

    # re-fill NaN-gapped values (semi-hack)
    nan_inds, = np.where(np.isnan(spectrum[:,1]))
    if len(nan_inds) > 0:
      spectrum[nan_inds,1] = spectrum[nan_inds+1,1]

    alg = self.get_argument('alg')
    logging.info('Peak fitting with alg=%r', alg)
    if alg == 'manual':
      base_type = self.get_argument('basetype')
      lb = float(self.get_argument('lb'))
      ub = float(self.get_argument('ub'))
      bounds = sorted((lb,ub))
      peak_x, peak_y, base, peak_data = _manual_peak_area(spectrum, bounds,
                                                          base_type)
      # show the area we integrated
      ax.fill_between(peak_x, base, peak_y, facecolor='gray', alpha=0.5)
    elif alg == 'fit':
      kind = self.get_argument('fitkind')
      loc = float(self.get_argument('fitloc'))
      xres = float(self.get_argument('xres'))
      loc_fixed = bool(int(self.get_argument('locfixed')))
      bands, ints = spectrum.T
      peak_mask, peak_y, peak_data = fit_single_peak(
          bands, ints, loc, fit_kind=kind, log_fn=logging.info,
          band_resolution=xres, loc_fixed=loc_fixed)
      peak_x = bands[peak_mask]

      # show the fitted peak
      ax.plot(peak_x, peak_y, 'k-', linewidth=2, alpha=0.75)

    # Finish plotting
    if len(peak_x) > 0:
      xpad = (peak_x[-1] - peak_x[0]) / 3.
      ax.set_xlim((max(xlim[0], peak_x[0] - xpad),
                   min(xlim[1], peak_x[-1] + xpad)))
    else:
      ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig_data.manager.canvas.draw()

    # write the peak data as a response
    return self.write(json_encode(peak_data))


def _manual_peak_area(spectrum, bounds, base_type='region'):
  s, t = np.searchsorted(spectrum[:,0], bounds)
  x, y = spectrum[s:t].T
  if len(x) == 0:
    return x, y, 0, dict()
  if t <= s:
    base, area = 0, 0
  else:
    if base_type == 'region':
      base = y.min()
    else:
      base = min(y[0], y[-1])
    area = scipy.integrate.simps(y - base, x)
  peak_data = dict(xmin=float(x[0]), xmax=float(x[-1]),
                   height=float(y.max()-base), area=float(area))
  return x, y, base, peak_data


routes = [
    (r'/_peak', PeakHandler),
    (r'/([0-9]+)/peak_area\.csv', PeakHandler),
]
