from __future__ import absolute_import
import ast
import logging
import numpy as np
import os
from superman.baseline import BL_CLASSES
from superman.baseline.common import Baseline

from .base import BaseHandler


def ds_view_kwargs(request, return_blr_params=False, **extra_kwargs):
  method = request.get_argument('blr_method', '').lower()
  segmented_str = request.get_argument('blr_segmented', 'false')
  inverted_str = request.get_argument('blr_inverted', 'false')

  if method and method not in BL_CLASSES:
    raise ValueError('Invalid blr method: %r' % method)
  if segmented_str not in ('true', 'false'):
    raise ValueError('Invalid blr segmented flag: %r' % segmented_str)
  if inverted_str not in ('true', 'false'):
    raise ValueError('Invalid blr inverted flag: %r' % inverted_str)

  segmented = segmented_str == 'true'
  inverted = inverted_str == 'true'
  lb = float(request.get_argument('blr_lb', '') or '-inf')
  ub = float(request.get_argument('blr_ub', '') or 'inf')
  step = float(request.get_argument('blr_step', '') or 0)

  # initialize the baseline correction object
  bl_obj = BL_CLASSES[method]() if method else NullBaseline()
  params = {}
  for key in bl_obj.param_ranges():
    param = ast.literal_eval(request.get_argument('blr_' + key, 'None'))
    if param is not None:
      params[key] = param
      setattr(bl_obj, key, param)

  trans = dict(
      chan_mask=bool(int(request.get_argument('chan_mask', 0))),
      pp=request.get_argument('pp', ''), blr_obj=bl_obj, blr_inverted=inverted,
      blr_segmented=segmented, crop=(lb, ub, step), **extra_kwargs)

  if return_blr_params:
    return trans, params
  return trans


class BaselineHandler(BaseHandler):
  def get(self, fignum):
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      return self.write('Oops, something went wrong. Try again?')
    spectrum = fig_data.get_trajectory('upload')
    bl = fig_data.baseline
    if bl is None:
      bl = np.zeros(spectrum.shape[0])
    fname = 'baseline.' + os.path.splitext(fig_data.title)[0] + '.txt'
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition', 'attachment; filename='+fname)
    for (x,y),b in zip(spectrum, bl):
      self.write('%g\t%g\t%g\t%g\n' % (x, y, b, y-b))
    self.finish()

  def post(self):
    # Check arguments first to fail fast.
    fig_data = self.get_fig_data()
    if fig_data is None:
      return

    trans = ds_view_kwargs(self)
    del trans['pp']
    del trans['chan_mask']

    fig_data.add_transform('baseline-corrected', **trans)
    logging.info('Running BLR: %r', trans)

    if len(fig_data.figure.axes) == 2:
      # comparison view for the baseline page
      ax1, ax2 = fig_data.figure.axes
      fig_data.plot('upload', ax=ax1)
      bands, corrected = fig_data.get_trajectory('baseline-corrected').T
      baseline = trans['blr_obj'].baseline.ravel()
      fig_data.baseline = baseline
      ax1.plot(bands, baseline, 'r-')
      ax2.plot(bands, corrected, 'k-')
      ax2.set_title('Corrected')
      fig_data.manager.canvas.draw()
    else:
      # regular old plot of the corrected spectrum
      fig_data.plot('baseline-corrected')

# Define the routes for each handler.
routes = [
    (r'/_baseline', BaselineHandler),
    (r'/([0-9]+)/baseline\.txt', BaselineHandler),
]


# A do-nothing baseline, for consistency
class NullBaseline(Baseline):
  def _fit_many(self, bands, intensities):
    return 0

  def fit_transform(self, bands, intensities, segment=False, invert=False):
    return intensities

  def param_ranges(self):
    return {}
