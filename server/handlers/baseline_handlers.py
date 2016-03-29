from __future__ import absolute_import
import ast
import numpy as np
import os
from superman.baseline import BL_CLASSES

from .base import BaseHandler


def setup_blr_object(request):
  method = request.get_argument('blr_method', '').lower()
  if not method:
    # XXX: lame hack
    return None, False, -np.inf, np.inf, None
  if method not in BL_CLASSES:
    raise ValueError('Invalid blr method:', method)

  segmented = request.get_argument('blr_segmented', 'false')
  if segmented not in ('true', 'false'):
    raise ValueError('Invalid blr segmented flag:', segmented)
  do_segmented = segmented == 'true'

  lb = request.get_argument('blr_lb', '')
  lb = float(lb) if lb else -np.inf
  ub = request.get_argument('blr_ub', '')
  ub = float(ub) if ub else np.inf

  # initialize the baseline correction object
  bl_obj = BL_CLASSES[method]()
  params = {}
  for key in bl_obj.param_ranges():
    param = ast.literal_eval(request.get_argument('blr_' + key, 'None'))
    if param is not None:
      params[key] = param
      setattr(bl_obj, key, param)

  return bl_obj, do_segmented, lb, ub, params


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

    bl_obj, do_segmented, lb, ub, _ = setup_blr_object(self)
    if bl_obj is None:
      return

    fig_data.add_transform('baseline-corrected', blr_obj=bl_obj,
                           blr_segmented=do_segmented, crop=(lb, ub))

    if len(fig_data.figure.axes) == 2:
      # comparison view for the baseline page
      ax1, ax2 = fig_data.figure.axes
      fig_data.plot('upload', ax=ax1)
      bands, corrected = fig_data.get_trajectory('baseline-corrected').T
      baseline = bl_obj.baseline.ravel()
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
