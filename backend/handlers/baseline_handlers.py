from __future__ import absolute_import
import logging
import numpy as np
import os

from .base import BaseHandler


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

    trans = self.ds_view_kwargs()
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
