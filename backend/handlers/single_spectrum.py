from __future__ import absolute_import
import logging
import numpy as np
import os

from .common import BaseHandler


class SelectorHandler(BaseHandler):
  def post(self):
    ds = self.request_one_ds('kind', 'name')
    if ds is None:
      return self.visible_error(404, 'Dataset not found.')
    logging.info('Generating selector for dataset: %s', ds)
    return self.render('_spectrum_selector.html', ds=ds)


class SelectHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    ds = self.request_one_ds()
    if ds is None:
      return self.visible_error(404, 'Dataset not found.')

    name = self.get_argument('name', None)
    if name is None:
      idx = int(self.get_argument('idx'))
      if not (0 <= idx < ds.num_spectra()):
        return self.visible_error(403, 'Invalid spectrum number.',
                                  'Index %d out of bounds in dataset %s',
                                  idx, ds)
      name = 'Spectrum %d' % idx
    else:
      # XXX: hack to match dtype of pkey
      name = np.array(name, dtype=ds.pkey.keys.dtype).item()
      idx = ds.pkey.key2index(name)

    fig_data.set_selected(ds.view(mask=[idx]), title=str(name))
    axlimits = fig_data.plot()
    return self.write_json(axlimits)


class PreprocessHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    pp = self.get_argument('pp')
    fig_data.add_transform('pp', pp=pp)
    axlimits = fig_data.plot('pp')
    return self.write_json(axlimits)


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
    try:
      bands, corrected = fig_data.get_trajectory('baseline-corrected').T
    except Exception:
      logging.exception('BLR failed.')
      return self.visible_error(400, 'Baseline correction failed.')

    if len(fig_data.figure.axes) == 2:
      # comparison view for the baseline page
      ax1, ax2 = fig_data.figure.axes
      fig_data.plot('upload', ax=ax1)
      baseline = trans['blr_obj'].baseline.ravel()
      fig_data.baseline = baseline
      ax1.plot(bands, baseline, 'r-')
      ax2.plot(bands, corrected, 'k-')
      ax2.set_title('Corrected')
    else:
      # regular old plot of the corrected spectrum
      ax = fig_data.figure.gca()
      ax.clear()
      ax.plot(bands, corrected, '-')
      ax.set_title(fig_data.title)
    fig_data.manager.canvas.draw()


routes = [
    (r'/_dataset_selector', SelectorHandler),
    (r'/_select', SelectHandler),
    (r'/_pp', PreprocessHandler),
    (r'/_baseline', BaselineHandler),
    (r'/([0-9]+)/baseline\.txt', BaselineHandler),
]
