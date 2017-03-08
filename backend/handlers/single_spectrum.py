from __future__ import absolute_import
import logging
import numpy as np

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

    ds_name = self.get_argument('ds_name')
    ds_kind = self.get_argument('ds_kind')
    ds = self.get_dataset(ds_kind, ds_name)
    if ds is None:
      msg = "Can't find dataset: %s [%s]" % (ds_name, ds_kind)
      return self.visible_error(404, msg)

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


class ZoomFigureHandler(BaseHandler):
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      return self.visible_error(403, 'Broken connection to server.')

    xmin = float(self.get_argument('xmin'))
    xmax = float(self.get_argument('xmax'))
    ymin = float(self.get_argument('ymin'))
    ymax = float(self.get_argument('ymax'))
    ax = fig_data.figure.axes[0]
    ax.set_xlim((xmin,xmax))
    ax.set_ylim((ymin,ymax))
    fig_data.manager.canvas.draw()


routes = [
    (r'/_dataset_selector', SelectorHandler),
    (r'/_select', SelectHandler),
    (r'/_pp', PreprocessHandler),
    (r'/_zoom', ZoomFigureHandler),
]
