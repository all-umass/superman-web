from __future__ import absolute_import
from .common import BaseHandler


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
    (r'/_zoom', ZoomFigureHandler),
]
