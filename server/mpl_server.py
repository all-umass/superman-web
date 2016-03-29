import matplotlib
matplotlib.use('WebAgg')
matplotlib.rc('axes', facecolor='none')
import io
import json
import socket
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
from matplotlib.figure import Figure

try:
  from matplotlib.backends.backend_webagg_core import (
      FigureManagerWebAgg, new_figure_manager_given_figure)
except ImportError:
  # Older matplotlibs have this in backend_webagg
  from matplotlib.backends.backend_webagg import (
      FigureManagerWebAgg, new_figure_manager_given_figure)

# positions for various stages of processing
_key2idx = {'upload': 0, 'baseline-corrected': 1, 'pp': 2}


class FigData(object):
  def __init__(self, fig, manager):
    self.figure = fig
    self.manager = manager
    self.title = ''
    self.filter_mask = Ellipsis
    self.baseline = None
    self.explorer_data = []
    self.hist_data = []
    self._ds_view = None
    self._transformations = [None, None, None]

  def set_selected(self, ds_view, title=''):
    self.title = title
    self._ds_view = ds_view
    self._transformations = [ds_view.transformations, None, None]

  def add_transform(self, key, **transformations):
    idx = _key2idx[key]
    self._transformations[idx] = transformations
    for i in range(idx + 1, len(self._transformations)):
      self._transformations[i] = None

  def get_trans(self, key='pp'):
    idx = _key2idx[key]
    trans = {}
    for t in filter(None, self._transformations[:idx+1]):
      trans.update(t)
    return trans

  def get_trajectory(self, key='pp'):
    self._ds_view.transformations = self.get_trans(key=key)
    traj, = self._ds_view.get_trajectories()
    return traj

  def plot(self, key='pp', ax=None):
    bands, ints = self.get_trajectory(key=key).T

    for _ax in self.figure.axes:
      _ax.cla()
    if ax is None:
      if self.figure.axes:
        ax = self.figure.axes[0]
      else:
        ax = self.figure.gca()

    ax.plot(bands, ints, '-')
    ax.set_title(self.title)
    self.manager.canvas.draw()
    # return the axis limits
    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    return [xmin, xmax, ymin, ymax]


class MatplotlibServer(tornado.web.Application):
  class MplJs(tornado.web.RequestHandler):
    """
    Serves the generated matplotlib javascript file.  The content
    is dynamically generated based on which toolbar functions the
    user has defined.
    """
    def get(self):
      self.set_header('Content-Type', 'application/javascript')
      self.write(FigureManagerWebAgg.get_javascript())

  class Download(tornado.web.RequestHandler):
    """
    Handles downloading of the figure in various file formats.
    """
    def get(self, fignum, fmt):
      fignum = int(fignum)
      manager = self.application.figure_data[fignum].manager
      mimetypes = {
          'ps': 'application/postscript',
          'eps': 'application/postscript',
          'pdf': 'application/pdf',
          'svg': 'image/svg+xml',
          'png': 'image/png',
          'jpeg': 'image/jpeg',
          'tif': 'image/tiff',
          'emf': 'application/emf'
      }
      self.set_header('Content-Type', mimetypes.get(fmt, 'binary'))
      # write the file
      buff = io.BytesIO()
      manager.canvas.print_figure(buff, format=fmt)
      self.write(buff.getvalue())

  class WebSocket(tornado.websocket.WebSocketHandler):
    """
    A websocket for interactive communication between the plot in
    the browser and the server.

    In addition to the methods required by tornado, it is required to
    have two callback methods:

        - ``send_json(json_content)`` is called by matplotlib when
          it needs to send json to the browser.  `json_content` is
          a JSON tree (Python dictionary), and it is the responsibility
          of this implementation to encode it as a string to send over
          the socket.

        - ``send_binary(blob)`` is called to send binary image data
          to the browser.
    """
    supports_binary = True

    def open(self, fignum):
      self.fignum = int(fignum)
      # Register the websocket with the FigureManager.
      manager = self.application.figure_data[self.fignum].manager
      manager.add_web_socket(self)
      if hasattr(self, 'set_nodelay'):
        self.set_nodelay(True)

    def on_close(self):
      # When the socket is closed, deregister the websocket with
      # the FigureManager.
      manager = self.application.figure_data.pop(self.fignum).manager
      manager.remove_web_socket(self)

    def on_message(self, message):
      # Every message has a "type" and a "figure_id".
      message = json.loads(message)
      # if not message['type'] == 'motion_notify':
      #     print 'got message', message
      if message['type'] == 'supports_binary':
        self.supports_binary = message['value']
      else:
        manager = self.application.figure_data[self.fignum].manager
        manager.handle_json(message)

    def send_json(self, content):
      self.write_message(json.dumps(content))

    def send_binary(self, blob):
      if self.supports_binary:
        self.write_message(blob, binary=True)
      else:
        data_uri = "data:image/png;base64,{0}".format(
            blob.encode('base64').replace('\n', ''))
        self.write_message(data_uri)

  def __init__(self, *handlers, **kwargs):
    handlers = [
        (r'/_static/(.*)',
         tornado.web.StaticFileHandler,
         {'path': FigureManagerWebAgg.get_static_file_path()}),
        ('/mpl.js', self.MplJs),
        (r'/([0-9]+)/ws', self.WebSocket),
        (r'/([0-9]+)/download.([a-z0-9.]+)', self.Download),
    ] + list(handlers)
    super(MatplotlibServer, self).__init__(handlers, **kwargs)
    self.figure_data = {}  # id -> FigData

  def run_forever(self, port):
    self.listen(port)
    print('Running at http://%s:%d/' % (socket.gethostname(), port))
    print('Press Ctrl+C to quit')
    tornado.ioloop.IOLoop.instance().start()

  def register_new_figure(self, size):
    fig = Figure(figsize=size, facecolor='none', edgecolor='none',
                 frameon=False, tight_layout=True)
    fignum = id(fig)
    manager = new_figure_manager_given_figure(fignum, fig)
    self.figure_data[fignum] = FigData(fig, manager)
    return fignum
