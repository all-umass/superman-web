from __future__ import absolute_import
import ast
import logging
import numpy as np
import os
from collections import defaultdict
from superman.preprocess import preprocess
from superman.dataset import LookupMetadata
from tornado import gen
from threading import Thread

from .base import BaseHandler, MultiDatasetHandler


class SearchHandler(MultiDatasetHandler):
  @gen.coroutine
  def post(self):
    fig_data = self.get_fig_data()
    if fig_data is None:
      self.visible_error(403, "Broken connection to server.")
      return
    # get the most up to date spectrum
    try:
      query = fig_data.get_trajectory(nan_gap=None)
    except ValueError:
      logging.error('No query spectrum to search with.')
      return

    num_results = int(self.get_argument('num_results', 10))

    metric = self.get_argument('metric')
    metric += ':' + self.get_argument('param')
    min_window = int(self.get_argument('min_window'))
    num_comps = int(self.get_argument('num_comps'))
    score_pct = int(self.get_argument('score_pct'))
    kwargs = dict(num_endmembers=num_comps, num_results=num_results,
                  metric=metric, min_window=min_window, score_pct=score_pct)
    logging.info('Search params: %r', kwargs)

    if query[0,0] > query[1,0]:
      query = np.flipud(query)

    if abs(1 - query[:,1].max()) > 0.001:
      # WSM needs max-normalization, so we force it.
      logging.warning('Applying max-normalization to query before search')
      query[:,1] = preprocess(query[:,1:2].T, 'normalize:max').ravel()

    # prepare the target library
    extra_kwargs = dict(nan_gap=None)
    if self.get_argument('pp', None) is None:
      logging.warning('Applying max-normalization to library before search')
      extra_kwargs['pp'] = 'normalize:max'
    all_ds_views, _ = self.prepare_ds_views(fig_data, **extra_kwargs)
    if all_ds_views is None:
      self.visible_error(404, 'Failed to look up dataset(s).')
      return

    # TODO: enable searching over multiple datasets
    if len(all_ds_views) != 1:
      self.visible_error(403, 'Can only search over one dataset.')
      return
    ds_view, = all_ds_views
    ds = ds_view.ds

    # search!
    try:
      top_names, top_sim = yield gen.Task(_async_search, ds_view, query, kwargs)
    except ValueError as e:
      logging.error('During whole_spectrum_search: %s', str(e))
      return

    # get query info
    query_name, = fig_data._ds_view.get_primary_keys()
    query_meta = {label: data[0] for data, label
                  in _lookup_metas(fig_data._ds_view)}

    # get any LookupMetadata associated with the matches
    all_names = [n for names in top_names for n in names]
    if ds.pkey is None:
      ds_view.mask = np.array([int(n.rsplit(' ', 1)[1]) for n in all_names],
                              dtype=int)
    else:
      ds_view.mask, = np.nonzero(ds.filter_metadata(dict(pkey=all_names)))
    all_names = ds_view.get_primary_keys()
    top_meta = defaultdict(dict)
    for data, label in _lookup_metas(ds_view):
      for name, x in zip(all_names, data):
        top_meta[name][label] = x

    header = ['Rank', 'Score', 'Match', 'Plot']
    if len(top_names[0]) > 1:
      header[2] = 'Matches'
    rows = zip(top_sim, top_names)
    self.render('_search_results.html', header=header, rows=rows,
                ds_name=ds.name, ds_kind=ds.kind, top_meta=top_meta,
                query_name=query_name, query_meta=query_meta)


def _async_search(ds_view, query, wsm_kwargs, callback=None):
  t = Thread(target=lambda: callback(
      ds_view.whole_spectrum_search(query, **wsm_kwargs)))
  t.daemon = True
  t.start()


def _lookup_metas(ds_view):
  for key, m in ds_view.ds.metadata.items():
    if isinstance(m, LookupMetadata):
      yield ds_view.get_metadata(key)


class CompareHandler(BaseHandler):
  def get(self, fignum):
    '''Downloads plot data as text.'''
    fig_data = self.get_fig_data(int(fignum))
    if fig_data is None:
      self.write('Oops, something went wrong. Try again?')
      return

    ax = fig_data.figure.gca()
    lines = [l.get_xydata() for l in ax.lines]
    if ax.legend_ is not None:
      names = [t.get_text() for t in ax.legend_.get_texts()]
    else:
      names = [ax.get_title()]

    fname = os.path.basename(self.request.path)
    self.set_header('Content-Type', 'text/plain')
    self.set_header('Content-Disposition',
                    'attachment; filename='+fname)
    self.write('Spectrum,Axis\n')
    for name, traj in zip(names, lines):
      self.write('%s,x,' % name)
      self.write(','.join('%g' % x for x in traj[:,0]))
      self.write('\n,y,')
      self.write(','.join('%g' % y for y in traj[:,1]))
      self.write('\n')
    self.finish()

  def post(self):
    fig_data = self.get_fig_data()
    ds = self.request_one_ds('target_kind', 'target_name')
    if fig_data is None or ds is None:
      return

    # emulate the library preparation process
    trans = self.ds_view_kwargs(nan_gap=None)
    if not trans['pp']:
      trans['pp'] = 'normalize:max'

    # select only the comparison samples
    names = ast.literal_eval(self.get_argument('compare'))
    # TODO: make this work if ds.pkey is None
    mask = ds.filter_metadata(dict(pkey=names))
    ds_view = ds.view(mask=mask, **trans)

    fig_data.figure.clf(keep_observers=True)
    fig_data.plot()
    ax = fig_data.figure.gca()
    for comp in ds_view.get_trajectories():
      ax.plot(comp[:,0], comp[:,1], '-')
    ax.legend([fig_data.title] + names)
    fig_data.manager.canvas.draw()


routes = [
    (r'/_search', SearchHandler),
    (r'/_compare', CompareHandler),
    (r'/([0-9]+)/search_results\.csv', CompareHandler)
]
