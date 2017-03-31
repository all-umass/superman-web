from __future__ import absolute_import
import logging

from .common import BaseHandler


class SearchMetadataHandler(BaseHandler):
  def post(self):
    ds_kinds = self.get_arguments('ds_kind[]')
    case_sensitive = bool(int(self.get_argument('case_sensitive')))
    full_text = bool(int(self.get_argument('full_text')))
    query_str = self.get_argument('query')
    logging.info('Searching: query=%r, case=%s, fulltext=%s, kinds=%s',
                 query_str, case_sensitive, full_text, ds_kinds)

    if not query_str:
      return self.visible_error(403, 'Must supply a query.')

    datasets = self.all_datasets()
    if ds_kinds:
      datasets = [ds for ds in datasets if ds.kind in ds_kinds]

    results = []
    for ds in datasets:
      res = ds.search_metadata(query_str, full_text=full_text,
                               case_sensitive=case_sensitive)
      if res:
        results.append((ds, res))

    results.sort(key=lambda t: (t[0].kind, t[0].name))
    self.render('_search_results.html', results=results)


routes = [
    (r'/_search_metadata', SearchMetadataHandler),
]
