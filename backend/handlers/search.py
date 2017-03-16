from __future__ import absolute_import
import logging
import shlex

from .common import BaseHandler
from ..web_datasets import DATASETS, LookupMetadata, TagMetadata


class SearchMetadataHandler(BaseHandler):
  def post(self):
    ds_kinds = self.get_arguments('ds_kind[]')
    case_sensitive = bool(int(self.get_argument('case_sensitive')))
    include_pkey = bool(int(self.get_argument('include_pkey')))
    name_query = self.get_argument('name_query')
    text_query = self.get_argument('text_query')
    logging.info('Searching: name=%r, text=%r, case=%s, pkey=%s, kinds=%s',
                 name_query, text_query, case_sensitive, include_pkey, ds_kinds)

    if not (name_query or text_query):
      return self.visible_error(403, 'Must supply at least one query.')
    name_query = _parse_query(name_query, case_sensitive)
    text_query = _parse_query(text_query, case_sensitive)

    if not ds_kinds:
      ds_kinds = DATASETS.keys()

    results = []
    for kind in ds_kinds:
      for ds in DATASETS[kind].values():
        res = _search_dataset(ds, name_query, text_query, include_pkey)
        if res:
          results.append((ds, res))

    results.sort(key=lambda t: (t[0].kind, t[0].name))
    self.render('_search_results.html', results=results)


def _search_dataset(ds, name_query, text_query, include_pkey):
  ds_results = {}

  if include_pkey and ds.pkey is not None and text_query is not None:
    if any(text_query(k) for k in ds.pkey.keys):
      ds_results['pkey'] = True

  name_res, text_res = [], []
  for key, m in ds.metadata.items():
    name = m.display_name(key)

    if name_query is not None and name_query(name):
      name_res.append(name)
    if text_query is not None and _search_text(m, text_query):
      text_res.append(name)

  if name_res:
    ds_results['meta_name'] = name_res
  if text_res:
    ds_results['meta_text'] = text_res

  return ds_results


def _search_text(meta, query):
  if isinstance(meta, LookupMetadata):
    return any(query(x) for x in meta.uniques)
  if isinstance(meta, TagMetadata):
    return any(query(x) for x in meta.tags)
  return False


def _parse_query(query, case_sensitive=False):
  tokens = shlex.split(query)
  if not tokens:
    return None

  # AND expressions bind tightest, then OR, then adjacency (implicit AND)
  for lexeme, expr_type in [(u'AND', AndExpr), (u'OR', OrExpr)]:
    while True:
      idxs = [i for i, x in enumerate(tokens)
              if not isinstance(x, QueryExpr) and x.upper() == lexeme]
      if not idxs:
        break
      idx = idxs[0]
      lhs = tokens[idx-1]
      rhs = tokens[idx+1]
      tokens = tokens[:idx-1] + [expr_type(lhs, rhs)] + tokens[idx+2:]

  # AND together any remaining roots
  while len(tokens) > 1:
    rhs = tokens.pop()
    lhs = tokens.pop()
    tokens.append(AndExpr(lhs, rhs))

  # get a matcher function from the root
  root, = tokens
  logging.info('Parsed query: %r', root)
  return _matcher(root, case_sensitive)


def _matcher(token, case_sensitive):
  if isinstance(token, QueryExpr):
    return token.matcher(case=case_sensitive)
  elif case_sensitive:
    return lambda text: token in text
  else:
    low = token.lower()
    return lambda text: low in text.lower()


class QueryExpr(object):
  def __init__(self, lhs, rhs):
    self.lhs = lhs
    self.rhs = rhs


class AndExpr(QueryExpr):
  def matcher(self, case=False):
    lhs = _matcher(self.lhs, case)
    rhs = _matcher(self.rhs, case)
    return lambda text: lhs(text) and rhs(text)

  def __repr__(self):
    return u'(AND: %r %r)' % (self.lhs, self.rhs)


class OrExpr(QueryExpr):
  def matcher(self, case=False):
    lhs = _matcher(self.lhs, case)
    rhs = _matcher(self.rhs, case)
    return lambda text: lhs(text) or rhs(text)

  def __repr__(self):
    return u'(OR: %r %r)' % (self.lhs, self.rhs)


routes = [
    (r'/_search_metadata', SearchMetadataHandler),
]
