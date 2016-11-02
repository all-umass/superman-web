from __future__ import absolute_import
import ast
import logging
import numpy as np
import tornado.web
from superman.baseline import BL_CLASSES
from superman.baseline.common import Baseline

from ..web_datasets import DATASETS


class BaseHandler(tornado.web.RequestHandler):
  is_private = True

  def get_fig_data(self, fignum=None):
    if fignum is None:
      fignum = int(self.get_argument('fignum', 0))
    if fignum not in self.application.figure_data:
      logging.error('Invalid figure number: %d', fignum)
      return None
    return self.application.figure_data[fignum]

  def get_current_user(self):
    # Necessary for authentication,
    # see http://tornado.readthedocs.org/en/latest/guide/security.html
    if self.is_private:
      return self.get_secure_cookie('user')
    return True

  def _include_private_datasets(self):
    return self.is_private and self.get_current_user()

  def all_datasets(self):
    include_private = self._include_private_datasets()
    return [d for dd in DATASETS.values() for d in dd.values()
            if d.is_public or include_private]

  def get_dataset(self, ds_kind, ds_name):
    ds = DATASETS[ds_kind].get(ds_name, None)
    if ds is None or ds.is_public or self._include_private_datasets():
      return ds
    return None

  def dataset_kinds(self):
    return DATASETS.keys()

  def request_one_ds(self, kind_arg='ds_kind', name_arg='ds_name'):
    return self.get_dataset(self.get_argument(kind_arg),
                            self.get_argument(name_arg))

  def request_many_ds(self, kind_arg='ds_kind[]', name_arg='ds_name[]'):
    return filter(None, [self.get_dataset(k, n) for k, n in
                         zip(self.get_arguments(kind_arg),
                             self.get_arguments(name_arg))])

  def ds_view_kwargs(self, return_blr_params=False, **extra_kwargs):
    method = self.get_argument('blr_method', '').lower()
    segmented_str = self.get_argument('blr_segmented', 'false')
    inverted_str = self.get_argument('blr_inverted', 'false')

    if method and method not in BL_CLASSES:
      raise ValueError('Invalid blr method: %r' % method)
    if segmented_str not in ('true', 'false'):
      raise ValueError('Invalid blr segmented flag: %r' % segmented_str)
    if inverted_str not in ('true', 'false'):
      raise ValueError('Invalid blr inverted flag: %r' % inverted_str)

    segmented = segmented_str == 'true'
    inverted = inverted_str == 'true'
    lb = float(self.get_argument('blr_lb', '') or '-inf')
    ub = float(self.get_argument('blr_ub', '') or 'inf')
    step = float(self.get_argument('blr_step', '') or 0)

    # initialize the baseline correction object
    bl_obj = BL_CLASSES[method]() if method else NullBaseline()
    params = {}
    for key in bl_obj.param_ranges():
      param = ast.literal_eval(self.get_argument('blr_' + key, 'None'))
      if param is not None:
        params[key] = param
        setattr(bl_obj, key, param)

    trans = dict(
        chan_mask=bool(int(self.get_argument('chan_mask', 0))),
        pp=self.get_argument('pp', ''), blr_obj=bl_obj, blr_inverted=inverted,
        blr_segmented=segmented, crop=(lb, ub, step), **extra_kwargs)

    if return_blr_params:
      return trans, params
    return trans


class MultiDatasetHandler(BaseHandler):
  def prepare_ds_views(self, fig_data, max_num_spectra=99999):
    all_ds = self.request_many_ds()

    if not all_ds:
      logging.warning('No dataset(s) found')
      return None, 0

    num_spectra = sum(np.count_nonzero(fig_data.filter_mask[ds])
                      for ds in all_ds)

    if not 0 < num_spectra <= max_num_spectra:
      logging.warning('Too many spectra chosen: %d > %d', num_spectra,
                      max_num_spectra)
      return None, num_spectra

    # set up the dataset view object
    trans, blr_params = self.ds_view_kwargs(return_blr_params=True)
    all_ds_views = [ds.view(mask=fig_data.filter_mask[ds], **trans)
                    for ds in all_ds]

    # check to see if anything changed since the last view we had
    view_keys = ['chan_mask', 'pp', 'blr_method', 'blr_segmented',
                 'blr_inverted', 'blr_lb', 'blr_ub', 'blr_step']
    for k in sorted(blr_params):
      view_keys.append('blr_' + k)
    view_params = tuple(self.get_argument(k, '') for k in view_keys)
    # if the view changed, invalidate the cache
    if view_params != fig_data.explorer_view_params:
      fig_data.clear_explorer_cache()
      fig_data.explorer_view_params = view_params

    return all_ds_views, num_spectra


# A do-nothing baseline, for consistency
class NullBaseline(Baseline):
  def _fit_many(self, bands, intensities):
    return 0

  def fit_transform(self, bands, intensities, segment=False, invert=False):
    return intensities

  def param_ranges(self):
    return {}
