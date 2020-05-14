from __future__ import absolute_import, print_function, division
import ast
import json
import logging
import numpy as np
import tornado.web
from superman.baseline import BL_CLASSES
from superman.baseline.common import Baseline
from superman.dataset import MultiDatasetView
from six.moves import zip_longest
from six.moves import zip

from ..web_datasets import DATASETS

__all__ = ['BLR_KWARGS', 'BaseHandler', 'MultiDatasetHandler']


def _make_blr_kwargs():
    def compute_step(lb, ub, kind):
        if kind == 'integer':
            return 1
        if kind == 'log':
            lb, ub = np.log10((lb, ub))
        return (ub - lb) / 100.
    return dict(
        bl_classes=sorted((key, bl()) for key, bl in BL_CLASSES.items()),
        compute_step=compute_step, log10=np.log10)


# make a singleton for use in various page handlers
BLR_KWARGS = _make_blr_kwargs()


# See https://stackoverflow.com/a/57915246/10601
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


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
        return list(DATASETS.keys())

    def request_one_ds(self, kind_arg='ds_kind', name_arg='ds_name'):
        return self.get_dataset(self.get_argument(kind_arg),
                                self.get_argument(name_arg))

    def request_many_ds(self, kind_arg='ds_kind[]', name_arg='ds_name[]'):
        many_ds = [self.get_dataset(k, n) for k, n in
                   zip(self.get_arguments(kind_arg),
                       self.get_arguments(name_arg))]
        return [ds for ds in many_ds if ds]

    def visible_error(self, status, msg, *log_args):
        if not log_args:
            logging.error(msg)
        else:
            logging.error(*log_args)
        self.set_status(status)
        self.finish("Error: " + msg)

    def ds_view_kwargs(self, **extra_kwargs):
        method = self.get_argument('blr_method', '').lower()
        segmented_str = self.get_argument('blr_segmented', 'false')
        inverted_str = self.get_argument('blr_inverted', 'false')
        flip_str = self.get_argument('blr_flip', 'false')

        if method and method not in BL_CLASSES:
            raise ValueError('Invalid blr method: %r' % method)
        if segmented_str not in ('true', 'false'):
            raise ValueError('Invalid blr segmented flag: %r' % segmented_str)
        if inverted_str not in ('true', 'false'):
            raise ValueError('Invalid blr inverted flag: %r' % inverted_str)
        if inverted_str not in ('true', 'false'):
            raise ValueError('Invalid blr flip flag: %r' % flip_str)

        segmented = segmented_str == 'true'
        inverted = inverted_str == 'true'
        flip = flip_str == 'true'

        # collect (lb,ub,step) tuples, so long as they're not all blank
        crops = [(float(lb or '-inf'), float(ub or 'inf'), float(step or 0))
                 for lb, ub, step in zip_longest(self.get_arguments('crop_lb[]'),
                                                 self.get_arguments(
                                                     'crop_ub[]'),
                                                 self.get_arguments('crop_step[]'))
                 if lb or ub or step]

        # initialize the baseline correction object
        bl_obj = BL_CLASSES[method]() if method else _NullBaseline()
        params = {}
        for key in bl_obj.param_ranges():
            param = ast.literal_eval(self.get_argument('blr_' + key, 'None'))
            if param is not None:
                params[key] = param
                setattr(bl_obj, key, param)

        return dict(
            chan_mask=bool(int(self.get_argument('chan_mask', 0))),
            pp=self.get_argument('pp', ''), blr_obj=bl_obj, blr_inverted=inverted,
            blr_segmented=segmented, flip=flip, crop=tuple(crops), **extra_kwargs)

    def write_json(self, obj):
        """Essentially self.write(json.dumps(obj)),
        but with some fixes to avoid invalid JSON (NaN, Infinity, etc).
        """
        s = json.dumps(obj, cls=NumpyEncoder).replace('</', '<\\/')
        s = s.replace('NaN', 'null').replace('Infinity', 'null')
        self.write(s)


class MultiDatasetHandler(BaseHandler):
    def prepare_ds_views(self, fig_data, max_num_spectra=99999,
                         **extra_view_kwargs):
        all_ds = self.request_many_ds()
        trans = self.ds_view_kwargs(**extra_view_kwargs)
        all_ds_views = [ds.view(mask=fig_data.filter_mask[ds], **trans)
                        for ds in all_ds]
        mdv = MultiDatasetView(all_ds_views)

        if 0 < mdv.num_spectra() <= max_num_spectra:
            return mdv
        else:
            logging.warning('Bad number of spectra chosen: %d',
                            mdv.num_spectra())
            return None


# A do-nothing baseline, for consistency
class _NullBaseline(Baseline):
    def _fit_many(self, bands, intensities):
        return 0

    def fit_transform(self, bands, intensities, segment=False, invert=False):
        return intensities

    def param_ranges(self):
        return {}
