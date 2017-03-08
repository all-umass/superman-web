from __future__ import absolute_import
import itertools

from .baseline import routes as baseline_routes
from .classifications import routes as classify_routes
from .compositions import routes as composition_routes
from .datasets import routes as dataset_routes
from .filterplots import routes as filterplot_routes
from .generic import routes as generic_routes
from .matching import routes as matching_routes
from .generic_models import routes as model_routes
from .peakfit import routes as peak_routes
from .predictions import routes as predict_routes
from .subpages import routes as page_routes
from .upload import routes as upload_routes

all_routes = list(itertools.chain(
    page_routes, dataset_routes, baseline_routes, matching_routes,
    generic_routes, filterplot_routes, peak_routes, composition_routes,
    predict_routes, upload_routes, model_routes, classify_routes))
