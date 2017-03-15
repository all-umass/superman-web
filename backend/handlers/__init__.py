from __future__ import absolute_import
import itertools

from .classifications import routes as classify_routes
from .compositions import routes as composition_routes
from .datasets import routes as dataset_routes
from .explorer import routes as explorer_routes
from .figure import routes as figure_routes
from .filterplots import routes as filterplot_routes
from .generic_models import routes as model_routes
from .matching import routes as matching_routes
from .peakfit import routes as peak_routes
from .predictions import routes as predict_routes
from .search import routes as search_routes
from .single_spectrum import routes as spectrum_routes
from .subpages import routes as page_routes
from .upload import routes as upload_routes

all_routes = list(itertools.chain(
    page_routes, dataset_routes, matching_routes, explorer_routes,
    figure_routes, spectrum_routes, filterplot_routes, peak_routes,
    composition_routes, predict_routes, upload_routes, model_routes,
    classify_routes, search_routes
))
