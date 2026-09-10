"""Short-term photovoltaic generation forecasting on prosumer-scale inverter data."""

from .baselines import BASELINE_NAMES, get_baseline
from .config import SEED, SITES, Site, set_seed
from .dataset import common_window, load_hourly, load_site, months, restrict
from .evaluation import Evaluation, comparison, holdout, metrics, walk_forward
from .features import (
    BASE_FEATURES,
    CORE_FEATURES,
    FEATURE_GROUPS,
    FEATURE_SETS,
    FULL_FEATURES,
    TARGET,
    add_recency_weights,
    build_features,
    design_matrix,
)
from .forecast import Forecaster, bias_factor, climatology, daily_energy, horizon_index
from .models import DEFAULT_MODELS, MODEL_NAMES, get_model

__version__ = "1.0.0"

__all__ = [
    "BASELINE_NAMES",
    "BASE_FEATURES",
    "CORE_FEATURES",
    "DEFAULT_MODELS",
    "Evaluation",
    "FEATURE_GROUPS",
    "FEATURE_SETS",
    "FULL_FEATURES",
    "Forecaster",
    "MODEL_NAMES",
    "SEED",
    "SITES",
    "Site",
    "TARGET",
    "add_recency_weights",
    "bias_factor",
    "build_features",
    "climatology",
    "common_window",
    "comparison",
    "daily_energy",
    "design_matrix",
    "get_baseline",
    "get_model",
    "holdout",
    "horizon_index",
    "load_hourly",
    "load_site",
    "metrics",
    "months",
    "restrict",
    "set_seed",
    "walk_forward",
]
