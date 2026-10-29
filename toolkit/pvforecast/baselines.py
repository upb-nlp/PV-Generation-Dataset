"""Persistence reference forecasters."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .features import TARGET

CLEARSKY = "clearsky_ghi_wm2"
CLEARSKY_FLOOR = 50.0


def _lagged(series: pd.Series, timestamps: pd.DatetimeIndex, hours: int) -> np.ndarray:
    return series.reindex(timestamps - pd.Timedelta(hours=hours)).to_numpy(dtype=np.float64)


def naive_persistence(df: pd.DataFrame) -> Callable:
    """The capacity factor of the previous hour."""
    capacity_factor = df[TARGET]

    def model_fn(X_train, y_train, w_train):
        def predict_fn(X_test, timestamps):
            return np.clip(_lagged(capacity_factor, timestamps, 1), 0.0, 1.0)

        return predict_fn

    return model_fn


def day_persistence(df: pd.DataFrame) -> Callable:
    """The capacity factor at the same hour of the previous day."""
    capacity_factor = df[TARGET]

    def model_fn(X_train, y_train, w_train):
        def predict_fn(X_test, timestamps):
            return np.clip(_lagged(capacity_factor, timestamps, 24), 0.0, 1.0)

        return predict_fn

    return model_fn


def smart_persistence(df: pd.DataFrame) -> Callable:
    """Clear-sky-index persistence: the previous hour rescaled by the change in clear-sky irradiance.

    Predicts CF(t-1) * GHI_cs(t) / GHI_cs(t-1), i.e. it holds the clear-sky index of the previous
    hour constant and lets the deterministic diurnal cycle move the forecast.
    """
    capacity_factor = df[TARGET]
    if CLEARSKY not in df.columns:
        raise ValueError(f"smart persistence requires the '{CLEARSKY}' column")
    clearsky = df[CLEARSKY]

    def model_fn(X_train, y_train, w_train):
        def predict_fn(X_test, timestamps):
            previous = _lagged(capacity_factor, timestamps, 1)
            now = clearsky.reindex(timestamps).to_numpy(dtype=np.float64)
            before = np.maximum(_lagged(clearsky, timestamps, 1), CLEARSKY_FLOOR)
            return np.clip(previous * now / before, 0.0, 1.0)

        return predict_fn

    return model_fn


BASELINES: dict[str, Callable[[pd.DataFrame], Callable]] = {
    "naive_persistence": naive_persistence,
    "smart_persistence": smart_persistence,
    "day_persistence": day_persistence,
}

BASELINE_NAMES = tuple(BASELINES)


def get_baseline(name: str, df: pd.DataFrame) -> Callable:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline '{name}'; expected one of {sorted(BASELINES)}")
    return BASELINES[name](df)
