"""Persistence reference forecasters."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .features import TARGET

CLEARSKY = "clearsky_ghi_wm2"
CLEARSKY_FLOOR = 10.0


def _lagged(series: pd.Series, timestamps: pd.DatetimeIndex, hours: int) -> np.ndarray:
    return series.reindex(timestamps - pd.Timedelta(hours=hours)).to_numpy(dtype=np.float64)


def naive_persistence(df: pd.DataFrame) -> Callable:
    capacity_factor = df[TARGET]

    def model_fn(X_train, y_train, w_train):
        anchor = float(y_train[-1]) if len(y_train) else 0.0

        def predict_fn(X_test, timestamps):
            values = _lagged(capacity_factor, timestamps, 1)
            return np.clip(np.nan_to_num(values, nan=anchor), 0.0, 1.0)

        return predict_fn

    return model_fn


def day_persistence(df: pd.DataFrame) -> Callable:
    capacity_factor = df[TARGET]

    def model_fn(X_train, y_train, w_train):
        anchor = float(y_train[-1]) if len(y_train) else 0.0

        def predict_fn(X_test, timestamps):
            values = _lagged(capacity_factor, timestamps, 24)
            return np.clip(np.nan_to_num(values, nan=anchor), 0.0, 1.0)

        return predict_fn

    return model_fn


def smart_persistence(df: pd.DataFrame) -> Callable:
    capacity_factor = df[TARGET]
    if CLEARSKY not in df.columns:
        raise ValueError(f"smart persistence requires the '{CLEARSKY}' column")
    clearsky = df[CLEARSKY]

    def model_fn(X_train, y_train, w_train):
        anchor = float(y_train[-1]) if len(y_train) else 0.0

        def predict_fn(X_test, timestamps):
            yesterday = np.nan_to_num(_lagged(capacity_factor, timestamps, 24), nan=anchor)
            now = clearsky.reindex(timestamps).to_numpy(dtype=np.float64)
            before = _lagged(clearsky, timestamps, 24)

            scale = np.ones_like(yesterday)
            usable = np.isfinite(now) & np.isfinite(before) & (before > CLEARSKY_FLOOR)
            scale[usable] = now[usable] / before[usable]
            return np.clip(yesterday * scale, 0.0, 1.0)

        return predict_fn

    return model_fn


BASELINES: dict[str, Callable[[pd.DataFrame], Callable]] = {
    "naive_persistence": naive_persistence,
    "day_persistence": day_persistence,
    "smart_persistence": smart_persistence,
}

BASELINE_NAMES = tuple(BASELINES)


def get_baseline(name: str, df: pd.DataFrame) -> Callable:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline '{name}'; expected one of {sorted(BASELINES)}")
    return BASELINES[name](df)
