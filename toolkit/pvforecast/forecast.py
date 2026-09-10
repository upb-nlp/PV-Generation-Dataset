"""Operational forecasting of future generation from an inverter history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Site
from .features import (
    DAYTIME,
    ELEVATION_THRESHOLD,
    FULL_FEATURES,
    LAG_HOURS,
    TARGET,
    WEATHER_FEATURES,
    WEIGHT,
    add_recency_weights,
    design_matrix,
    physics_terms,
    solar_geometry,
    time_encoding,
)
from .models import xgboost

CLIMATOLOGY_COLUMNS = WEATHER_FEATURES + [TARGET]
BIAS_BOUNDS = (0.5, 2.0)
BIAS_MIN_HOURS = 24
BIAS_MIN_CLIMATOLOGY = 0.05
ROLLING_LAGS = (1, 2, 3, 4, 5, 6)


def climatology(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in CLIMATOLOGY_COLUMNS if c in df.columns]
    grouped = df[available].groupby([df.index.month, df.index.hour]).mean()
    grouped.index.names = ["month", "hour"]
    return grouped


def _lookup(table: pd.DataFrame, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    keys = pd.MultiIndex.from_arrays([index.month, index.hour], names=["month", "hour"])
    values = table.reindex(keys)[columns]
    values.index = index
    return values


def bias_factor(
    history: pd.DataFrame,
    clim: pd.DataFrame,
    lookback_days: int = 30,
) -> float:
    if history.empty or TARGET not in history.columns:
        return 1.0

    recent = history[history.index >= history.index.max() - pd.Timedelta(days=lookback_days)]
    if DAYTIME in recent.columns:
        recent = recent[recent[DAYTIME] == 1]
    if len(recent) < BIAS_MIN_HOURS:
        return 1.0

    expected = _lookup(clim, recent.index, [TARGET])[TARGET]
    usable = expected > BIAS_MIN_CLIMATOLOGY
    if usable.sum() < BIAS_MIN_HOURS:
        return 1.0

    ratios = recent.loc[usable, TARGET] / expected[usable]
    return float(np.clip(ratios.median(), *BIAS_BOUNDS))


def horizon_index(start: pd.Timestamp, days: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=days * 24, freq="h")


class Forecaster:
    def __init__(
        self,
        site: Site,
        feature_cols: list[str] = FULL_FEATURES,
        model_factory=xgboost,
        recency_correction: bool = False,
        lookback_days: int = 30,
    ):
        self.site = site
        self.feature_cols = feature_cols
        self.recency_correction = recency_correction
        self.lookback_days = lookback_days
        self.model_factory = model_factory
        self.predict_fn = None
        self.clim: pd.DataFrame | None = None
        self.bias = 1.0
        self.history_end: pd.Timestamp | None = None

    def fit(self, history: pd.DataFrame) -> "Forecaster":
        daytime = history[history[DAYTIME] == 1]
        if daytime.empty:
            raise ValueError("history contains no daytime rows")

        weighted = add_recency_weights(daytime)
        self.predict_fn = self.model_factory()(
            design_matrix(weighted, self.feature_cols),
            weighted[TARGET].to_numpy(dtype=np.float32),
            weighted[WEIGHT].to_numpy(dtype=np.float32),
        )

        self.clim = climatology(history)
        if self.recency_correction:
            self.bias = bias_factor(history, self.clim, self.lookback_days)
        self.history_end = history.index.max()
        return self

    def features(
        self,
        horizon: pd.DatetimeIndex,
        weather: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.clim is None:
            raise RuntimeError("call fit() before building features")

        geometry = solar_geometry(horizon, self.site)
        frame = pd.concat([geometry, time_encoding(horizon)], axis=1)

        conditions = _lookup(self.clim, horizon, WEATHER_FEATURES)
        supplied = pd.Series(False, index=horizon)
        if weather is not None and not weather.empty:
            overlap = weather.reindex(index=horizon, columns=WEATHER_FEATURES)
            supplied = overlap.notna().all(axis=1)
            conditions.loc[supplied] = overlap.loc[supplied]
        frame[WEATHER_FEATURES] = conditions

        lags = {
            hours: _lookup(self.clim, horizon - pd.Timedelta(hours=hours), [TARGET])[TARGET]
            for hours in ROLLING_LAGS
        }
        for hours in LAG_HOURS:
            frame[f"cf_lag_{hours}h"] = lags[hours].to_numpy()

        window = np.column_stack([lags[h].to_numpy() for h in ROLLING_LAGS])
        frame["cf_roll3h_mean"] = window[:, :3].mean(axis=1)
        frame["cf_roll3h_std"] = window[:, :3].std(axis=1, ddof=1)
        frame["cf_roll6h_mean"] = window.mean(axis=1)

        frame[["temp_efficiency_factor", "cloud_transmission", "clearsky_index"]] = (
            physics_terms(
                frame["temp_c"],
                frame["cloud_cover_pct"],
                frame["ghi_wm2"],
                frame["clearsky_ghi_wm2"],
            )
        )

        frame["weather_observed"] = supplied.to_numpy()
        return frame.fillna(0.0)

    def predict(
        self,
        horizon: pd.DatetimeIndex,
        weather: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if self.predict_fn is None:
            raise RuntimeError("call fit() before predict()")

        frame = self.features(horizon, weather)
        predicted = np.asarray(
            self.predict_fn(design_matrix(frame, self.feature_cols), horizon), dtype=np.float64
        )

        night = frame["solar_elevation_deg"].to_numpy() <= ELEVATION_THRESHOLD
        climatological = ~frame["weather_observed"].to_numpy()
        predicted = np.where(climatological, predicted * self.bias, predicted)
        predicted = np.clip(np.where(night, 0.0, predicted), 0.0, 1.0)

        return pd.DataFrame(
            {
                "predicted_cf": predicted,
                "predicted_kw": predicted * self.site.capacity_kwp,
                "weather_observed": frame["weather_observed"].to_numpy(),
            },
            index=horizon,
        )


def daily_energy(hourly: pd.DataFrame) -> pd.DataFrame:
    daily = hourly["predicted_kw"].resample("D").sum().to_frame("energy_kwh")
    daily["weather_observed"] = hourly["weather_observed"].resample("D").max()
    return daily


def monthly_energy(hourly: pd.DataFrame) -> pd.DataFrame:
    monthly = hourly["predicted_kw"].resample("MS").sum().to_frame("energy_kwh")
    monthly.index = monthly.index.to_period("M").astype(str)
    return monthly
