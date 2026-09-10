"""Construction of the hourly feature matrix and the capacity-factor target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Site
from .dataset import POWER_COLUMN

TARGET = "capacity_factor"
DAYTIME = "is_daytime"
WEIGHT = "sample_weight"

ELEVATION_THRESHOLD = 5.0
LAG_HOURS = (1, 2, 3, 6)
TEMPERATURE_COEFFICIENT = -0.0045
REFERENCE_TEMPERATURE = 25.0

SOLAR_FEATURES = [
    "solar_elevation_deg",
    "solar_azimuth_deg",
    "solar_zenith_deg",
    "clearsky_ghi_wm2",
    "elevation_x_capacity",
]
WEATHER_FEATURES = [
    "temp_c",
    "humidity_pct",
    "cloud_cover_pct",
    "wind_speed_kmh",
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
]
TIME_FEATURES = [
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "month_sin",
    "month_cos",
]
LAG_FEATURES = [f"cf_lag_{h}h" for h in LAG_HOURS]
ROLLING_FEATURES = ["cf_roll3h_mean", "cf_roll3h_std", "cf_roll6h_mean"]
PHYSICS_FEATURES = ["temp_efficiency_factor", "cloud_transmission", "clearsky_index"]

BASE_FEATURES = (
    SOLAR_FEATURES + WEATHER_FEATURES + TIME_FEATURES + LAG_FEATURES + ROLLING_FEATURES
)
FULL_FEATURES = BASE_FEATURES + PHYSICS_FEATURES
CORE_FEATURES = LAG_FEATURES + ROLLING_FEATURES

FEATURE_SETS: dict[str, list[str]] = {
    "core": CORE_FEATURES,
    "base": BASE_FEATURES,
    "full": FULL_FEATURES,
}

FEATURE_GROUPS: dict[str, list[str]] = {
    "solar_geometry": SOLAR_FEATURES,
    "weather": WEATHER_FEATURES,
    "time_encoding": TIME_FEATURES,
    "lag": LAG_FEATURES,
    "rolling": ROLLING_FEATURES,
    "physics": PHYSICS_FEATURES,
}

REQUIRED_COLUMNS = [POWER_COLUMN] + WEATHER_FEATURES


def solar_geometry(index: pd.DatetimeIndex, site: Site) -> pd.DataFrame:
    import pvlib

    location = pvlib.location.Location(site.latitude, site.longitude, tz="UTC")
    position = location.get_solarposition(index)
    clearsky = location.get_clearsky(index)
    elevation = position["elevation"].to_numpy().clip(min=0.0)

    return pd.DataFrame(
        {
            "solar_elevation_deg": elevation,
            "solar_azimuth_deg": position["azimuth"].to_numpy(),
            "solar_zenith_deg": position["zenith"].to_numpy(),
            "clearsky_ghi_wm2": clearsky["ghi"].to_numpy(),
            "elevation_x_capacity": elevation * site.capacity_kwp / 90.0,
        },
        index=index,
    )


def time_encoding(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour.to_numpy()
    day_of_year = index.day_of_year.to_numpy()
    month = index.month.to_numpy()

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "doy_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        },
        index=index,
    )


def physics_terms(
    temperature: pd.Series,
    cloud_cover: pd.Series,
    ghi: pd.Series,
    clearsky_ghi: pd.Series,
) -> pd.DataFrame:
    efficiency = 1.0 + TEMPERATURE_COEFFICIENT * (temperature - REFERENCE_TEMPERATURE)
    transmission = 1.0 - 0.8 * cloud_cover / 100.0
    index = ghi / clearsky_ghi.clip(lower=1.0)

    return pd.DataFrame(
        {
            "temp_efficiency_factor": efficiency.clip(0.7, 1.1),
            "cloud_transmission": transmission.clip(0.2, 1.0),
            "clearsky_index": index.clip(0.0, 1.2),
        }
    )


def build_features(df: pd.DataFrame, site: Site) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"input frame is missing required columns: {missing}")

    out = df.copy()
    out[TARGET] = (out[POWER_COLUMN] / site.capacity_kwp).clip(0.0, 1.0).fillna(0.0)

    geometry = solar_geometry(out.index, site)
    out[geometry.columns] = geometry

    encoding = time_encoding(out.index)
    out[encoding.columns] = encoding

    capacity_factor = out[TARGET]
    for hours in LAG_HOURS:
        out[f"cf_lag_{hours}h"] = capacity_factor.shift(hours)

    # rolling windows close one step before the predicted hour, so no feature
    # can contain the value being predicted
    window_source = capacity_factor.shift(1)
    out["cf_roll3h_mean"] = window_source.rolling(3, min_periods=1).mean()
    out["cf_roll3h_std"] = window_source.rolling(3, min_periods=1).std().fillna(0.0)
    out["cf_roll6h_mean"] = window_source.rolling(6, min_periods=1).mean()

    physics = physics_terms(
        out["temp_c"], out["cloud_cover_pct"], out["ghi_wm2"], out["clearsky_ghi_wm2"]
    )
    out[physics.columns] = physics

    out[DAYTIME] = (out["solar_elevation_deg"] > ELEVATION_THRESHOLD).astype(int)
    return out


def add_recency_weights(
    df: pd.DataFrame,
    decay_per_month: float = 0.92,
    cutoff_months: int = 18,
) -> pd.DataFrame:
    if df.empty:
        return df

    age_months = np.asarray((df.index.max() - df.index).days, dtype=float) / 30.44
    within = age_months <= cutoff_months
    kept = df[within].copy()

    weights = decay_per_month ** age_months[within]
    kept[WEIGHT] = weights / weights.mean()
    return kept


def design_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return (
        df.reindex(columns=feature_cols, fill_value=0.0)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def daytime_mask(df: pd.DataFrame) -> np.ndarray:
    if DAYTIME in df.columns:
        return (df[DAYTIME] == 1).to_numpy()
    return np.ones(len(df), dtype=bool)
