"""Loading and slicing of the released two-site hourly PV dataset.

Every frame returned here sits on a complete hourly UTC grid: the row stamped ``u`` holds the
mean inverter output over ``[u, u + 1 h)`` and the weather that describes the same interval.
Hours without inverter data are kept as rows with missing values, so lags never jump a gap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SITES, Site

LOCAL_COLUMN = "datetime_local"
UTC_COLUMN = "datetime_utc"
POWER_COLUMN = "active_power_kw"
LOCAL_TIMEZONE = "Europe/Bucharest"

WEATHER_COLUMNS = [
    "temp_c",
    "humidity_pct",
    "cloud_cover_pct",
    "wind_speed_kmh",
    "ghi_wm2",
    "dni_wm2",
    "dhi_wm2",
    "precip_mm",
]
DERIVED_COLUMNS = ["solar_elevation_deg", "solar_azimuth_deg", "capacity_factor"]


def site_path(root: str | Path, site_key: str) -> Path:
    return Path(root) / site_key / f"{site_key}_hourly_weather.csv"


def realign_release(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the v1.0 release, whose local-time inverter rows carry UTC-stamped weather.

    Inverter timestamps are local clock time (EET/EEST); the weather columns were joined on the
    same naive stamp although Open-Meteo returned them in UTC. Open-Meteo radiation is the mean
    over the hour ending at its stamp, so the record stamped ``u + 1 h`` describes ``[u, u + 1 h)``.
    """
    local = pd.DatetimeIndex(pd.to_datetime(df[LOCAL_COLUMN]))
    aware = local.tz_localize(LOCAL_TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    keep = ~aware.isna()

    inverter_columns = [
        c for c in df.columns if c not in WEATHER_COLUMNS + DERIVED_COLUMNS + [LOCAL_COLUMN]
    ]
    inverter = df.loc[keep, inverter_columns].copy()
    inverter.index = aware[keep].tz_convert("UTC").tz_localize(None)
    inverter = inverter[~inverter.index.duplicated(keep="first")]

    weather = df[[c for c in WEATHER_COLUMNS if c in df.columns]].copy()
    weather.index = local
    weather = weather[~weather.index.duplicated(keep="first")]

    grid = pd.date_range(inverter.index.min(), inverter.index.max(), freq="h")
    out = inverter.reindex(grid)
    matched = weather.reindex(grid + pd.Timedelta(hours=1))
    out[weather.columns] = matched.to_numpy()
    return out


def load_hourly(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    df = pd.read_csv(path)
    if POWER_COLUMN not in df.columns:
        raise ValueError(f"{path}: missing column '{POWER_COLUMN}'")

    if UTC_COLUMN in df.columns:
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df[UTC_COLUMN])))
        df = df.drop(columns=[c for c in (UTC_COLUMN, LOCAL_COLUMN) if c in df.columns])
        df = df[~df.index.duplicated(keep="first")].sort_index()
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    else:
        df = realign_release(df)

    df = df[df.index >= df.index.min().ceil("D")]
    df.index.name = "datetime"
    return df


def load_site(root: str | Path, site_key: str) -> tuple[pd.DataFrame, Site]:
    if site_key not in SITES:
        raise KeyError(f"unknown site '{site_key}'; expected one of {sorted(SITES)}")
    return load_hourly(site_path(root, site_key)), SITES[site_key]


def months(df: pd.DataFrame) -> list[pd.Period]:
    return sorted(df.index.to_period("M").unique())


def restrict(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    out = df
    if start is not None:
        out = out[out.index >= pd.Period(start, freq="M").to_timestamp()]
    if end is not None:
        out = out[out.index < (pd.Period(end, freq="M") + 1).to_timestamp()]
    if out.empty:
        raise ValueError(f"no rows left after restricting to {start}..{end}")
    return out


def common_window(*frames: pd.DataFrame) -> tuple[str, str]:
    starts = [months(f)[0] for f in frames]
    ends = [months(f)[-1] for f in frames]
    return str(max(starts)), str(min(ends))
