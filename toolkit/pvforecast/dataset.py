"""Loading and slicing of the released two-site hourly PV dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SITES, Site

INDEX_COLUMN = "datetime_local"
POWER_COLUMN = "active_power_kw"


def site_path(root: str | Path, site_key: str) -> Path:
    return Path(root) / site_key / f"{site_key}_hourly_weather.csv"


def load_hourly(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")

    df = pd.read_csv(path, parse_dates=[INDEX_COLUMN])
    df = df.set_index(INDEX_COLUMN).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "datetime"

    if POWER_COLUMN not in df.columns:
        raise ValueError(f"{path}: missing column '{POWER_COLUMN}'")
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
