"""Weather inputs aligned to generation hours: reanalysis, archived forecasts, or none.

``reanalysis`` is the Open-Meteo archive shipped in the dataset. It describes the hour being
predicted and becomes available only days later, so it serves as a retrospective reference.
``nwp`` is the archived forecast issued 24 h before each valid time (lead 1), which a live
system holds before the hour starts. ``none`` drops weather altogether.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import SITES
from .features import WEATHER_FEATURES

DEFAULT_NWP_DIR = Path(__file__).resolve().parents[1] / "data" / "nwp"
WEATHER_SOURCES = ("nwp", "reanalysis", "none")


def nwp_path(site_key: str, lead: int, nwp_dir: str | Path = DEFAULT_NWP_DIR) -> Path:
    """Forecast file for a site, shared by any site registered at the same coordinates."""
    nwp_dir = Path(nwp_dir)
    own = nwp_dir / f"{site_key}_lead{lead}.csv"
    if own.exists():
        return own

    site = SITES[site_key]
    for key, other in SITES.items():
        same_place = (other.latitude, other.longitude) == (site.latitude, site.longitude)
        candidate = nwp_dir / f"{key}_lead{lead}.csv"
        if same_place and candidate.exists():
            return candidate
    raise FileNotFoundError(f"no archived forecast for {site_key} at lead {lead} in {nwp_dir}")


def load_nwp(site_key: str, lead: int = 1, nwp_dir: str | Path = DEFAULT_NWP_DIR) -> pd.DataFrame:
    """Archived forecast at a fixed lead, indexed by the start of the generation hour it describes."""
    frame = pd.read_csv(nwp_path(site_key, lead, nwp_dir), parse_dates=["valid_time_utc"])
    frame = frame.set_index("valid_time_utc").sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame.index = frame.index - pd.Timedelta(hours=1)
    frame.index.name = "datetime"
    return frame[WEATHER_FEATURES]


def weather_inputs(
    df: pd.DataFrame,
    site_key: str,
    source: str,
    nwp_dir: str | Path = DEFAULT_NWP_DIR,
) -> pd.DataFrame | None:
    if source == "reanalysis":
        return df[WEATHER_FEATURES]
    if source == "nwp":
        return load_nwp(site_key, 1, nwp_dir).reindex(df.index)
    if source == "none":
        return None
    raise KeyError(f"unknown weather source '{source}'; expected one of {WEATHER_SOURCES}")
