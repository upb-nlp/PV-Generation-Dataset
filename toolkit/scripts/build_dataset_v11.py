"""Build release 1.1 of the two-site dataset from release 1.0.

Release 1.0 stamped inverter rows in local clock time but joined weather and solar position
that were computed for the same stamps read as UTC. Release 1.1 keeps every measured value and
re-pairs them: each row is one UTC hour [u, u + 1 h), carries its local start time, the
Open-Meteo record valid at u + 1 h (radiation averaged over the hour), and the solar position at
the midpoint of the hour. Hours without inverter data stay as rows with empty measurements.
The archived forecasts used in the paper are copied alongside.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast.config import SITES
from pvforecast.dataset import LOCAL_TIMEZONE, WEATHER_COLUMNS, realign_release, site_path
from pvforecast.features import solar_geometry

INVERTER_ORDER = [
    "active_power_kw", "dc_input_power_kw", "daily_energy_kwh", "cumulative_energy_kwh",
    "inverter_temp_c", "grid_freq_hz", "power_factor", "efficiency_pct", "inverter_status", "is_generating",
]


def build(source: Path, site_key: str) -> pd.DataFrame:
    site = SITES[site_key]
    aligned = realign_release(pd.read_csv(site_path(source, site_key)))
    geometry = solar_geometry(aligned.index, site)

    out = pd.DataFrame(index=aligned.index)
    out["datetime_utc"] = aligned.index.strftime("%Y-%m-%d %H:%M:%S")
    out["datetime_local"] = (
        aligned.index.tz_localize("UTC").tz_convert(LOCAL_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S%z")
    )
    for column in INVERTER_ORDER + WEATHER_COLUMNS:
        if column in aligned.columns:
            out[column] = aligned[column]
    out["solar_elevation_deg"] = geometry["solar_elevation_deg"].round(4)
    out["solar_azimuth_deg"] = geometry["solar_azimuth_deg"].round(4)
    out["capacity_factor"] = (aligned["active_power_kw"] / site.capacity_kwp).clip(0.0, 1.0).round(4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="a checkout of dataset release 1.0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--nwp-dir", default="data/nwp")
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; remove it first")

    for site_key in ("site_a", "site_b"):
        frame = build(source, site_key)
        target = out / site_key / f"{site_key}_hourly_weather.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        shutil.copytree(source / site_key / "raw", out / site_key / "raw")
        print(f"{site_key}: {len(frame)} hours, {int(frame['active_power_kw'].isna().sum())} without inverter data -> {target}")

    nwp_out = out / "archived_forecasts"
    nwp_out.mkdir(parents=True)
    for path in sorted(Path(args.nwp_dir).glob("site_a_lead*.csv")):
        shutil.copy2(path, nwp_out / path.name.replace("site_a_", "site_a_and_b_"))
    for name in ("LICENSE", "CITATION.cff", "README.md"):
        if (source / name).exists():
            shutil.copy2(source / name, out / name)
    print(f"archived forecasts -> {nwp_out}")


if __name__ == "__main__":
    main()
