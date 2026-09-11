"""Download archived numerical weather predictions at a fixed forecast lead time."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast.config import SITES

ENDPOINT = "https://historical-forecast-api.open-meteo.com/v1/forecast"
VARIABLES = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "cloud_cover": "cloud_cover_pct",
    "wind_speed_10m": "wind_speed_kmh",
    "shortwave_radiation": "ghi_wm2",
    "direct_normal_irradiance": "dni_wm2",
    "diffuse_radiation": "dhi_wm2",
}
MAX_LEAD_DAYS = 7


def fetch(latitude: float, longitude: float, start: str, end: str, lead_days: int) -> pd.DataFrame:
    suffix = f"_previous_day{lead_days}" if lead_days else ""
    requested = [name + suffix for name in VARIABLES]
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(requested),
            "timezone": "UTC",
        }
    )

    with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=120) as response:
        payload = json.load(response)

    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        raise RuntimeError(f"empty response for lead {lead_days}")

    frame = pd.DataFrame({"datetime_local": pd.to_datetime(hourly["time"])})
    for source, target in VARIABLES.items():
        frame[target] = hourly.get(source + suffix)
    return frame.set_index("datetime_local").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="site_a", choices=sorted(SITES))
    parser.add_argument("--start", default="2026-05-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--leads", nargs="*", type=int, default=[0, 1, 3, 5, 7])
    parser.add_argument("--out", default="data/nwp")
    args = parser.parse_args()

    site = SITES[args.site]
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)

    for lead in args.leads:
        if lead > MAX_LEAD_DAYS:
            print(f"lead {lead}: beyond the {MAX_LEAD_DAYS}-day archive, skipped")
            continue

        frame = fetch(site.latitude, site.longitude, args.start, args.end, lead)
        coverage = frame.notna().all(axis=1).mean() * 100
        path = destination / f"{args.site}_nwp_lead{lead}.csv"
        frame.to_csv(path)
        print(f"lead {lead:>2}d: {len(frame)} hours, {coverage:.1f}% complete -> {path}")
        time.sleep(1)


if __name__ == "__main__":
    main()
