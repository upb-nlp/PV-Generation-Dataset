"""Download archived numerical weather predictions issued a fixed number of days before valid time.

Uses the Open-Meteo previous-runs archive: the variable suffix ``_previous_dayN`` holds the value
predicted N x 24 h before the valid time, so lead N is available at any forecast origin at least
N days before the hour it describes. Timestamps are valid times in UTC; radiation variables are
means over the hour ending at the valid time.
"""

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

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
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
PERIODS = {
    "site_a": ("2025-01-03", "2026-07-01"),
    "site_b": ("2025-04-30", "2026-06-01"),
}


def suffix(lead: int) -> str:
    return f"_previous_day{lead}" if lead else ""


def fetch(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    leads: list[int],
) -> tuple[dict[int, pd.DataFrame], dict]:
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "hourly": ",".join(name + suffix(lead) for lead in leads for name in VARIABLES),
            "timezone": "GMT",
        }
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=180) as response:
                payload = json.load(response)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(10 * (attempt + 1))

    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        raise RuntimeError(f"empty response for {start}..{end}")

    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]), name="valid_time_utc")
    frames = {
        lead: pd.DataFrame(
            {target: hourly.get(source + suffix(lead)) for source, target in VARIABLES.items()},
            index=index,
        )
        for lead in leads
    }
    grid = {k: payload.get(k) for k in ("latitude", "longitude", "elevation")}
    return frames, grid


def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    chunks = []
    cursor = first
    while cursor <= last:
        stop = min(cursor + pd.offsets.MonthEnd(0), last)
        chunks.append((cursor.strftime("%Y-%m-%d"), stop.strftime("%Y-%m-%d")))
        cursor = stop + pd.Timedelta(days=1)
    return chunks


def download(latitude, longitude, start, end, leads) -> tuple[dict[int, pd.DataFrame], dict]:
    parts: dict[int, list[pd.DataFrame]] = {lead: [] for lead in leads}
    grid = {}
    for chunk_start, chunk_end in month_chunks(start, end):
        frames, grid = fetch(latitude, longitude, chunk_start, chunk_end, leads)
        for lead, frame in frames.items():
            parts[lead].append(frame)
        time.sleep(1)
    return {lead: pd.concat(frames).sort_index() for lead, frames in parts.items()}, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="*", default=sorted(PERIODS), choices=sorted(PERIODS))
    parser.add_argument("--leads", nargs="*", type=int, default=list(range(MAX_LEAD_DAYS + 1)))
    parser.add_argument("--out", default="data/nwp")
    args = parser.parse_args()

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {}

    for key in args.sites:
        site = SITES[key]
        start, end = PERIODS[key]
        frames, grid = download(site.latitude, site.longitude, start, end, args.leads)
        metadata[key] = {"requested": [site.latitude, site.longitude], "model_grid": grid}
        for lead, frame in frames.items():
            path = destination / f"{key}_lead{lead}.csv"
            frame.to_csv(path)
            coverage = frame.notna().all(axis=1).mean() * 100
            print(f"{key} lead {lead}: {len(frame)} hours, {coverage:.2f}% complete -> {path}")

    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
