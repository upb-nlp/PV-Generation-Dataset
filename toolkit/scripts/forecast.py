"""Forecast future generation for a site from its recorded inverter history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import Forecaster, daily_energy, horizon_index, set_seed
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    prepare,
    results_path,
)
from pvforecast.forecast import monthly_energy

WEATHER_INDEX_CANDIDATES = ("datetime_local", "datetime", "time", "date")


def load_weather(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    column = next(
        (c for c in WEATHER_INDEX_CANDIDATES if c in frame.columns), frame.columns[0]
    )
    frame[column] = pd.to_datetime(frame[column])
    return frame.set_index(column).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--from-date", default=None, help="first forecast hour, YYYY-MM-DD")
    parser.add_argument("--weather", default=None, help="CSV of forecast weather")
    parser.add_argument(
        "--recency-correction",
        action="store_true",
        help="rescale the climatological part of the horizon by recent output "
             "(off by default: it helps at 6 of 19 origins)",
    )
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    history, site = prepare(args)
    forecaster = Forecaster(
        site,
        feature_cols=feature_columns(args),
        recency_correction=args.recency_correction,
    ).fit(history)

    start = (
        pd.Timestamp(args.from_date)
        if args.from_date
        else history.index.max() + pd.Timedelta(hours=1)
    )
    weather = load_weather(args.weather) if args.weather else None

    hourly = forecaster.predict(horizon_index(start, args.days), weather)
    daily = daily_energy(hourly)

    hourly_path = results_path(args, "forecast", f"{args.site}_hourly.csv")
    daily_path = results_path(args, "forecast", f"{args.site}_daily.csv")
    hourly.to_csv(hourly_path)
    daily.to_csv(daily_path)

    print(f"\nsite            : {site.key} ({site.capacity_kwp:.0f} kWp)")
    print(f"history         : {history.index.min():%Y-%m-%d} to {history.index.max():%Y-%m-%d}")
    print(f"horizon         : {hourly.index.min():%Y-%m-%d} to {hourly.index.max():%Y-%m-%d}")
    print(f"observed weather: {int(hourly['weather_observed'].sum())} of {len(hourly)} hours")
    print(f"total energy    : {daily['energy_kwh'].sum():.0f} kWh")
    print(f"daily mean      : {daily['energy_kwh'].mean():.1f} kWh")
    print("\nmonthly totals")
    print(monthly_energy(hourly).to_string())
    print(f"\nsaved -> {hourly_path}\nsaved -> {daily_path}")


if __name__ == "__main__":
    main()
