"""Multi-step forecast accuracy by horizon, scored against recorded generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import Forecaster, set_seed
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    prepare,
    results_path,
    write_table,
)
from pvforecast.features import DAYTIME, TARGET, WEATHER_FEATURES
from pvforecast.forecast import _lookup


def segment_metrics(frame: pd.DataFrame, capacity_kwp: float) -> dict[str, float]:
    daytime = frame[frame[DAYTIME] == 1]
    if len(daytime) < 2:
        return {}

    error = daytime["actual_cf"] - daytime["predicted_cf"]
    variance = float(((daytime["actual_cf"] - daytime["actual_cf"].mean()) ** 2).sum())

    daily_actual = frame["actual_kw"].resample("D").sum()
    daily_predicted = frame["predicted_kw"].resample("D").sum()
    valid = daily_actual > 0

    return {
        "days": int(valid.sum()),
        "nrmse": float(np.sqrt((error**2).mean()) * 100.0),
        "mae_kw": float(error.abs().mean() * capacity_kwp),
        "r2": float(1.0 - (error**2).sum() / variance) if variance > 1e-10 else float("nan"),
        "daily_mape": float(
            ((daily_predicted[valid] - daily_actual[valid]).abs() / daily_actual[valid]).mean()
            * 100.0
        ),
        "energy_bias_pct": float(
            (daily_predicted[valid].sum() / daily_actual[valid].sum() - 1.0) * 100.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--forecast-from", default="2026-05", help="first forecast month, YYYY-MM")
    parser.add_argument("--weather-days", type=int, default=14)
    parser.add_argument("--recency-correction", action="store_true")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    features, site = prepare(args)
    columns = feature_columns(args)

    start = pd.Period(args.forecast_from, freq="M").to_timestamp()
    history = features[features.index < start]
    actual = features[features.index >= start]
    if history.empty or actual.empty:
        raise SystemExit(f"no data on one side of {args.forecast_from}")

    forecaster = Forecaster(
        site, feature_cols=columns, recency_correction=args.recency_correction
    ).fit(history)

    horizon = pd.date_range(start, actual.index.max(), freq="h")
    weather_end = start + pd.Timedelta(days=args.weather_days)
    weather = actual.loc[actual.index < weather_end, WEATHER_FEATURES]

    hourly = forecaster.predict(horizon, weather)
    joined = hourly.join(
        actual[[TARGET, DAYTIME]].rename(columns={TARGET: "actual_cf"}), how="inner"
    )
    joined["actual_kw"] = joined["actual_cf"] * site.capacity_kwp

    climatological = _lookup(forecaster.clim, joined.index, [TARGET])[TARGET].to_numpy()
    joined["climatology_kw"] = climatological * site.capacity_kwp * forecaster.bias

    offsets = (joined.index - start).days
    joined["week"] = offsets // 7 + 1
    joined["regime"] = np.where(
        joined.index < weather_end, "observed weather", "climatological weather"
    )

    weekly = pd.DataFrame(
        [
            {"week": week, "from": frame.index.min().date(), **segment_metrics(frame, site.capacity_kwp)}
            for week, frame in joined.groupby("week")
        ]
    )

    rows = []
    for regime, frame in joined.groupby("regime", sort=False):
        rows.append({"segment": regime, **segment_metrics(frame, site.capacity_kwp)})
        reference = pd.DataFrame(
            {
                "predicted_cf": frame["climatology_kw"] / site.capacity_kwp,
                "predicted_kw": frame["climatology_kw"],
                "actual_cf": frame["actual_cf"],
                "actual_kw": frame["actual_kw"],
                DAYTIME: frame[DAYTIME],
            },
            index=frame.index,
        )
        rows.append(
            {"segment": f"{regime} (climatology only)", **segment_metrics(reference, site.capacity_kwp)}
        )
    rows.append({"segment": "full horizon", **segment_metrics(joined, site.capacity_kwp)})

    label = args.label or f"{args.site}_{args.forecast_from}"
    print(f"\nrecency bias factor: {forecaster.bias:.3f}")
    print(f"training history   : {history.index.min().date()} to {history.index.max().date()}")
    print(f"forecast horizon   : {joined.index.min().date()} to {joined.index.max().date()}")

    write_table(weekly, results_path(args, "tables", f"horizon_weekly_{label}.csv"), "By horizon week")
    write_table(
        pd.DataFrame(rows), results_path(args, "tables", f"horizon_regime_{label}.csv"), "By regime"
    )


if __name__ == "__main__":
    main()
