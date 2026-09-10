"""Forecast accuracy as a function of the lead time of the weather input."""

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


def score(frame: pd.DataFrame, capacity_kwp: float) -> dict[str, float]:
    daytime = frame[frame[DAYTIME] == 1]
    error = daytime["actual_cf"] - daytime["predicted_cf"]
    variance = float(((daytime["actual_cf"] - daytime["actual_cf"].mean()) ** 2).sum())

    daily_actual = frame["actual_kw"].resample("D").sum()
    daily_predicted = frame["predicted_kw"].resample("D").sum()

    return {
        "nrmse": round(float(np.sqrt((error**2).mean()) * 100), 2),
        "mae_kw": round(float(error.abs().mean() * capacity_kwp), 2),
        "r2": round(float(1.0 - (error**2).sum() / variance), 3),
        "energy_bias_pct": round(
            float((daily_predicted.sum() / daily_actual.sum() - 1.0) * 100), 1
        ),
    }


def load_nwp(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["datetime_local"])
    return frame.set_index("datetime_local").sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--forecast-from", default="2026-05")
    parser.add_argument("--nwp-dir", default="data/nwp")
    parser.add_argument("--leads", nargs="*", type=int, default=[0, 1, 3, 5, 7])
    parser.add_argument("--near-days", type=int, default=14)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    features, site = prepare(args)
    columns = feature_columns(args)

    start = pd.Period(args.forecast_from, freq="M").to_timestamp()
    history = features[features.index < start]
    actual = features[features.index >= start]
    forecaster = Forecaster(site, feature_cols=columns).fit(history)

    horizon = pd.date_range(start, actual.index.max(), freq="h")
    near_end = start + pd.Timedelta(days=args.near_days)

    sources: dict[str, pd.DataFrame | None] = {"climatology": None}
    for lead in args.leads:
        path = Path(args.nwp_dir) / f"{args.site}_nwp_lead{lead}.csv"
        if path.exists():
            label = "analysis (lead 0)" if lead == 0 else f"NWP lead {lead} d"
            sources[label] = load_nwp(path)
    sources["reanalysis (perfect)"] = actual[WEATHER_FEATURES]

    reference = actual["ghi_wm2"]
    rows = []
    for label, weather in sources.items():
        predicted = forecaster.predict(horizon, weather)
        joined = predicted.join(
            actual[[TARGET, DAYTIME]].rename(columns={TARGET: "actual_cf"}), how="inner"
        )
        joined["actual_kw"] = joined["actual_cf"] * site.capacity_kwp

        near = joined[joined.index < near_end]
        row = {"weather_source": label}
        row.update({f"near_{k}": v for k, v in score(near, site.capacity_kwp).items()})
        row.update({f"full_{k}": v for k, v in score(joined, site.capacity_kwp).items()})

        if weather is not None and "ghi_wm2" in weather.columns:
            aligned = weather["ghi_wm2"].reindex(reference.index)
            day = actual[DAYTIME] == 1
            row["ghi_rmse"] = round(
                float(np.sqrt(((aligned[day] - reference[day]) ** 2).mean())), 1
            )
        else:
            row["ghi_rmse"] = None
        rows.append(row)

    write_table(
        pd.DataFrame(rows),
        results_path(args, "tables", f"nwp_lead_time_{args.site}_{args.forecast_from}.csv"),
        f"Forecast accuracy by weather lead time - {args.site}, "
        f"{start.date()} onward (near window = first {args.near_days} days)",
    )


if __name__ == "__main__":
    main()
