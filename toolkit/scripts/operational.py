"""Fixed-origin multi-day forecasts that use only information available at the forecast origin.

For every daily origin O (00:00 UTC) the gradient-boosted forecaster is trained on the history
before O and forecasts the next 14 days hour by hour, with lag inputs seeded from climatology.
The weather for an hour valid ``h`` hours after O comes from the archived run issued
``24 * ceil(h / 24)`` hours before that hour, i.e. no later than O; past the seven-day archive
the weather input is climatological. Reference inputs: climatological weather throughout
(no weather information), the reanalysis (retrospective, not available at O) and the
month-by-hour mean capacity factor itself. A variant applies the trailing 30-day recency
correction to the climatological-weather forecast.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HORIZON_DAYS = 14
MAX_LEAD_DAYS = 7
FIRST_ORIGIN = {"site_a": "2026-01-01", "site_b": "2026-05-01"}
STATE: dict = {}


def setup(data_root: str, site_key: str, selected_path: str, threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    from pvforecast.dataset import load_site
    from pvforecast.features import TARGET, build_features
    from pvforecast.weather import load_nwp, weather_inputs

    raw, site = load_site(data_root, site_key)
    STATE["features"] = build_features(raw, site, weather_inputs(raw, site_key, "nwp"))
    STATE["reanalysis"] = weather_inputs(raw, site_key, "reanalysis")
    STATE["leads"] = {k: load_nwp(site_key, k) for k in range(1, MAX_LEAD_DAYS + 1)}
    STATE["site"] = site
    STATE["threads"] = threads
    params = json.loads(Path(selected_path).read_text())["xgboost"]
    params.pop("val_nrmse", None)
    STATE["params"] = params


def weather_at_origin(horizon: pd.DatetimeIndex, origin: pd.Timestamp) -> pd.DataFrame:
    """Archived forecast issued no later than the origin, or missing past the archive."""
    valid_after = (horizon + pd.Timedelta(hours=1) - origin) / pd.Timedelta(hours=1)
    lead = np.ceil(valid_after / 24).astype(int)
    frame = pd.DataFrame(index=horizon, columns=STATE["leads"][1].columns, dtype=float)
    for k, table in STATE["leads"].items():
        rows = horizon[lead == k]
        frame.loc[rows] = table.reindex(rows).to_numpy()
    return frame


def run_origin(origin: pd.Timestamp) -> list[dict]:
    from pvforecast.features import DAYTIME, FULL_FEATURES, TARGET
    from pvforecast.forecast import Forecaster, _lookup
    from pvforecast.models import get_model

    df, site = STATE["features"], STATE["site"]
    history = df[df.index < origin]
    horizon = pd.date_range(origin, periods=HORIZON_DAYS * 24, freq="h")
    actual = df.reindex(horizon)

    forecaster = Forecaster(
        site,
        FULL_FEATURES,
        model_factory=lambda: get_model("xgboost", **STATE["params"], seed=0, n_jobs=STATE["threads"]),
        recency_correction=True,
    ).fit(history)
    bias = forecaster.bias
    forecaster.bias = 1.0

    variants = {
        "nwp_at_origin": forecaster.predict(horizon, weather_at_origin(horizon, origin))["predicted_cf"],
        "climatological_weather": forecaster.predict(horizon, None)["predicted_cf"],
        "reanalysis": forecaster.predict(horizon, STATE["reanalysis"].reindex(horizon))["predicted_cf"],
        "cf_climatology": pd.Series(
            _lookup(forecaster.clim, horizon, [TARGET])[TARGET].to_numpy(), index=horizon
        ).where(actual["solar_elevation_deg"] > 5.0, 0.0).clip(0, 1),
    }
    forecaster.bias = bias
    variants["climatological_weather_recency"] = forecaster.predict(horizon, None)["predicted_cf"]

    scored = (actual[DAYTIME] == 1) & actual[TARGET].notna()
    day = ((horizon - origin) // pd.Timedelta(days=1)) + 1
    rows = []
    for name, predicted in variants.items():
        error = (actual[TARGET] - predicted).where(scored)
        for k in range(1, HORIZON_DAYS + 1):
            in_day = (day == k) & scored.to_numpy()
            if not in_day.any():
                continue
            e = error[in_day].to_numpy()
            rows.append({
                "origin": origin, "horizon_day": k, "variant": name, "hours": int(in_day.sum()),
                "sse": float(np.sum(e**2)), "sae": float(np.sum(np.abs(e))),
                "energy_actual": float(actual[TARGET][in_day].sum()), "energy_pred": float(predicted[in_day].sum()),
                "bias_factor": bias,
            })
    return rows


def block_bootstrap(per_origin: pd.DataFrame, block: int, draws: int, seed: int = 0) -> np.ndarray:
    """nRMSE resamples over blocks of consecutive origins (rows: origins, columns: sse, hours)."""
    rng = np.random.default_rng(seed)
    n = len(per_origin)
    starts = np.arange(max(n - block + 1, 1))
    sse, hours = per_origin["sse"].to_numpy(), per_origin["hours"].to_numpy()
    out = np.empty(draws)
    for i in range(draws):
        picks = np.concatenate([np.arange(s, min(s + block, n)) for s in rng.choice(starts, size=int(np.ceil(n / block)))])[:n]
        out[i] = np.sqrt(sse[picks].sum() / hours[picks].sum()) * 100
    return out


def summarise(table: pd.DataFrame, block: int = 7, draws: int = 2000) -> pd.DataFrame:
    rows = []
    for (variant, k), group in table.groupby(["variant", "horizon_day"]):
        group = group.sort_values("origin")
        nrmse = np.sqrt(group["sse"].sum() / group["hours"].sum()) * 100
        boot = block_bootstrap(group, block, draws)
        rows.append({"variant": variant, "horizon_day": k, "origins": len(group), "nrmse": nrmse,
                     "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
                     "energy_bias_pct": (group["energy_pred"].sum() / group["energy_actual"].sum() - 1) * 100})

    paired = table.pivot_table(index=["origin", "horizon_day"], columns="variant", values=["sse", "hours"])
    for k in sorted(table["horizon_day"].unique()):
        sub = paired.xs(k, level="horizon_day").sort_index()
        frame = pd.DataFrame({"a": sub[("sse", "nwp_at_origin")], "b": sub[("sse", "climatological_weather")],
                              "h": sub[("hours", "nwp_at_origin")]}).dropna()
        rng = np.random.default_rng(1)
        n = len(frame)
        starts = np.arange(max(n - block + 1, 1))
        diffs = []
        for _ in range(draws):
            picks = np.concatenate([np.arange(s, min(s + block, n)) for s in rng.choice(starts, size=int(np.ceil(n / block)))])[:n]
            h = frame["h"].to_numpy()[picks].sum()
            diffs.append((np.sqrt(frame["a"].to_numpy()[picks].sum() / h) - np.sqrt(frame["b"].to_numpy()[picks].sum() / h)) * 100)
        point = (np.sqrt(frame["a"].sum() / frame["h"].sum()) - np.sqrt(frame["b"].sum() / frame["h"].sum())) * 100
        rows.append({"variant": "difference_nwp_minus_climatological", "horizon_day": k, "origins": n,
                     "nrmse": point, "ci_low": np.percentile(diffs, 2.5), "ci_high": np.percentile(diffs, 97.5)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--sites", nargs="*", default=["site_a", "site_b"])
    parser.add_argument("--selected", default="results/revision/tuning/selected.json")
    parser.add_argument("--results-dir", default="results/revision/operational")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()

    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    for site_key in args.sites:
        setup(args.data_root, site_key, args.selected, args.threads)
        end = STATE["features"].index.max()
        origins = pd.date_range(FIRST_ORIGIN[site_key], end - pd.Timedelta(days=HORIZON_DAYS) + pd.Timedelta(hours=1), freq="D")
        started = time.perf_counter()
        with Pool(args.workers, initializer=setup, initargs=(args.data_root, site_key, args.selected, args.threads)) as pool:
            rows = [row for chunk in pool.map(run_origin, origins, chunksize=1) for row in chunk]
        table = pd.DataFrame(rows)
        table.to_csv(out / f"{site_key}_per_origin.csv", index=False)
        summary = summarise(table)
        summary.to_csv(out / f"{site_key}_by_horizon_day.csv", index=False)
        print(f"{site_key}: {len(origins)} origins {origins[0].date()}..{origins[-1].date()} ({time.perf_counter() - started:.0f}s)")
        print(summary.pivot_table(index="horizon_day", columns="variant", values="nrmse").round(2).to_string())


if __name__ == "__main__":
    main()
