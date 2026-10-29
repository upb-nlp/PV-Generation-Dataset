"""Run the walk-forward experiments of the revised study as independent, resumable jobs.

Each job is one model, one seed and one data configuration; it writes per-month scores and
per-hour predictions under ``<results-dir>/<experiment>/``. Every table of the paper is built
from these files, so a configuration that appears in several tables comes from a single run.
All jobs score the same hours: daytime hours whose 24-hour input window is complete under both
weather sources.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEEDS = (0, 1, 2)
MODELS = ("xgboost", "lightgbm", "vanilla_lstm", "cnn_lstm", "attention_lstm", "transformer", "nbeats")
BASELINES = ("naive_persistence", "smart_persistence", "day_persistence")
TREES = ("xgboost", "lightgbm", "xgboost_warm", "lightgbm_warm")
CONFIGS = {"nwp": ("nwp", "full"), "reanalysis": ("reanalysis", "full"), "inverter_only": ("none", "endogenous")}
WINDOWS = (1, 2, 3, 4, 5, 6, 9, 12)
HOLDOUT_MONTHS = 3
DECAYS = (0.85, 0.88, 0.92, 0.95, 0.98, 1.0)
OFFSETS_KM = (5.0, 10.0)
BEARINGS = tuple(range(0, 360, 45))
TREE_GRID = {"n_estimators": [300, 500, 800], "max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1]}
ABLATIONS = {
    "lag": ["lag"],
    "rolling": ["rolling"],
    "weather": ["weather"],
    "physics": ["physics"],
    "weather_and_physics": ["weather", "physics"],
    "solar_geometry": ["solar_geometry"],
    "time_encoding": ["time_encoding"],
    "all_but_lag_rolling": ["solar_geometry", "time_encoding", "weather", "physics"],
}
COST = {"transformer": 10, "transformer_warm": 10, "nbeats": 6, "attention_lstm": 5, "cnn_lstm": 3, "vanilla_lstm": 3}

ARGS: dict = {}


def jobs_for(experiments: list[str]) -> list[dict]:
    jobs: list[dict] = []

    def add(**job):
        jobs.append({"kind": "walk", "site": "site_a", "weather": "nwp", "feature_set": "full",
                     "start": None, "end": None, "min_train": 6, "decay": 0.92, "drop": None,
                     "offset": None, "seed": 0, **job})

    if "benchmark" in experiments:
        for config, (weather, features) in CONFIGS.items():
            for model, seed in itertools.product(MODELS, SEEDS):
                add(experiment="benchmark", tag=config, weather=weather, feature_set=features, model=model, seed=seed)
            for baseline in BASELINES:
                add(experiment="benchmark", tag=config, weather=weather, feature_set=features, model=baseline)
    if "cross_site" in experiments:
        for site, start, end in (("site_a", "2025-05", "2026-05"), ("site_b", None, None)):
            for model, seed in itertools.product(MODELS, SEEDS):
                add(experiment="cross_site", tag=site, site=site, start=start, end=end, model=model, seed=seed)
            for baseline in BASELINES:
                add(experiment="cross_site", tag=site, site=site, start=start, end=end, model=baseline)
    if "warm_start" in experiments:
        for model, seed in itertools.product(("xgboost_warm", "lightgbm_warm", "transformer_warm"), SEEDS):
            add(experiment="warm_start", tag="nwp", model=model, seed=seed)
    if "physics" in experiments:
        for model, seed in itertools.product(("xgboost", "lightgbm", "transformer", "cnn_lstm"), SEEDS):
            add(experiment="physics", tag="base", feature_set="base", model=model, seed=seed)
    if "ablation" in experiments:
        for (name, groups), seed in itertools.product(ABLATIONS.items(), SEEDS):
            add(experiment="ablation", tag=name, model="xgboost", drop=groups, seed=seed)
    if "decay" in experiments:
        for model, decay, seed in itertools.product(("xgboost", "lightgbm"), DECAYS, SEEDS):
            add(experiment="decay", tag=f"decay{decay:.2f}", model=model, decay=decay, seed=seed)
    if "site_b_geometry" in experiments:
        for km, bearing, seed in itertools.product(OFFSETS_KM, BEARINGS, SEEDS):
            add(experiment="site_b_geometry", tag=f"{int(km)}km_{bearing:03d}", site="site_b",
                model="xgboost", offset=(km, bearing), seed=seed)
        for seed in SEEDS:
            add(experiment="site_b_geometry", tag="proxy", site="site_b", model="xgboost", seed=seed)
    if "training_window" in experiments:
        for model, seed in itertools.product(MODELS, SEEDS):
            jobs.append({"kind": "window", "experiment": "training_window", "tag": "nwp", "site": "site_a",
                         "weather": "nwp", "feature_set": "full", "model": model, "seed": seed})
        for baseline in BASELINES:
            jobs.append({"kind": "window", "experiment": "training_window", "tag": "nwp", "site": "site_a",
                         "weather": "nwp", "feature_set": "full", "model": baseline, "seed": 0})
        for model in ("xgboost", "lightgbm"):
            jobs.append({"kind": "window_grid", "experiment": "training_window", "tag": "grid", "site": "site_a",
                         "weather": "nwp", "feature_set": "full", "model": model, "seed": 0})
    return jobs


def output_path(job: dict) -> Path:
    name = f"{job['model']}_s{job['seed']}" if job["model"] not in BASELINES else job["model"]
    suffix = {"walk": "_walk_forward.csv", "window": "_windows.csv", "window_grid": "_window_grid.csv"}[job["kind"]]
    return Path(ARGS["results_dir"]) / job["experiment"] / job["tag"] / f"{name}{suffix}"


def priority(job: dict) -> tuple:
    deep = job["model"] not in TREES + BASELINES
    order = ["benchmark", "cross_site", "training_window", "warm_start", "physics"]
    rank = order.index(job["experiment"]) if job["experiment"] in order else len(order)
    if job["experiment"] == "benchmark" and job["tag"] != "nwp":
        rank = 3
    return (deep, rank, -COST.get(job["model"], 1))


@lru_cache(maxsize=8)
def frames(site_key: str, start: str | None, end: str | None):
    from pvforecast.dataset import load_site, restrict
    from pvforecast.features import FULL_FEATURES, build_features, scored_mask
    from pvforecast.weather import weather_inputs

    raw, site = load_site(ARGS["data_root"], site_key)
    if start or end:
        raw = restrict(raw, start, end)
    built = {source: build_features(raw, site, weather_inputs(raw, site_key, source)) for source in ("nwp", "reanalysis", "none")}
    import pandas as pd

    mask = pd.Series(
        scored_mask(built["nwp"], FULL_FEATURES) & scored_mask(built["reanalysis"], FULL_FEATURES),
        index=raw.index,
    )
    return built, mask, site


def model_kwargs(name: str, seed: int, threads: int) -> dict:
    selected = json.loads(Path(ARGS["selected"]).read_text())
    base = name.removesuffix("_warm")
    params = dict(selected.get(base, {}))
    params.pop("val_nrmse", None)
    if base in ("xgboost", "lightgbm"):
        return {**params, "seed": seed, "n_jobs": threads}
    return {**params, "seed": seed}


def with_offset_geometry(df, site, offset):
    import math

    from pvforecast.config import Site
    from pvforecast.features import physics_terms, solar_geometry

    km, bearing = offset
    north = km * math.cos(math.radians(bearing)) / 111.32
    east = km * math.sin(math.radians(bearing)) / (111.32 * math.cos(math.radians(site.latitude)))
    moved = Site(site.key, site.latitude + north, site.longitude + east, site.capacity_kwp)
    out = df.copy()
    geometry = solar_geometry(out.index, moved)
    out[geometry.columns] = geometry
    out[["temp_efficiency_factor", "cloud_transmission", "clearsky_index"]] = physics_terms(
        out["temp_c"], out["cloud_cover_pct"], out["ghi_wm2"], out["clearsky_ghi_wm2"]
    ).to_numpy()
    return out


def run_job(job: dict) -> str:
    threads = ARGS["threads"]
    out = output_path(job)
    if out.exists():
        return f"skip {out}"

    import torch

    torch.set_num_threads(threads)
    from pvforecast.baselines import get_baseline
    from pvforecast.evaluation import holdout, metrics, walk_forward
    from pvforecast.features import FEATURE_GROUPS, FEATURE_SETS, scored_mask
    from pvforecast.models import get_model
    import pandas as pd

    built, mask, site = frames(job["site"], job.get("start"), job.get("end"))
    df = built[job["weather"]]
    columns = list(FEATURE_SETS[job["feature_set"]])
    if job.get("drop"):
        removed = {c for group in job["drop"] for c in FEATURE_GROUPS[group]}
        columns = [c for c in columns if c not in removed]
    if job.get("offset"):
        df = with_offset_geometry(df, site, job["offset"])

    def make_model(name, seed, **extra):
        if name in BASELINES:
            return get_baseline(name, df)
        return get_model(name, **{**model_kwargs(name, seed, threads), **extra})

    started = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)

    if job["kind"] == "walk":
        evaluation = walk_forward(df, make_model(job["model"], job["seed"]), job["model"], feature_cols=columns,
                                  min_train_months=job["min_train"], capacity_kwp=site.capacity_kwp,
                                  decay_per_month=job["decay"], mask=mask)
        evaluation.save(out, predictions=True)
        summary = f"{evaluation.mean()['nrmse']:.2f}%"
    else:
        months = sorted(df.index.to_period("M").unique())
        test_start = months[-HOLDOUT_MONTHS].to_timestamp()
        test_df = df[df.index >= test_start]
        test_mask = mask.loc[test_df.index].to_numpy()
        history = df[df.index < test_start]
        rows = []
        if job["kind"] == "window" and job["model"] in BASELINES:
            scores = holdout(history, test_df, make_model(job["model"], 0), job["model"], columns,
                             site.capacity_kwp, test_mask=test_mask)
            rows.append({"window_months": None, **scores})
        else:
            grid = [dict(zip(TREE_GRID, v)) for v in itertools.product(*TREE_GRID.values())] if job["kind"] == "window_grid" else [{}]
            for window, extra in itertools.product(WINDOWS, grid):
                if window > len(months) - HOLDOUT_MONTHS:
                    continue
                first = months[-(HOLDOUT_MONTHS + window)]
                last = months[-(HOLDOUT_MONTHS + 1)]
                train_df = history[history.index >= first.to_timestamp()]
                scores = holdout(train_df, test_df, make_model(job["model"], job["seed"], **extra), job["model"],
                                 columns, site.capacity_kwp, test_mask=test_mask)
                rows.append({"window_months": window, "first_month": str(first), "last_month": str(last),
                             "holdout": f"{months[-HOLDOUT_MONTHS]}..{months[-1]}", **extra, **scores})
        pd.DataFrame(rows).to_csv(out, index=False)
        summary = f"{len(rows)} rows"

    return f"done {out.relative_to(ARGS['results_dir'])}  {summary}  ({time.perf_counter() - started:.0f}s)"


def init_worker(args: dict) -> None:
    ARGS.update(args)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(args["threads"])
    logging.disable(logging.WARNING)
    import warnings

    warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--results-dir", default="results/revision")
    parser.add_argument("--selected", default="results/revision/tuning/selected.json")
    parser.add_argument("--experiments", nargs="*", default=[
        "benchmark", "cross_site", "training_window", "warm_start", "physics", "ablation", "decay", "site_b_geometry"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--only-trees", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    settings = {"data_root": args.data_root, "results_dir": args.results_dir, "selected": args.selected,
                "threads": args.threads}
    ARGS.update(settings)
    jobs = sorted(jobs_for(args.experiments), key=priority)
    if args.only_trees:
        jobs = [j for j in jobs if j["model"] in TREES + BASELINES]
    pending = [j for j in jobs if not output_path(j).exists()]
    print(f"{len(jobs)} jobs, {len(pending)} pending", flush=True)
    if args.dry_run:
        for job in pending:
            print(output_path(job))
        return

    with Pool(args.workers, initializer=init_worker, initargs=(settings,)) as pool:
        for line in pool.imap_unordered(run_job, pending, chunksize=1):
            print(time.strftime("%H:%M:%S"), line, flush=True)


if __name__ == "__main__":
    main()
