"""Hyperparameter selection on a temporal validation split, applied identically to every model.

Train on the first five months of the site, validate on the sixth (which precedes every
walk-forward test month). Each family searches nine configurations over a capacity axis and a
learning-rate axis; training length (boosting rounds or epochs) is chosen by early stopping on
the validation loss. The configuration with the lowest validation nRMSE is written to
``selected.json`` and used unchanged in every experiment.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TREE_GRID = {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.05, 0.1]}
DEEP_GRID = {"width": [32, 64, 128], "learning_rate": [3e-4, 1e-3, 3e-3]}
TREE_MODELS = ("xgboost", "lightgbm")
DEEP_MODELS = ("vanilla_lstm", "cnn_lstm", "attention_lstm", "transformer", "nbeats")
MAX_ROUNDS = 3000
ROUND_PATIENCE = 100


def split(args):
    from pvforecast.cli import prepare
    from pvforecast.evaluation import training_frame
    from pvforecast.features import FEATURE_SETS, TARGET, WEIGHT, design_matrix, scored_mask

    df, site = prepare(args)
    columns = FEATURE_SETS[args.feature_set]
    months = sorted(df.index.to_period("M").unique())
    validation = months[args.train_months]
    start, end = validation.to_timestamp(), (validation + 1).to_timestamp()

    train_df = training_frame(df[df.index < start], columns, 0.92, 18)
    val_df = df[(df.index >= start) & (df.index < end)]
    mask = scored_mask(df, columns)[(df.index >= start) & (df.index < end)]
    return {
        "X_train": design_matrix(train_df, columns),
        "y_train": train_df[TARGET].fillna(0.0).to_numpy(np.float32),
        "w_train": train_df[WEIGHT].to_numpy(np.float32),
        "X_val": design_matrix(val_df, columns),
        "y_val": val_df[TARGET].to_numpy(np.float64),
        "mask": mask,
        "capacity": site.capacity_kwp,
        "period": f"train {train_df.index.min().date()}..{train_df.index.max().date()}, validate {validation}",
    }


def tree_trial(model: str, params: dict, data: dict, threads: int) -> dict:
    keep = data["w_train"] > 0
    X, y, w = data["X_train"][keep], data["y_train"][keep], data["w_train"][keep]
    Xv, yv = data["X_val"][data["mask"]], data["y_val"][data["mask"]]

    if model == "xgboost":
        from xgboost import XGBRegressor
        from pvforecast.models import XGBOOST_PARAMS

        est = XGBRegressor(**{**XGBOOST_PARAMS, **params, "n_estimators": MAX_ROUNDS, "n_jobs": threads,
                              "random_state": 0, "early_stopping_rounds": ROUND_PATIENCE, "eval_metric": "rmse"})
        est.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], verbose=False)
        rounds = int(est.best_iteration) + 1
        pred = est.predict(Xv, iteration_range=(0, rounds))
    else:
        import lightgbm as lgb
        from pvforecast.models import LIGHTGBM_PARAMS

        est = lgb.LGBMRegressor(**{**LIGHTGBM_PARAMS, **params, "n_estimators": MAX_ROUNDS, "n_jobs": threads,
                                   "random_state": 0})
        est.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="rmse",
                callbacks=[lgb.early_stopping(ROUND_PATIENCE, verbose=False)])
        rounds = int(est.best_iteration_)
        pred = est.predict(Xv, num_iteration=rounds)

    error = yv - np.clip(pred, 0, 1)
    return {"model": model, **params, "n_estimators": rounds,
            "val_nrmse": float(np.sqrt(np.mean(error**2)) * 100)}


def deep_trial(job: tuple) -> dict:
    model, params, data, threads = job
    import torch

    torch.set_num_threads(threads)
    from pvforecast import deep

    generator = deep.seed_everything(0)
    X_seq, y_seq, w_seq = deep.sequences(data["X_train"], data["y_train"], data["w_train"], deep.SEQUENCE_LENGTH)
    loader = deep.loader_for(X_seq, y_seq, w_seq, generator)
    context = data["X_train"][-(deep.SEQUENCE_LENGTH - 1):]
    inputs = deep.windows(context, data["X_val"], deep.SEQUENCE_LENGTH)
    mask, target = data["mask"], data["y_val"]

    def validation(net) -> float:
        pred = np.clip(deep._predict(net, inputs), 0, 1)
        return float(np.mean((pred[mask] - target[mask]) ** 2))

    net = deep.build_model(model, data["X_train"].shape[1], params["width"])
    started = time.perf_counter()
    net, best_epoch = deep.train(net, loader, deep.EPOCHS, params["learning_rate"], validation, deep.PATIENCE)
    return {"model": model, **params, "epochs": int(best_epoch),
            "val_nrmse": float(np.sqrt(validation(net)) * 100),
            "seconds": round(time.perf_counter() - started, 1)}


def main() -> None:
    from pvforecast.cli import add_dataset_arguments, results_path

    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--train-months", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--models", nargs="*", default=list(TREE_MODELS + DEEP_MODELS))
    args = parser.parse_args()

    data = split(args)
    print(data["period"], flush=True)
    rows = []

    for model in [m for m in args.models if m in TREE_MODELS]:
        for values in itertools.product(*TREE_GRID.values()):
            params = dict(zip(TREE_GRID, values))
            rows.append(tree_trial(model, params, data, args.workers * args.threads))
            print(rows[-1], flush=True)

    jobs = [
        (model, dict(zip(DEEP_GRID, values)), data, args.threads)
        for model in [m for m in args.models if m in DEEP_MODELS]
        for values in itertools.product(*DEEP_GRID.values())
    ]
    jobs.sort(key=lambda job: -job[1]["width"])
    with Pool(args.workers) as pool:
        for row in pool.imap_unordered(deep_trial, jobs):
            rows.append(row)
            print(row, flush=True)

    grid = pd.DataFrame(rows)
    out_dir = results_path(args, "tuning", "grid.csv").parent
    grid.to_csv(out_dir / "grid.csv", index=False)

    selected = {}
    for model, group in grid.groupby("model"):
        best = group.sort_values("val_nrmse").iloc[0].to_dict()
        keys = ["max_depth", "learning_rate", "n_estimators"] if model in TREE_MODELS else ["width", "learning_rate", "epochs"]
        selected[model] = {k: (int(best[k]) if k != "learning_rate" else float(best[k])) for k in keys}
        selected[model]["val_nrmse"] = round(float(best["val_nrmse"]), 3)
    selected["_protocol"] = {"split": data["period"], "weather": args.weather, "feature_set": args.feature_set}
    (out_dir / "selected.json").write_text(json.dumps(selected, indent=2))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
