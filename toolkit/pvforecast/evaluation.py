"""Error metrics, rolling-origin evaluation and fixed-holdout evaluation."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .features import (
    FULL_FEATURES,
    TARGET,
    WEIGHT,
    add_recency_weights,
    daytime_mask,
    design_matrix,
)

log = logging.getLogger(__name__)

METRIC_NAMES = ("mae", "rmse", "mape", "nrmse", "r2")
DAYTIME_THRESHOLD = 0.01


def nan_metrics() -> dict[str, float]:
    return {name: float("nan") for name in METRIC_NAMES}


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_kwp: float = 50.0,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = (y_true > DAYTIME_THRESHOLD) & np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return nan_metrics()

    observed = y_true[mask]
    predicted = np.clip(y_pred[mask], 0.0, 1.0)
    error = observed - predicted

    rmse = float(np.sqrt(np.mean(error**2)) * capacity_kwp)
    residual = float(np.sum(error**2))
    variance = float(np.sum((observed - observed.mean()) ** 2))

    return {
        "mae": float(np.mean(np.abs(error)) * capacity_kwp),
        "rmse": rmse,
        "mape": float(np.mean(np.abs(error / observed)) * 100.0),
        "nrmse": rmse / capacity_kwp * 100.0,
        "r2": 1.0 - residual / variance if variance > 1e-10 else float("nan"),
    }


@dataclass
class Iteration:
    index: int
    test_month: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_rows: int
    test_rows: int
    train_time_sec: float
    scores: dict[str, float]
    y_true: Optional[np.ndarray] = field(default=None, repr=False)
    y_pred: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Evaluation:
    name: str
    iterations: list[Iteration] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "iteration": it.index,
                    "test_month": it.test_month,
                    "train_start": it.train_start,
                    "train_end": it.train_end,
                    "train_rows": it.train_rows,
                    "test_rows": it.test_rows,
                    "train_time_sec": it.train_time_sec,
                    **it.scores,
                }
                for it in self.iterations
            ]
        )

    def mean(self) -> dict[str, float]:
        table = self.frame()
        if table.empty:
            return nan_metrics()
        return {name: float(table[name].mean()) for name in METRIC_NAMES}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame().to_csv(path, index=False)
        return path


def _split_months(df: pd.DataFrame, min_train_months: int) -> list[pd.Period]:
    months = sorted(df.index.to_period("M").unique())
    if len(months) < min_train_months + 1:
        raise ValueError(
            f"need at least {min_train_months + 1} months of data, got {len(months)}"
        )
    return months[min_train_months:]


def _fit_predict(
    model_fn: Callable,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, float]:
    X_train = design_matrix(train_df, feature_cols)
    y_train = train_df[TARGET].fillna(0.0).to_numpy(dtype=np.float32)
    w_train = train_df[WEIGHT].to_numpy(dtype=np.float32)

    started = time.time()
    predict_fn = model_fn(X_train, y_train, w_train)
    predictions = np.asarray(
        predict_fn(design_matrix(test_df, feature_cols), test_df.index), dtype=np.float64
    )
    return predictions, round(time.time() - started, 2)


def walk_forward(
    df: pd.DataFrame,
    model_fn: Callable,
    name: str,
    feature_cols: list[str] = FULL_FEATURES,
    min_train_months: int = 6,
    capacity_kwp: float = 50.0,
    decay_per_month: float = 0.92,
    cutoff_months: int = 18,
    results_dir: str | Path | None = None,
) -> Evaluation:
    evaluation = Evaluation(name=name)
    test_months = _split_months(df, min_train_months)

    for position, period in enumerate(test_months, start=1):
        test_start = period.to_timestamp()
        test_end = (period + 1).to_timestamp()

        train_df = add_recency_weights(
            df[df.index < test_start], decay_per_month, cutoff_months
        )
        test_df = df[(df.index >= test_start) & (df.index < test_end)]
        if train_df.empty or test_df.empty:
            log.warning("%s [%s]: empty train or test slice, skipped", name, period)
            continue

        mask = daytime_mask(test_df)
        y_true = test_df[TARGET].to_numpy(dtype=np.float64)

        try:
            y_pred, elapsed = _fit_predict(model_fn, train_df, test_df, feature_cols)
        except Exception as error:
            log.error("%s [%s]: %s: %s", name, period, type(error).__name__, error)
            evaluation.iterations.append(
                Iteration(
                    index=position,
                    test_month=str(period),
                    train_start=train_df.index.min(),
                    train_end=train_df.index.max(),
                    train_rows=len(train_df),
                    test_rows=int(mask.sum()),
                    train_time_sec=0.0,
                    scores=nan_metrics(),
                )
            )
            continue

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"{name}: model returned {len(y_pred)} predictions for {len(y_true)} rows"
            )

        scores = metrics(y_true[mask], y_pred[mask], capacity_kwp)
        evaluation.iterations.append(
            Iteration(
                index=position,
                test_month=str(period),
                train_start=train_df.index.min(),
                train_end=train_df.index.max(),
                train_rows=len(train_df),
                test_rows=int(mask.sum()),
                train_time_sec=elapsed,
                scores=scores,
                y_true=y_true[mask],
                y_pred=y_pred[mask],
            )
        )
        log.info(
            "%s [%d/%d] %s: nRMSE=%.2f%%  MAE=%.2f kW  R2=%.3f  (%.1fs)",
            name,
            position,
            len(test_months),
            period,
            scores["nrmse"],
            scores["mae"],
            scores["r2"],
            elapsed,
        )

    if results_dir is not None:
        evaluation.save(Path(results_dir) / f"{name}_walk_forward.csv")

    summary = evaluation.mean()
    log.info(
        "%s mean: nRMSE=%.2f%%  MAE=%.2f kW  R2=%.3f",
        name,
        summary["nrmse"],
        summary["mae"],
        summary["r2"],
    )
    return evaluation


def holdout(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_fn: Callable,
    name: str,
    feature_cols: list[str] = FULL_FEATURES,
    capacity_kwp: float = 50.0,
    decay_per_month: float = 0.92,
    cutoff_months: int = 18,
) -> dict[str, float]:
    if WEIGHT not in train_df.columns:
        train_df = add_recency_weights(train_df, decay_per_month, cutoff_months)

    mask = daytime_mask(test_df)
    y_true = test_df[TARGET].to_numpy(dtype=np.float64)

    try:
        y_pred, elapsed = _fit_predict(model_fn, train_df, test_df, feature_cols)
    except Exception as error:
        log.error("%s: %s: %s", name, type(error).__name__, error)
        return {**nan_metrics(), "train_rows": len(train_df), "test_rows": int(mask.sum())}

    scores = metrics(y_true[mask], y_pred[mask], capacity_kwp)
    log.info(
        "%s: nRMSE=%.2f%%  MAE=%.2f kW  R2=%.3f  train=%d rows  (%.1fs)",
        name,
        scores["nrmse"],
        scores["mae"],
        scores["r2"],
        len(train_df),
        elapsed,
    )
    return {
        **scores,
        "train_rows": len(train_df),
        "test_rows": int(mask.sum()),
        "train_time_sec": elapsed,
    }


def comparison(
    results: dict[str, Evaluation],
    reference: str = "vanilla_lstm",
) -> pd.DataFrame:
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    baseline = None
    if reference in results:
        baseline = results[reference].frame().get("nrmse")

    rows = []
    for name, evaluation in results.items():
        row = {"model": name, **{k: round(v, 4) for k, v in evaluation.mean().items()}}

        p_value = float("nan")
        if wilcoxon is not None and baseline is not None and name != reference:
            current = evaluation.frame().get("nrmse")
            if current is not None:
                n = min(len(baseline), len(current))
                paired = ~(baseline[:n].isna() | current[:n].isna())
                if paired.sum() >= 5:
                    _, p_value = wilcoxon(baseline[:n][paired], current[:n][paired])
        row[f"wilcoxon_p_vs_{reference}"] = round(p_value, 4) if np.isfinite(p_value) else None
        rows.append(row)

    table = pd.DataFrame(rows)
    table["rank"] = table["nrmse"].rank(method="min").astype("Int64")
    return table.sort_values("rank").reset_index(drop=True)
