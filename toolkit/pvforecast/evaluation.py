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
    design_matrix,
    row_validity,
    scored_mask,
)

log = logging.getLogger(__name__)

METRIC_NAMES = ("mae", "rmse", "mape", "nrmse", "r2")
MAPE_THRESHOLD = 0.01


def nan_metrics() -> dict[str, float]:
    return {name: float("nan") for name in METRIC_NAMES}


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    capacity_kwp: float = 50.0,
) -> dict[str, float]:
    """Scores over every supplied hour; MAPE only where the observed capacity factor exceeds 1%."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return nan_metrics()

    observed = y_true[mask]
    predicted = np.clip(y_pred[mask], 0.0, 1.0)
    error = observed - predicted

    rmse = float(np.sqrt(np.mean(error**2)) * capacity_kwp)
    residual = float(np.sum(error**2))
    variance = float(np.sum((observed - observed.mean()) ** 2))
    producing = observed > MAPE_THRESHOLD

    return {
        "mae": float(np.mean(np.abs(error)) * capacity_kwp),
        "rmse": rmse,
        "mape": float(np.mean(np.abs(error[producing] / observed[producing])) * 100.0)
        if producing.any()
        else float("nan"),
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
    timestamps: Optional[pd.DatetimeIndex] = field(default=None, repr=False)


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

    def predictions(self) -> pd.DataFrame:
        parts = [
            pd.DataFrame(
                {
                    "timestamp": it.timestamps,
                    "test_month": it.test_month,
                    "y_true": it.y_true,
                    "y_pred": it.y_pred,
                }
            )
            for it in self.iterations
            if it.y_pred is not None and it.timestamps is not None
        ]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def mean(self) -> dict[str, float]:
        table = self.frame()
        if table.empty:
            return nan_metrics()
        return {name: float(table[name].mean()) for name in METRIC_NAMES}

    def save(self, path: str | Path, predictions: bool = False) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame().to_csv(path, index=False)
        if predictions:
            self.predictions().to_csv(
                path.with_name(path.stem + "_predictions.csv.gz"), index=False, compression="gzip"
            )
        return path


def _split_months(df: pd.DataFrame, min_train_months: int) -> list[pd.Period]:
    months = sorted(df.index.to_period("M").unique())
    if len(months) < min_train_months + 1:
        raise ValueError(
            f"need at least {min_train_months + 1} months of data, got {len(months)}"
        )
    return months[min_train_months:]


def training_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    decay_per_month: float,
    cutoff_months: int,
) -> pd.DataFrame:
    """Recency-weighted training rows; rows with a missing target or input get zero weight."""
    weighted = add_recency_weights(df, decay_per_month, cutoff_months)
    weighted[WEIGHT] = weighted[WEIGHT].where(row_validity(weighted, feature_cols), 0.0)
    return weighted


def _fit_predict(
    model_fn: Callable,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, float]:
    X_train = design_matrix(train_df, feature_cols)
    y_train = train_df[TARGET].fillna(0.0).to_numpy(dtype=np.float32)
    w_train = train_df[WEIGHT].to_numpy(dtype=np.float32)

    started = time.perf_counter()
    predict_fn = model_fn(X_train, y_train, w_train)
    predictions = np.asarray(
        predict_fn(design_matrix(test_df, feature_cols), test_df.index), dtype=np.float64
    )
    return predictions, round(time.perf_counter() - started, 3)


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
    score_cols: list[str] | None = None,
    save_predictions: bool = False,
    mask: pd.Series | None = None,
) -> Evaluation:
    """Train on months 1..N, predict month N+1, slide by one month.

    Hours are scored when they are daytime and the 24-hour input window behind them is complete
    for ``score_cols`` (defaults to the full feature set), so every model is scored on the same hours.
    A precomputed boolean ``mask`` aligned with ``df`` overrides that rule.
    """
    evaluation = Evaluation(name=name)
    test_months = _split_months(df, min_train_months)
    if mask is not None:
        scored = mask.reindex(df.index).fillna(False).astype(bool)
    else:
        scored = pd.Series(scored_mask(df, score_cols or FULL_FEATURES), index=df.index)

    for position, period in enumerate(test_months, start=1):
        test_start = period.to_timestamp()
        test_end = (period + 1).to_timestamp()

        train_df = training_frame(
            df[df.index < test_start], feature_cols, decay_per_month, cutoff_months
        )
        test_df = df[(df.index >= test_start) & (df.index < test_end)]
        if train_df.empty or test_df.empty:
            log.warning("%s [%s]: empty train or test slice, skipped", name, period)
            continue

        mask = scored.loc[test_df.index].to_numpy()
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
                    train_rows=int((train_df[WEIGHT] > 0).sum()),
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
                train_rows=int((train_df[WEIGHT] > 0).sum()),
                test_rows=int(mask.sum()),
                train_time_sec=elapsed,
                scores=scores,
                y_true=y_true[mask],
                y_pred=y_pred[mask],
                timestamps=test_df.index[mask],
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
        evaluation.save(Path(results_dir) / f"{name}_walk_forward.csv", predictions=save_predictions)

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
    test_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Fit once on ``train_df`` and score ``test_df`` on ``test_mask`` (daytime, complete inputs)."""
    train_df = training_frame(train_df, feature_cols, decay_per_month, cutoff_months)
    mask = test_mask if test_mask is not None else scored_mask(test_df, feature_cols)
    y_true = test_df[TARGET].to_numpy(dtype=np.float64)

    try:
        y_pred, elapsed = _fit_predict(model_fn, train_df, test_df, feature_cols)
    except Exception as error:
        log.error("%s: %s: %s", name, type(error).__name__, error)
        return {**nan_metrics(), "train_rows": int((train_df[WEIGHT] > 0).sum()), "test_rows": int(mask.sum())}

    scores = metrics(y_true[mask], y_pred[mask], capacity_kwp)
    log.info(
        "%s: nRMSE=%.2f%%  MAE=%.2f kW  R2=%.3f  train=%d rows  (%.1fs)",
        name,
        scores["nrmse"],
        scores["mae"],
        scores["r2"],
        int((train_df[WEIGHT] > 0).sum()),
        elapsed,
    )
    return {
        **scores,
        "train_rows": int((train_df[WEIGHT] > 0).sum()),
        "test_rows": int(mask.sum()),
        "train_time_sec": elapsed,
    }


def comparison(
    results: dict[str, Evaluation],
    reference: str = "vanilla_lstm",
) -> pd.DataFrame:
    """Mean scores per model and a Wilcoxon signed-rank test against ``reference``, paired by month."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None

    baseline = None
    if reference in results:
        baseline = results[reference].frame().set_index("test_month")["nrmse"]

    rows = []
    for name, evaluation in results.items():
        row = {"model": name, **{k: round(v, 4) for k, v in evaluation.mean().items()}}

        p_value = float("nan")
        if wilcoxon is not None and baseline is not None and name != reference:
            current = evaluation.frame().set_index("test_month")["nrmse"]
            paired = pd.concat([baseline, current], axis=1, join="inner").dropna()
            if len(paired) >= 5:
                _, p_value = wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1])
        row[f"wilcoxon_p_vs_{reference}"] = round(p_value, 4) if np.isfinite(p_value) else None
        rows.append(row)

    table = pd.DataFrame(rows)
    table["rank"] = table["nrmse"].rank(method="min").astype("Int64")
    return table.sort_values("rank").reset_index(drop=True)
