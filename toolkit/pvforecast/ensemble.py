"""Combination of several evaluated forecasters into ensemble predictions."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .evaluation import Evaluation, Iteration, metrics


def _aligned(members: dict[str, Evaluation]) -> list[tuple[Iteration, dict[str, np.ndarray]]]:
    names = list(members)
    reference = members[names[0]]
    indexed = {
        name: {it.test_month: it for it in evaluation.iterations}
        for name, evaluation in members.items()
    }

    aligned = []
    for iteration in reference.iterations:
        month = iteration.test_month
        available = {
            name: indexed[name][month].y_pred
            for name in names
            if month in indexed[name] and indexed[name][month].y_pred is not None
        }
        if len(available) != len(names) or iteration.y_true is None:
            continue
        if any(len(p) != len(iteration.y_true) for p in available.values()):
            continue
        aligned.append((iteration, available))
    return aligned


def _combine(
    members: dict[str, Evaluation],
    name: str,
    blend: Callable[[dict[str, np.ndarray]], np.ndarray],
    capacity_kwp: float,
) -> Evaluation:
    ensemble = Evaluation(name=name)
    for iteration, predictions in _aligned(members):
        y_pred = np.clip(blend(predictions), 0.0, 1.0)
        ensemble.iterations.append(
            Iteration(
                index=iteration.index,
                test_month=iteration.test_month,
                train_start=iteration.train_start,
                train_end=iteration.train_end,
                train_rows=iteration.train_rows,
                test_rows=iteration.test_rows,
                train_time_sec=0.0,
                scores=metrics(iteration.y_true, y_pred, capacity_kwp),
                y_true=iteration.y_true,
                y_pred=y_pred,
            )
        )
    return ensemble


def average(members: dict[str, Evaluation], capacity_kwp: float = 50.0) -> Evaluation:
    return _combine(
        members,
        "ensemble_average",
        lambda preds: np.mean(list(preds.values()), axis=0),
        capacity_kwp,
    )


def inverse_error_weighted(
    members: dict[str, Evaluation],
    capacity_kwp: float = 50.0,
) -> tuple[Evaluation, dict[str, float]]:
    raw = {name: 1.0 / max(ev.mean()["nrmse"], 1e-6) for name, ev in members.items()}
    total = sum(raw.values())
    weights = {name: value / total for name, value in raw.items()}

    ensemble = _combine(
        members,
        "ensemble_weighted",
        lambda preds: sum(weights[name] * preds[name] for name in preds),
        capacity_kwp,
    )
    return ensemble, weights


def ridge_stack(
    members: dict[str, Evaluation],
    capacity_kwp: float = 50.0,
    alpha: float = 1.0,
    min_rows: int = 10,
) -> Evaluation:
    from sklearn.linear_model import Ridge

    names = list(members)
    aligned = _aligned(members)
    ensemble = Evaluation(name="ensemble_stack")

    for position in range(1, len(aligned)):
        history = aligned[:position]
        X_meta = np.column_stack(
            [np.concatenate([preds[name] for _, preds in history]) for name in names]
        )
        y_meta = np.concatenate([it.y_true for it, _ in history])
        if len(y_meta) < min_rows:
            continue

        model = Ridge(alpha=alpha).fit(X_meta, y_meta)
        iteration, predictions = aligned[position]
        y_pred = np.clip(
            model.predict(np.column_stack([predictions[name] for name in names])), 0.0, 1.0
        )

        ensemble.iterations.append(
            Iteration(
                index=iteration.index,
                test_month=iteration.test_month,
                train_start=iteration.train_start,
                train_end=iteration.train_end,
                train_rows=iteration.train_rows,
                test_rows=iteration.test_rows,
                train_time_sec=0.0,
                scores=metrics(iteration.y_true, y_pred, capacity_kwp),
                y_true=iteration.y_true,
                y_pred=y_pred,
            )
        )
    return ensemble
