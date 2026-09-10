"""Accuracy as a function of the amount of training history, on a fixed holdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import get_baseline, get_model, holdout, months, set_seed
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    prepare,
    results_path,
    write_table,
)
from pvforecast.models import MODEL_NAMES

DEFAULT_WINDOWS = (1, 2, 3, 4, 5, 6, 9, 12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--models", nargs="*", default=["xgboost", "transformer"], choices=list(MODEL_NAMES))
    parser.add_argument("--windows", nargs="*", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--holdout-months", type=int, default=3)
    parser.add_argument("--baseline", default="naive_persistence")
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    df, site = prepare(args)
    columns = feature_columns(args)
    calendar = months(df)

    if len(calendar) <= args.holdout_months:
        raise SystemExit(f"need more than {args.holdout_months} months of data")

    holdout_start = calendar[-args.holdout_months].to_timestamp()
    test_df = df[df.index >= holdout_start]
    history = df[df.index < holdout_start]

    rows = []
    reference = holdout(
        history,
        test_df,
        get_baseline(args.baseline, df),
        args.baseline,
        feature_cols=columns,
        capacity_kwp=site.capacity_kwp,
    )
    rows.append({"model": args.baseline, "window_months": None, **reference})

    for name in args.models:
        for window in sorted(args.windows):
            if window > len(calendar) - args.holdout_months:
                continue
            window_start = calendar[-(args.holdout_months + window)].to_timestamp()
            train_df = history[history.index >= window_start]
            scores = holdout(
                train_df,
                test_df,
                get_model(name),
                f"{name} [{window} mo]",
                feature_cols=columns,
                capacity_kwp=site.capacity_kwp,
            )
            rows.append({"model": name, "window_months": window, **scores})

    write_table(
        pd.DataFrame(rows),
        results_path(args, "tables", f"training_window_{args.site}.csv"),
        f"Training-window sweep - {args.site}, holdout {test_df.index.min().date()} "
        f"to {test_df.index.max().date()}",
    )


if __name__ == "__main__":
    main()
