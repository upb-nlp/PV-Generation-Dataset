"""Rolling-origin comparison of forecasters and persistence baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import comparison, get_baseline, get_model, set_seed, walk_forward
from pvforecast.baselines import BASELINE_NAMES
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    prepare,
    results_path,
    write_table,
)
from pvforecast.models import DEFAULT_MODELS, MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS), choices=list(MODEL_NAMES))
    parser.add_argument("--baselines", nargs="*", default=list(BASELINE_NAMES), choices=list(BASELINE_NAMES))
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--reference", default="vanilla_lstm")
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    df, site = prepare(args)
    columns = feature_columns(args)
    output = results_path(args, "benchmark")
    output.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in args.baselines:
        results[name] = walk_forward(
            df,
            get_baseline(name, df),
            name,
            feature_cols=columns,
            min_train_months=args.min_train_months,
            capacity_kwp=site.capacity_kwp,
            results_dir=output,
        )

    for name in args.models:
        results[name] = walk_forward(
            df,
            get_model(name),
            name,
            feature_cols=columns,
            min_train_months=args.min_train_months,
            capacity_kwp=site.capacity_kwp,
            results_dir=output,
        )

    table = comparison(results, reference=args.reference)
    write_table(
        table,
        results_path(args, "tables", f"benchmark_{args.site}_{args.feature_set}.csv"),
        f"Walk-forward comparison - {args.site}, {len(columns)} features",
    )


if __name__ == "__main__":
    main()
