"""Ensemble combinations of several forecasters under the same walk-forward split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import comparison, get_model, set_seed, walk_forward
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    prepare,
    results_path,
    write_table,
)
from pvforecast.ensemble import average, inverse_error_weighted, ridge_stack
from pvforecast.models import MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument(
        "--members",
        nargs="*",
        default=["xgboost", "lightgbm", "transformer_warm"],
        choices=list(MODEL_NAMES),
    )
    parser.add_argument("--min-train-months", type=int, default=6)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    df, site = prepare(args)
    columns = feature_columns(args)
    output = results_path(args, "ensemble")
    output.mkdir(parents=True, exist_ok=True)

    members = {
        name: walk_forward(
            df,
            get_model(name),
            name,
            feature_cols=columns,
            min_train_months=args.min_train_months,
            capacity_kwp=site.capacity_kwp,
            results_dir=output,
        )
        for name in args.members
    }

    weighted, weights = inverse_error_weighted(members, site.capacity_kwp)
    combined = {
        **members,
        "ensemble_average": average(members, site.capacity_kwp),
        "ensemble_weighted": weighted,
        "ensemble_stack": ridge_stack(members, site.capacity_kwp),
    }
    for name, evaluation in combined.items():
        if name not in members:
            evaluation.save(output / f"{name}_walk_forward.csv")

    print("\ninverse-error weights: " + ", ".join(f"{k}={v:.3f}" for k, v in weights.items()))
    write_table(
        comparison(combined, reference=args.members[0]),
        results_path(args, "tables", f"ensemble_{args.site}.csv"),
        f"Ensemble evaluation - {args.site}",
    )


if __name__ == "__main__":
    main()
