"""Feature-group ablation: contribution of each group to forecast accuracy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import FEATURE_GROUPS, get_model, set_seed, walk_forward
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    prepare,
    results_path,
    write_table,
)
from pvforecast.models import MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--model", default="xgboost", choices=list(MODEL_NAMES))
    parser.add_argument("--groups", nargs="*", default=list(FEATURE_GROUPS), choices=list(FEATURE_GROUPS))
    parser.add_argument("--min-train-months", type=int, default=6)
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    df, site = prepare(args)
    columns = feature_columns(args)
    output = results_path(args, "ablation")
    output.mkdir(parents=True, exist_ok=True)

    reference = walk_forward(
        df,
        get_model(args.model),
        f"{args.model}_all_features",
        feature_cols=columns,
        min_train_months=args.min_train_months,
        capacity_kwp=site.capacity_kwp,
        results_dir=output,
    ).mean()["nrmse"]

    rows = [
        {
            "ablated_group": "none",
            "n_features": len(columns),
            "nrmse": round(reference, 4),
            "delta_nrmse": 0.0,
        }
    ]

    for group in args.groups:
        removed = [c for c in FEATURE_GROUPS[group] if c in columns]
        if not removed:
            continue

        ablated = df.copy()
        ablated[removed] = 0.0
        score = walk_forward(
            ablated,
            get_model(args.model),
            f"{args.model}_no_{group}",
            feature_cols=columns,
            min_train_months=args.min_train_months,
            capacity_kwp=site.capacity_kwp,
            results_dir=output,
        ).mean()["nrmse"]

        rows.append(
            {
                "ablated_group": group,
                "n_features": len(removed),
                "nrmse": round(score, 4),
                "delta_nrmse": round(score - reference, 4),
            }
        )

    table = pd.DataFrame(rows).sort_values("delta_nrmse", ascending=False)
    write_table(
        table,
        results_path(args, "tables", f"ablation_{args.site}_{args.model}.csv"),
        f"Feature-group ablation - {args.model} on {args.site}",
    )


if __name__ == "__main__":
    main()
