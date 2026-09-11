"""Gain-based feature importance of the gradient-boosted forecaster."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import TARGET, add_recency_weights, set_seed
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    prepare,
    results_path,
    write_table,
)
from pvforecast.features import DAYTIME
from pvforecast.models import XGBOOST_PARAMS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--top", type=int, default=0, help="report only the N strongest features")
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    from xgboost import XGBRegressor

    df, site = prepare(args)
    columns = feature_columns(args)

    daytime = add_recency_weights(df[df[DAYTIME] == 1])
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        daytime[columns].fillna(0.0),
        daytime[TARGET],
        sample_weight=daytime["sample_weight"],
    )

    booster = model.get_booster()
    gains = booster.get_score(importance_type="gain")
    total = sum(gains.values())

    table = pd.DataFrame(
        {
            "feature": columns,
            "gain": [gains.get(c, 0.0) for c in columns],
        }
    )
    table["gain_pct"] = (table["gain"] / total * 100.0).round(2)
    table = table.sort_values("gain_pct", ascending=False).reset_index(drop=True)
    if args.top:
        table = table.head(args.top)

    write_table(
        table,
        results_path(args, "tables", f"importance_{args.site}_{args.feature_set}.csv"),
        f"XGBoost gain importance - {args.site}, {len(columns)} features",
    )


if __name__ == "__main__":
    main()
