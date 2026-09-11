"""Model ranking at two independent sites over an aligned calendar window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pvforecast import comparison, get_model, load_site, restrict, set_seed, walk_forward
from pvforecast.cli import (
    add_dataset_arguments,
    configure_logging,
    feature_columns,
    results_path,
    write_table,
)
from pvforecast.dataset import common_window
from pvforecast.features import build_features
from pvforecast.models import DEFAULT_MODELS, MODEL_NAMES


def evaluate_site(args, site_key: str, columns: list[str], output: Path) -> pd.DataFrame:
    frame, site = load_site(args.data_root, site_key)
    frame = restrict(frame, args.start, args.end)
    features = build_features(frame, site)

    results = {
        name: walk_forward(
            features,
            get_model(name),
            f"{site_key}_{name}",
            feature_cols=columns,
            min_train_months=args.min_train_months,
            capacity_kwp=site.capacity_kwp,
            results_dir=output,
        )
        for name in args.models
    }
    table = comparison(results, reference=args.reference)
    table["model"] = table["model"].str.replace(f"{site_key}_", "", regex=False)
    return table[["model", "nrmse", "mae", "r2", "rank"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument("--site-a", default="site_a")
    parser.add_argument("--site-b", default="site_b")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS), choices=list(MODEL_NAMES))
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--reference", default="site_a_vanilla_lstm")
    args = parser.parse_args()

    configure_logging()
    set_seed(args.seed)

    if args.start is None or args.end is None:
        frame_a, _ = load_site(args.data_root, args.site_a)
        frame_b, _ = load_site(args.data_root, args.site_b)
        start, end = common_window(frame_a, frame_b)
        args.start = args.start or start
        args.end = args.end or end
    print(f"aligned window: {args.start} .. {args.end}")

    columns = feature_columns(args)
    output = results_path(args, "cross_site")
    output.mkdir(parents=True, exist_ok=True)

    table_a = evaluate_site(args, args.site_a, columns, output)
    table_b = evaluate_site(args, args.site_b, columns, output)

    merged = table_a.merge(table_b, on="model", suffixes=(f"_{args.site_a}", f"_{args.site_b}"))
    merged = merged.sort_values(f"nrmse_{args.site_a}")

    from scipy.stats import spearmanr

    rho, p_value = spearmanr(
        merged[f"rank_{args.site_a}"], merged[f"rank_{args.site_b}"]
    )

    write_table(
        merged,
        results_path(args, "tables", f"cross_site_{args.start}_{args.end}.csv"),
        f"Cross-site comparison - {args.site_a} vs {args.site_b}, {args.start} to {args.end}",
    )
    print(f"\nSpearman rank correlation: rho={rho:.3f}  p={p_value:.4f}")


if __name__ == "__main__":
    main()
