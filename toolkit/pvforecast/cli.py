"""Shared command-line plumbing for the toolkit scripts."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config import SITES, Site
from .dataset import load_site, restrict
from .features import FEATURE_SETS, build_features
from .weather import DEFAULT_NWP_DIR, WEATHER_SOURCES, weather_inputs

DEFAULT_DATA_ROOT = "."


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


def add_dataset_arguments(
    parser: argparse.ArgumentParser,
    site_default: str = "site_a",
) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--site", default=site_default, choices=sorted(SITES))
    parser.add_argument("--feature-set", default="full", choices=sorted(FEATURE_SETS))
    parser.add_argument("--weather", default="nwp", choices=WEATHER_SOURCES,
                        help="nwp: forecasts issued 24 h ahead; reanalysis: retrospective reference")
    parser.add_argument("--nwp-dir", default=str(DEFAULT_NWP_DIR))
    parser.add_argument("--start", default=None, help="first month, YYYY-MM")
    parser.add_argument("--end", default=None, help="last month, YYYY-MM")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def prepare(args: argparse.Namespace, site_key: str | None = None) -> tuple[pd.DataFrame, Site]:
    key = site_key or args.site
    df, site = load_site(args.data_root, key)
    if args.start or args.end:
        df = restrict(df, args.start, args.end)
    weather = weather_inputs(df, key, args.weather, args.nwp_dir)
    return build_features(df, site, weather), site


def feature_columns(args: argparse.Namespace) -> list[str]:
    return FEATURE_SETS[args.feature_set]


def results_path(args: argparse.Namespace, *parts: str) -> Path:
    path = Path(args.results_dir).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_table(table: pd.DataFrame, path: Path, title: str) -> None:
    table.to_csv(path, index=False)
    print(f"\n{title}")
    print(table.to_string(index=False))
    print(f"\nsaved -> {path}")
