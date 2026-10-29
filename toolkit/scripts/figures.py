"""Regenerate the manuscript figures, as vector PDF, from the revision result tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SURFACE = "#ffffff"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e5e5e2"
NEUTRAL = "#8a8a85"

BOOSTING = "#eb6834"
DEEP = "#2a78d6"
SERIES = {
    "xgboost": "#eb6834",
    "lightgbm": "#c2410c",
    "transformer": "#1baf7a",
    "cnn_lstm": "#4a3aa7",
    "vanilla_lstm": "#2a78d6",
    "attention_lstm": "#0e7490",
    "nbeats": "#a21caf",
}
LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "transformer": "Transformer",
    "cnn_lstm": "CNN-LSTM",
    "vanilla_lstm": "Vanilla LSTM",
    "attention_lstm": "Attention-LSTM",
    "nbeats": "N-BEATS",
    "naive_persistence": "Naive persistence",
    "day_persistence": "Day persistence",
    "smart_persistence": "Smart persistence",
}
MODELS = ["xgboost", "lightgbm", "vanilla_lstm", "cnn_lstm", "attention_lstm", "transformer", "nbeats"]
BASELINES = {"naive_persistence", "day_persistence", "smart_persistence"}

STYLE = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": "#9a9a95",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
}


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path.with_suffix('.pdf').name}")


def small_window_curve(report: Path, out: Path) -> None:
    curves = pd.read_csv(report / "training_window.csv")
    months = pd.read_csv(report / "training_window_months.csv")
    references = months[months["model"].isin(BASELINES)] if "model" in months else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name in MODELS:
        frame = curves[curves["model"] == name].sort_values("window_months")
        if frame.empty:
            continue
        ax.plot(frame["window_months"], frame["mean"], marker="o", markersize=4, linewidth=1.8,
                color=SERIES[name], label=LABELS[name])
        ax.fill_between(frame["window_months"], frame["mean"] - frame["std"], frame["mean"] + frame["std"],
                        color=SERIES[name], alpha=0.12, linewidth=0)

    for _, row in references.iterrows():
        if row["model"] == "day_persistence":
            continue
        style = "--" if row["model"] == "smart_persistence" else ":"
        ax.axhline(row["nrmse"], linestyle=style, linewidth=1.3, color=INK)
        ax.annotate(f"{LABELS[row['model']].lower()}  {row['nrmse']:.1f}%", (12.4, row["nrmse"]),
                    xytext=(0, 3), textcoords="offset points", ha="right", fontsize=7.5, color=INK)

    ax.set_xlabel("Training window (months immediately before the holdout)")
    ax.set_ylabel("nRMSE (% of rated capacity)")
    ax.set_title("Accuracy versus available history, Site A\nfixed holdout April to June 2026, mean $\\pm$ s.d. over three seeds")
    ax.set_xticks(sorted(curves["window_months"].unique()))
    ax.grid(axis="y", alpha=0.7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, out / "small_window_curve")


def feature_importance(report: Path, out: Path, top: int = 15) -> None:
    table = pd.read_csv(report / "importance.csv").head(top)
    order = table.iloc[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(order["feature"], order["gain_pct"], color=DEEP, height=0.68)
    for name, value in zip(order["feature"], order["gain_pct"]):
        ax.annotate(f"{value:.1f}%", (value, name), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("XGBoost feature importance, Site A\nweather from forecasts issued 24 h ahead")
    ax.set_xlim(0, table["gain_pct"].max() * 1.16)
    ax.grid(axis="x", alpha=0.7)
    ax.tick_params(axis="y", length=0)
    save(fig, out / "feature_importance")


def operational_horizon(operational: Path, out: Path) -> None:
    variants = [
        ("nwp_at_origin", "Forecast weather issued before the origin", BOOSTING, "-"),
        ("climatological_weather", "Climatological weather (no weather information)", NEUTRAL, "--"),
        ("reanalysis", "Reanalysis (retrospective reference, not available)", DEEP, ":"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6))

    ax = axes[0]
    table = pd.read_csv(operational / "site_a_by_horizon_day.csv")
    for name, label, color, style in variants:
        frame = table[table["variant"] == name].sort_values("horizon_day")
        ax.plot(frame["horizon_day"], frame["nrmse"], linestyle=style, marker="o", markersize=3.5,
                linewidth=1.8, color=color, label=label)
    ax.axvspan(7.5, 14.5, color=GRID, alpha=0.6, linewidth=0)
    ax.annotate("beyond the 7-day\nforecast archive", (11, 0.5), xycoords=("data", "axes fraction"),
                ha="center", va="center", fontsize=7.5, color=MUTED)
    origins = pd.read_csv(operational / "site_a_per_origin.csv", usecols=["origin"])["origin"].nunique()
    ax.set_title(f"(a) Site A, {origins} daily origins")
    ax.set_xlabel("Horizon day")
    ax.set_ylabel("nRMSE (% of rated capacity)")
    ax.set_xticks(range(1, 15))
    ax.set_xlim(0.5, 14.5)
    ax.grid(axis="y", alpha=0.7)

    ax = axes[1]
    frame = table[(table["variant"] == "difference_nwp_minus_climatological") & (table["horizon_day"] <= 7)]
    ax.plot(frame["horizon_day"], frame["nrmse"], marker="o", markersize=3.5, linewidth=1.8, color=BOOSTING)
    ax.fill_between(frame["horizon_day"], frame["ci_low"], frame["ci_high"], color=BOOSTING, alpha=0.15, linewidth=0)
    ax.axhline(0.0, color=INK, linewidth=1)
    ax.set_title("(b) Gain from forecast weather, 95% CI")
    ax.set_xlabel("Horizon day")
    ax.set_ylabel("$\\Delta$ nRMSE vs climatological weather (points)")
    ax.set_xticks(range(1, 8))
    ax.grid(axis="y", alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, out / "operational_horizon")


def cross_site(report: Path, out: Path) -> None:
    table = pd.read_csv(report / "cross_site.csv")
    table = table[table["model"].isin(MODELS + ["smart_persistence", "naive_persistence"])].sort_values("nrmse_a")
    labels = [LABELS.get(m, m) for m in table["model"]]
    y = np.arange(len(table))[::-1]
    height = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(y + height / 2, table["nrmse_a"], height=height, color=DEEP, label="Site A")
    ax.barh(y - height / 2, table["nrmse_b"], height=height, color=BOOSTING, label="Site B")
    for position, value in zip(y + height / 2, table["nrmse_a"]):
        ax.annotate(f"{value:.1f}", (value, position), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    for position, value in zip(y - height / 2, table["nrmse_b"]):
        ax.annotate(f"{value:.1f}", (value, position), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("nRMSE (%)")
    ax.set_title("Two sites over the aligned window\ntest months November 2025 to May 2026")
    ax.set_xlim(0, table[["nrmse_a", "nrmse_b"]].to_numpy().max() * 1.16)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", alpha=0.7)
    ax.tick_params(axis="y", length=0)
    save(fig, out / "r11_cross_site_nrmse")


def rank_correlation(report: Path, out: Path) -> None:
    table = pd.read_csv(report / "cross_site.csv")
    table = table[table["model"].isin(MODELS)].copy()
    table["rank_a"] = table["nrmse_a"].rank().astype(int)
    table["rank_b"] = table["nrmse_b"].rank().astype(int)
    spearman = json.loads((report / "cross_site_spearman.json").read_text())

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    limits = [0.4, len(table) + 0.6]
    ax.plot(limits, limits, linestyle="--", linewidth=1.1, color=NEUTRAL, zorder=1)
    ax.scatter(table["rank_a"], table["rank_b"], s=70, color=DEEP, zorder=3)
    for _, row in table.iterrows():
        ax.annotate(LABELS[row["model"]], (row["rank_a"], row["rank_b"]), xytext=(7, 4),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Rank at Site A")
    ax.set_ylabel("Rank at Site B")
    ax.set_title(f"Rank agreement across sites\nSpearman $\\rho$ = {spearman['rho']:.3f} (exact p = {spearman['exact_p']:.3f})")
    ax.set_aspect("equal")
    ax.grid(alpha=0.5)
    save(fig, out / "r11_rank_correlation")


def monthly(folder: Path, name: str) -> pd.Series:
    files = sorted(folder.glob(f"{name}_s*_walk_forward.csv")) or sorted(folder.glob(f"{name}_walk_forward.csv"))
    return pd.concat([pd.read_csv(f).set_index("test_month")["nrmse"] for f in files], axis=1).mean(axis=1)


def site_walkforward(results: Path, out: Path) -> None:
    sources = {"Site A": results / "benchmark" / "nwp", "Site B": results / "cross_site" / "site_b"}
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for (label, folder), color in zip(sources.items(), (DEEP, BOOSTING)):
        model = monthly(folder, "xgboost")
        smart = monthly(folder, "smart_persistence")
        ax.plot(model.index, model.values, marker="o", markersize=4.5, linewidth=2, color=color, label=f"XGBoost, {label}")
        ax.plot(smart.index, smart.values, marker="s", markersize=3.5, linewidth=1.2, linestyle="--", color=color,
                alpha=0.8, label=f"Smart persistence, {label}")
    ax.set_ylabel("nRMSE (%)")
    ax.set_xlabel("Test month")
    ax.set_title("Month-by-month error of the boosted forecaster against smart persistence")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, None)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.55))
    ax.grid(axis="y", alpha=0.7)
    save(fig, out / "site_walkforward_xgb")


def bias_factor(operational: Path, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True)
    table = pd.read_csv(operational / "site_a_per_origin.csv", parse_dates=["origin"])
    factor = table.groupby("origin")["bias_factor"].first()
    pooled = table.groupby(["origin", "variant"])[["sse", "hours"]].sum()
    nrmse = np.sqrt(pooled["sse"] / pooled["hours"]).unstack() * 100
    effect = nrmse["climatological_weather_recency"] - nrmse["climatological_weather"]
    axes[0].plot(factor.index, factor.values, linewidth=1.6, color=DEEP)
    axes[1].plot(effect.index, effect.values, linewidth=1.2, color=DEEP)

    ax = axes[0]
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color=INK)
    ax.axhline(0.5, linestyle=":", linewidth=1.2, color=INK)
    ax.annotate("1.0: no correction", (0.995, 1.0), xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=7.5)
    ax.annotate("0.5: lower clip bound", (0.995, 0.5), xycoords=("axes fraction", "data"), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=7.5)
    ax.set_ylabel("Recency factor")
    ax.set_ylim(0.35, max(1.45, ax.get_ylim()[1]))
    ax.set_title("Trailing 30-day recency correction at Site A: factor and effect")
    ax.grid(axis="y", alpha=0.7)

    ax = axes[1]
    ax.axhline(0.0, color=INK, linewidth=1)
    ax.set_ylabel("$\\Delta$ nRMSE, days 1-14 (points)")
    ax.set_xlabel("Forecast origin")
    ax.annotate("above zero = the correction makes the forecast worse", (0.98, 0.92), xycoords="axes fraction",
                ha="right", fontsize=7.5, color=MUTED)
    ax.grid(axis="y", alpha=0.7)
    fig.tight_layout()
    save(fig, out / "site_bias_factor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/revision")
    parser.add_argument("--out", default="results/revision/figures")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    plt.rcParams.update(STYLE)
    results = Path(args.results_dir)
    report = results / "report"
    operational = results / "operational"
    out = Path(args.out)
    print("writing figures:")
    jobs = {
        "small_window": lambda: small_window_curve(report, out),
        "importance": lambda: feature_importance(report, out),
        "operational": lambda: operational_horizon(operational, out),
        "cross_site": lambda: cross_site(report, out),
        "rank": lambda: rank_correlation(report, out),
        "monthly": lambda: site_walkforward(results, out),
        "bias": lambda: bias_factor(operational, out),
    }
    for name, job in jobs.items():
        if args.only is None or name in args.only:
            job()


if __name__ == "__main__":
    main()
