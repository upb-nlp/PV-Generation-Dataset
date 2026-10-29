"""Emit the LaTeX table bodies of the revised manuscript from the report CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["xgboost", "lightgbm", "vanilla_lstm", "cnn_lstm", "attention_lstm", "transformer", "nbeats"]
BASELINES = ["smart_persistence", "naive_persistence", "day_persistence"]
LABELS = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "vanilla_lstm": "Vanilla LSTM", "cnn_lstm": "CNN-LSTM",
    "attention_lstm": "Attention-LSTM", "transformer": "Transformer", "nbeats": "N-BEATS",
    "naive_persistence": "Naive persistence", "smart_persistence": "Smart persistence",
    "day_persistence": "Day persistence", "xgboost_warm": "XGBoost", "lightgbm_warm": "LightGBM",
    "transformer_warm": "Transformer", "ensemble_stack": "Stacking (ridge)",
    "ensemble_weighted": "Inverse-error weighted average", "ensemble_average": "Simple average",
}
SHORT = {"xgboost": "XGB", "lightgbm": "LGBM", "vanilla_lstm": "LSTM", "cnn_lstm": "CNN-L",
         "attention_lstm": "Att-L", "transformer": "Trf", "nbeats": "N-B", "smart_persistence": "Smart",
         "naive_persistence": "Naive", "day_persistence": "Day"}


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}\\%"


def pval(p: float) -> str:
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def benchmark_table(report: Path) -> str:
    nwp = pd.read_csv(report / "benchmark_nwp.csv").set_index("model")
    inv = pd.read_csv(report / "benchmark_inverter_only.csv").set_index("model")
    rea = pd.read_csv(report / "benchmark_reanalysis.csv").set_index("model")
    tests = pd.read_csv(report / "pairwise_nwp.csv")

    def holm_vs(model: str, reference: str) -> float:
        row = tests[((tests.model_a == model) & (tests.model_b == reference)) | ((tests.model_a == reference) & (tests.model_b == model))]
        return float(row["holm_p"].iloc[0]) if len(row) else float("nan")

    lines = []
    order = nwp.loc[[m for m in MODELS if m in nwp.index]].sort_values("nrmse").index.tolist()
    for rank, model in enumerate(order, start=1):
        r = nwp.loc[model]
        lines.append(
            f"{rank} & {LABELS[model]} & {r.nrmse:.2f} $\\pm$ {r.nrmse_sd:.2f}\\% & {r.mae:.2f} & {r.r2:.3f} & "
            f"{r.skill_vs_smart_pct:+.1f}\\% & {pval(holm_vs(model, 'smart_persistence'))} & "
            f"{inv.loc[model, 'nrmse']:.2f}\\% & {rea.loc[model, 'nrmse']:.2f}\\% \\\\"
        )
    lines.append("\\midrule")
    for model in BASELINES:
        r = nwp.loc[model]
        lines.append(
            f"-- & {LABELS[model]} & {r.nrmse:.2f}\\% & {r.mae:.2f} & {r.r2:.3f} & "
            f"{r.skill_vs_smart_pct:+.1f}\\% & {'---' if model == 'smart_persistence' else pval(holm_vs(model, 'smart_persistence'))} & "
            f"{inv.loc[model, 'nrmse']:.2f}\\% & {rea.loc[model, 'nrmse']:.2f}\\% \\\\"
        )
    return "\n".join(lines)


def monthly_table(report: Path) -> str:
    monthly = pd.read_csv(report / "monthly_nwp.csv", index_col=0)
    columns = [c for c in MODELS + BASELINES if c in monthly.columns]
    header = "\\textbf{Month} & " + " & ".join(f"\\textbf{{{SHORT[c]}}}" for c in columns) + " \\\\"
    lines = [header, "\\midrule"]
    for month, row in monthly[columns].iterrows():
        best = row[[c for c in columns]].idxmin()
        cells = [f"\\textbf{{{row[c]:.2f}}}" if c == best else f"{row[c]:.2f}" for c in columns]
        lines.append(f"{month} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("Mean & " + " & ".join(f"{monthly[c].mean():.2f}" for c in columns) + " \\\\")
    return "\n".join(lines)


def pairwise_matrix(report: Path) -> str:
    tests = pd.read_csv(report / "pairwise_nwp.csv")
    order = [c for c in MODELS + BASELINES if c in set(tests.model_a) | set(tests.model_b)]
    lookup = {}
    for _, t in tests.iterrows():
        lookup[(t.model_a, t.model_b)] = (t.mean_diff_a_minus_b, t.holm_p)
        lookup[(t.model_b, t.model_a)] = (-t.mean_diff_a_minus_b, t.holm_p)
    header = " & " + " & ".join(f"\\textbf{{{SHORT[c]}}}" for c in order) + " \\\\"
    lines = [header, "\\midrule"]
    for i, a in enumerate(order):
        cells = []
        for j, b in enumerate(order):
            if i == j:
                cells.append("---")
            elif j > i:
                cells.append(f"{lookup[(a, b)][0]:+.2f}")
            else:
                cells.append(pval(lookup[(a, b)][1]))
        lines.append(f"\\textbf{{{SHORT[a]}}} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def cross_site_table(report: Path) -> str:
    table = pd.read_csv(report / "cross_site.csv").set_index("model")
    models = table.loc[[m for m in MODELS if m in table.index]].sort_values("nrmse_a")
    rank_b = models["nrmse_b"].rank().astype(int)
    lines = []
    for rank, (model, r) in enumerate(models.iterrows(), start=1):
        lines.append(f"{rank} & {LABELS[model]} & {r.nrmse_a:.2f} $\\pm$ {r.nrmse_sd_a:.2f}\\% & "
                     f"{r.nrmse_b:.2f} $\\pm$ {r.nrmse_sd_b:.2f}\\% & {rank_b[model]} \\\\")
    lines.append("\\midrule")
    for model in ("smart_persistence", "naive_persistence"):
        r = table.loc[model]
        lines.append(f"-- & {LABELS[model]} & {r.nrmse_a:.2f}\\% & {r.nrmse_b:.2f}\\% & -- \\\\")
    spearman = json.loads((report / "cross_site_spearman.json").read_text())
    return "\n".join(lines) + f"\n% Spearman rho = {spearman['rho']:.3f}, exact p = {spearman['exact_p']:.4f}"


def window_table(report: Path, windows=(1, 3, 6, 9, 12)) -> str:
    means = pd.read_csv(report / "training_window.csv")
    months = pd.read_csv(report / "training_window_months.csv")
    lines = []
    span = months.dropna(subset=["window_months"]).set_index("window_months")
    lines.append("Training months & " + " & ".join(
        f"{span.loc[w, 'first_month']}--{span.loc[w, 'last_month']}".replace("-0", "-") for w in windows) + " \\\\")
    lines.append("\\midrule")
    pivot = means.pivot(index="model", columns="window_months", values="mean")
    for model in [m for m in MODELS if m in pivot.index]:
        lines.append(f"{LABELS[model]} & " + " & ".join(pct(pivot.loc[model, w]) for w in windows) + " \\\\")
    base = months[months["model"].isin(BASELINES)].set_index("model")["nrmse"] if "model" in months else pd.Series()
    return "\n".join(lines) + "\n% baselines: " + json.dumps({k: round(v, 2) for k, v in base.items()})


def physics_table(report: Path) -> str:
    table = pd.read_csv(report / "physics.csv", index_col=0)
    return "\n".join(
        f"{LABELS[m]} & {r.base25:.2f}\\% & {r.full28:.2f}\\% & ${r.full28 - r.base25:+.2f}$ \\\\" for m, r in table.iterrows()
    )


def warm_table(report: Path) -> str:
    nwp = pd.read_csv(report / "benchmark_nwp.csv").set_index("model")
    warm = pd.read_csv(report / "warm_start.csv").set_index("model")
    lines = []
    for scratch, warmed in (("lightgbm", "lightgbm_warm"), ("xgboost", "xgboost_warm"), ("transformer", "transformer_warm")):
        s, w = nwp.loc[scratch], warm.loc[warmed]
        lines.append(f"{LABELS[scratch]} & {s.nrmse:.2f} $\\pm$ {s.nrmse_sd:.2f}\\% & {w.nrmse:.2f} $\\pm$ {w.nrmse_sd:.2f}\\% & "
                     f"${w.nrmse - s.nrmse:+.2f}$ \\\\")
    return "\n".join(lines)


def ensemble_table(report: Path) -> str:
    table = pd.read_csv(report / "ensembles.csv")
    lines = []
    for _, r in table.iterrows():
        lines.append(f"{LABELS.get(r.model, r.model)} & {r.nrmse_mean:.2f} $\\pm$ {r.nrmse_std:.2f}\\% & {r.mae_mean:.3f} & {r.r2_mean:.3f} \\\\")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="results/revision/report")
    args = parser.parse_args()
    report = Path(args.report)
    sections = {
        "benchmark": benchmark_table, "monthly": monthly_table, "pairwise": pairwise_matrix,
        "cross_site": cross_site_table, "training_window": window_table, "physics": physics_table,
        "warm_start": warm_table, "ensembles": ensemble_table,
    }
    out = report / "latex_tables.tex"
    with out.open("w", encoding="utf-8") as handle:
        for name, build in sections.items():
            try:
                body = build(report)
            except Exception as error:  # a table whose runs are not finished yet
                body = f"% not available: {type(error).__name__}: {error}"
            handle.write(f"% ===== {name}\n{body}\n\n")
            print(f"% ===== {name}\n{body}\n")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
