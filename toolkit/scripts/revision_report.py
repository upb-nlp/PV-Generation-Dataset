"""Tables and statistics of the revised study, built from the runs of ``revision_runner.py``.

Every number comes from a single canonical run per model, seed and configuration. Seed
averages are taken per test month before any comparison, so each model contributes one score
per month. Pairwise comparisons use the exact Wilcoxon signed-rank test over months with a
Holm correction across all pairs; paired mean differences carry t-based 95% intervals.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = ["xgboost", "lightgbm", "vanilla_lstm", "cnn_lstm", "attention_lstm", "transformer", "nbeats"]
BASELINES = ["naive_persistence", "smart_persistence", "day_persistence"]
LABELS = {
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "vanilla_lstm": "Vanilla LSTM", "cnn_lstm": "CNN-LSTM",
    "attention_lstm": "Attention-LSTM", "transformer": "Transformer", "nbeats": "N-BEATS",
    "naive_persistence": "Naive persistence", "smart_persistence": "Smart persistence",
    "day_persistence": "Day persistence", "xgboost_warm": "XGBoost warm-start",
    "lightgbm_warm": "LightGBM warm-start", "transformer_warm": "Transformer warm-start",
}
METRICS = ["nrmse", "mae", "r2"]


def load_runs(folder: Path, name: str) -> pd.DataFrame:
    """Per-month scores of every seed of one model (or the single run of a baseline)."""
    files = sorted(folder.glob(f"{name}_s*_walk_forward.csv")) or sorted(folder.glob(f"{name}_walk_forward.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        seed = path.stem.split("_s")[-1].split("_")[0] if "_s" in path.stem else "0"
        frame["seed"] = seed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_predictions(folder: Path, name: str, seed: int | None) -> pd.DataFrame:
    stem = f"{name}_s{seed}" if seed is not None else name
    return pd.read_csv(folder / f"{stem}_walk_forward_predictions.csv.gz", parse_dates=["timestamp"])


def summary_table(folder: Path, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, monthly = [], {}
    for name in names:
        runs = load_runs(folder, name)
        if runs.empty:
            continue
        per_seed = runs.groupby("seed")[METRICS + ["train_time_sec"]].mean()
        row = {"model": name, "seeds": len(per_seed), "folds": int(runs.groupby("seed").size().min()),
               "failed_folds": int(runs["nrmse"].isna().sum())}
        for metric in METRICS:
            row[metric] = per_seed[metric].mean()
            row[f"{metric}_sd"] = per_seed[metric].std(ddof=1) if len(per_seed) > 1 else 0.0
        row["train_sec_per_fold_median"] = runs["train_time_sec"].median()
        rows.append(row)
        monthly[name] = runs.groupby("test_month")["nrmse"].mean()
    table = pd.DataFrame(rows)
    smart = table.loc[table["model"] == "smart_persistence", "nrmse"]
    if len(smart):
        table["skill_vs_smart_pct"] = (1 - table["nrmse"] / smart.iloc[0]) * 100
    table["rank"] = table["nrmse"].rank(method="min").astype(int)
    return table.sort_values("nrmse").reset_index(drop=True), pd.DataFrame(monthly)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def pairwise(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(monthly.columns, 2):
        pair = monthly[[a, b]].dropna()
        diff = pair[a] - pair[b]
        n = len(diff)
        p = stats.wilcoxon(pair[a], pair[b], method="exact" if n <= 25 else "auto").pvalue if n >= 5 and diff.abs().sum() > 0 else 1.0
        half = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        rows.append({"model_a": a, "model_b": b, "months": n, "mean_diff_a_minus_b": diff.mean(),
                     "ci_low": diff.mean() - half, "ci_high": diff.mean() + half,
                     "a_better_months": int((diff < 0).sum()), "wilcoxon_p": p})
    table = pd.DataFrame(rows)
    table["holm_p"] = holm(table["wilcoxon_p"].tolist())
    return table


def spearman_exact(rank_a: list[int], rank_b: list[int]) -> tuple[float, float]:
    rho = stats.spearmanr(rank_a, rank_b).statistic
    n = len(rank_a)
    count = total = 0
    base = np.array(rank_a)
    for perm in itertools.permutations(rank_b):
        r = stats.spearmanr(base, perm).statistic
        total += 1
        count += abs(r) >= abs(rho) - 1e-12
    return float(rho), count / total


def ensembles(results: Path, seeds: list[int]) -> pd.DataFrame:
    from pvforecast.ensemble import average, inverse_error_weighted, ridge_stack
    from pvforecast.evaluation import Evaluation, Iteration, metrics

    def evaluation_from(frame: pd.DataFrame, name: str) -> Evaluation:
        ev = Evaluation(name=name)
        for i, (month, group) in enumerate(frame.groupby("test_month"), start=1):
            ev.iterations.append(Iteration(i, month, None, None, 0, len(group), 0.0,
                                           metrics(group["y_true"], group["y_pred"]),
                                           group["y_true"].to_numpy(), group["y_pred"].to_numpy(),
                                           pd.DatetimeIndex(group["timestamp"])))
        return ev

    rows = []
    for seed in seeds:
        members = {
            "xgboost": evaluation_from(load_predictions(results / "benchmark" / "nwp", "xgboost", seed), "xgboost"),
            "lightgbm": evaluation_from(load_predictions(results / "benchmark" / "nwp", "lightgbm", seed), "lightgbm"),
            "transformer_warm": evaluation_from(load_predictions(results / "warm_start" / "nwp", "transformer_warm", seed), "transformer_warm"),
        }
        weighted, history = inverse_error_weighted(members)
        combined = {**members, "ensemble_average": average(members), "ensemble_weighted": weighted,
                    "ensemble_stack": ridge_stack(members)}
        scored_months = {it.test_month for it in combined["ensemble_stack"].iterations}
        for name, ev in combined.items():
            folds = [it for it in ev.iterations if it.test_month in scored_months]
            rows.append({"seed": seed, "model": name, "folds": len(folds),
                         **{m: np.mean([it.scores[m] for it in folds]) for m in METRICS},
                         "last_weights": json.dumps({k: round(v, 3) for k, v in history[-1].items()}) if name == "ensemble_weighted" else ""})
    frame = pd.DataFrame(rows)
    out = frame.groupby("model")[METRICS].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["folds"] = frame.groupby("model")["folds"].first()
    return out.sort_values("nrmse_mean").reset_index()


def window_table(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in sorted(folder.glob("*_windows.csv")):
        frame = pd.read_csv(path)
        name = path.stem.replace("_windows", "")
        model = name.rsplit("_s", 1)[0] if "_s" in name and name.split("_s")[-1].isdigit() else name
        frame["model"] = model
        rows.append(frame)
    table = pd.concat(rows, ignore_index=True)
    models = table[table["window_months"].notna()]
    means = models.groupby(["model", "window_months"])["nrmse"].agg(["mean", "std", "count"]).reset_index()
    months = models.groupby("window_months")[["first_month", "last_month", "holdout", "train_rows"]].first().reset_index()
    baselines = table[table["window_months"].isna()][["model", "nrmse", "test_rows"]]
    return means, pd.concat([months, baselines], ignore_index=True)


def window_grid(folder: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p).assign(model=p.stem.split("_s")[0]) for p in folder.glob("*_window_grid.csv")]
    grid = pd.concat(frames, ignore_index=True)
    return grid.groupby(["model", "window_months"])["nrmse"].agg(["min", "median", "max"]).reset_index()


def seed_mean(folder: Path, name: str) -> float:
    runs = load_runs(folder, name)
    return runs.groupby("seed")["nrmse"].mean().mean() if not runs.empty else float("nan")


def geometry_offsets(data_root: str) -> pd.DataFrame:
    import pvlib

    from pvforecast.dataset import load_site

    raw, site = load_site(data_root, "site_b")
    index = raw.index + pd.Timedelta(minutes=30)
    base = pvlib.location.Location(site.latitude, site.longitude, tz="UTC")
    ref_pos, ref_cs = base.get_solarposition(index), base.get_clearsky(index)
    day = ref_pos["elevation"] > 5
    rows = []
    for km, bearing in itertools.product((5.0, 10.0), range(0, 360, 45)):
        north = km * math.cos(math.radians(bearing)) / 111.32
        east = km * math.sin(math.radians(bearing)) / (111.32 * math.cos(math.radians(site.latitude)))
        moved = pvlib.location.Location(site.latitude + north, site.longitude + east, tz="UTC")
        pos, cs = moved.get_solarposition(index), moved.get_clearsky(index)
        rows.append({"km": km, "bearing": bearing,
                     "elev_rmse_deg": float(np.sqrt(((pos["elevation"] - ref_pos["elevation"])[day] ** 2).mean())),
                     "elev_max_deg": float((pos["elevation"] - ref_pos["elevation"])[day].abs().max()),
                     "azim_rmse_deg": float(np.sqrt(((pos["azimuth"] - ref_pos["azimuth"])[day] ** 2).mean())),
                     "cs_ghi_rmse_wm2": float(np.sqrt(((cs["ghi"] - ref_cs["ghi"])[day] ** 2).mean())),
                     "cs_ghi_max_wm2": float((cs["ghi"] - ref_cs["ghi"])[day].abs().max())})
    return pd.DataFrame(rows)


def importance(data_root: str, selected: dict) -> pd.DataFrame:
    from pvforecast.dataset import load_site
    from pvforecast.evaluation import training_frame
    from pvforecast.features import FULL_FEATURES, TARGET, WEIGHT, build_features, design_matrix
    from pvforecast.models import get_model
    from pvforecast.weather import weather_inputs

    raw, site = load_site(data_root, "site_a")
    df = build_features(raw, site, weather_inputs(raw, "site_a", "nwp"))
    train = training_frame(df, FULL_FEATURES, 0.92, 18)
    params = {k: v for k, v in selected["xgboost"].items() if k != "val_nrmse"}
    predict = get_model("xgboost", **params, seed=0)(design_matrix(train, FULL_FEATURES),
                                                     train[TARGET].fillna(0).to_numpy(np.float32),
                                                     train[WEIGHT].to_numpy(np.float32))
    gain = predict.model.get_booster().get_score(importance_type="total_gain")
    named = {FULL_FEATURES[int(k[1:])]: v for k, v in gain.items()}
    table = pd.Series(named).reindex(FULL_FEATURES).fillna(0.0)
    shares = (table / table.sum() * 100).sort_values(ascending=False).rename("gain_pct")
    return shares.rename_axis("feature").reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/revision")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    results = Path(args.results_dir)
    out = results / "report"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    run = lambda part: args.only is None or part in args.only  # noqa: E731

    if run("benchmark"):
        for config in ("nwp", "reanalysis", "inverter_only"):
            folder = results / "benchmark" / config
            table, monthly = summary_table(folder, MODELS + BASELINES)
            table.to_csv(out / f"benchmark_{config}.csv", index=False)
            monthly.to_csv(out / f"monthly_{config}.csv")
            print(f"\n=== benchmark [{config}]\n{table.round(3).to_string(index=False)}")
            if config == "nwp":
                tests = pairwise(monthly[[c for c in MODELS + BASELINES if c in monthly]])
                tests.to_csv(out / "pairwise_nwp.csv", index=False)
                print(tests.round(4).to_string(index=False))

    if run("cross_site"):
        tables = {}
        for site in ("site_a", "site_b"):
            tables[site], _ = summary_table(results / "cross_site" / site, MODELS + BASELINES)
        merged = tables["site_a"][["model", "nrmse", "nrmse_sd"]].merge(
            tables["site_b"][["model", "nrmse", "nrmse_sd"]], on="model", suffixes=("_a", "_b"))
        models_only = merged[merged["model"].isin(MODELS)].copy()
        models_only["rank_a"] = models_only["nrmse_a"].rank().astype(int)
        models_only["rank_b"] = models_only["nrmse_b"].rank().astype(int)
        rho, p = spearman_exact(models_only["rank_a"].tolist(), models_only["rank_b"].tolist())
        merged.to_csv(out / "cross_site.csv", index=False)
        print(f"\n=== cross-site\n{merged.round(3).to_string(index=False)}\nSpearman rho={rho:.3f} exact p={p:.4f}")
        (out / "cross_site_spearman.json").write_text(json.dumps({"rho": rho, "exact_p": p}))

    if run("training_window"):
        means, months = window_table(results / "training_window" / "nwp")
        means.to_csv(out / "training_window.csv", index=False)
        months.to_csv(out / "training_window_months.csv", index=False)
        print(f"\n=== training window\n{means.pivot(index='model', columns='window_months', values='mean').round(2).to_string()}")
        print(months.to_string(index=False))
        grid = window_grid(results / "training_window" / "grid")
        grid.to_csv(out / "training_window_tree_grid.csv", index=False)
        print(grid.round(2).to_string(index=False))

    if run("variants"):
        nwp = results / "benchmark" / "nwp"
        warm = {name: seed_mean(results / "warm_start" / "nwp", name) for name in ("xgboost_warm", "lightgbm_warm", "transformer_warm")}
        scratch = {name: seed_mean(nwp, name) for name in ("xgboost", "lightgbm", "transformer")}
        print("\n=== warm start", {k: round(v, 3) for k, v in {**warm, **scratch}.items()})
        warm_table, _ = summary_table(results / "warm_start" / "nwp", ["xgboost_warm", "lightgbm_warm", "transformer_warm"])
        warm_table.to_csv(out / "warm_start.csv", index=False)

        physics = {m: (seed_mean(results / "physics" / "base", m), seed_mean(nwp, m)) for m in ("xgboost", "lightgbm", "transformer", "cnn_lstm")}
        pd.DataFrame(physics, index=["base25", "full28"]).T.to_csv(out / "physics.csv")
        print("=== physics", physics)

        full = seed_mean(nwp, "xgboost")
        ablation = {g.name: seed_mean(g, "xgboost") - full for g in sorted((results / "ablation").iterdir())}
        pd.Series(ablation, name="delta_nrmse").to_csv(out / "ablation.csv")
        print("=== ablation (full =", round(full, 3), ")", {k: round(v, 3) for k, v in ablation.items()})

        decay = {(m, g.name): seed_mean(g, m) for g in sorted((results / "decay").iterdir()) for m in ("xgboost", "lightgbm")}
        pd.Series(decay, name="nrmse").to_csv(out / "decay.csv")
        print("=== decay", {k: round(v, 3) for k, v in decay.items()})

        geo_runs = {g.name: load_runs(g, "xgboost").groupby("seed")["nrmse"].mean() for g in sorted((results / "site_b_geometry").iterdir())}
        proxy = geo_runs.pop("proxy")
        deltas = pd.DataFrame({k: v - proxy for k, v in geo_runs.items()})
        geo = pd.DataFrame({"delta_mean": deltas.mean(), "delta_min": deltas.min(), "delta_max": deltas.max()})
        geo.to_csv(out / "site_b_geometry.csv")
        print(f"=== site B geometry (proxy {proxy.mean():.3f})\n{geo.round(3).to_string()}")
        offsets = geometry_offsets(args.data_root)
        offsets.to_csv(out / "site_b_geometry_offsets.csv", index=False)
        print(offsets.round(3).to_string(index=False))

        ens = ensembles(results, [0, 1, 2])
        ens.to_csv(out / "ensembles.csv", index=False)
        print(f"=== ensembles\n{ens.round(3).to_string(index=False)}")

        selected = json.loads((results / "tuning" / "selected.json").read_text())
        imp = importance(args.data_root, selected)
        imp.to_csv(out / "importance.csv", index=False)
        print(f"=== importance\n{imp.head(12).round(2).to_string(index=False)}")


if __name__ == "__main__":
    main()
