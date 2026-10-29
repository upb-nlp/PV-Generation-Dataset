# PV Forecast Toolkit

Short-term photovoltaic generation forecasting for prosumer-scale installations.
Runs on the two-site dataset in this repository, or on any hourly inverter
history with the same columns.

It does two things: forecasts future generation from a recorded history, and
re-runs the experiments reported in the companion paper.

## Install

```
pip install -r toolkit/requirements.txt
```

PyTorch is only needed for the five deep models. Leave it out and the gradient
boosting models, the baselines and the operational forecaster all still work.

All commands below run from the repository root, so `--data-root .` finds
`site_a/` and `site_b/`.

## Forecast the next two weeks

```
python toolkit/scripts/forecast.py --data-root . --site site_a --days 14
```

This trains on the site's whole history and writes hourly and daily forecasts to
`results/forecast/`. Weather past the end of the record falls back to the site's
month-by-hour climatology. Supply a real forecast instead with `--weather
file.csv`, where the file has a timestamp column plus any of `temp_c`,
`humidity_pct`, `cloud_cover_pct`, `wind_speed_kmh`, `ghi_wm2`, `dni_wm2`,
`dhi_wm2`. `--recency-correction` rescales the climatological part of the
forecast by the observed-to-climatological ratio of the last 30 days; it is off
by default because it made the forecasts worse at most origins in the paper's
evaluation.

## Reproduce the paper

The paper's results come from one pipeline. Every model is tuned once, every
walk-forward configuration is run with three seeds as an independent, resumable
job, and the tables and figures are then built from those runs, so a
configuration that appears in several tables comes from a single run.

```
python toolkit/scripts/tune.py --data-root . --results-dir results/revision
python toolkit/scripts/revision_runner.py --data-root .
python toolkit/scripts/operational.py --data-root .
python toolkit/scripts/revision_report.py --data-root .
python toolkit/scripts/latex_tables.py
python toolkit/scripts/figures.py
```

| Script | What it does | Paper |
|---|---|---|
| `tune.py` | Searches nine configurations per model on one temporal validation month (train January–May 2025, validate June 2025) and sets the training length by early stopping; writes `results/revision/tuning/selected.json` | Table 1 |
| `revision_runner.py` | Runs the walk-forward experiments (`--experiments benchmark cross_site training_window warm_start physics ablation decay site_b_geometry`), seeds 0–2, under the three weather configurations | Tables 2–9, A2 |
| `operational.py` | Fixed-origin 14-day forecasts from every daily origin at Site A, with the weather fixed at the origin, and the recency-correction test | Table 10, Figures 6–7 |
| `revision_report.py` | Summary tables, exact Wilcoxon tests with Holm correction, ensembles, cross-site ranks, feature importance; writes `results/revision/report/` | Tables 2–9, A2–A3 |
| `latex_tables.py` | LaTeX table bodies from the report | all tables |
| `figures.py` | Vector PDF figures; writes `results/revision/figures/` | Figures 1–7 |

The boosting models and the baselines finish in minutes on a CPU; the deep
models take several hours. `revision_runner.py --only-trees` skips them.

The older single-experiment scripts still work for quick runs and take
`--data-root`, `--site`, `--feature-set`, `--weather`, `--start`, `--end`
(months as `YYYY-MM`), `--results-dir` and `--seed`:

| Script | What it does |
|---|---|
| `benchmark.py` | Compares the models against the persistence baselines |
| `training_window.py` | Varies how many months of history the model is given |
| `ablation.py` | Removes one feature group at a time |
| `ensemble.py` | Averaging, inverse-error weighting, ridge stacking |
| `cross_site.py` | Ranks the models at both sites over an aligned window |
| `importance.py` | Gain importance per feature |
| `deployment_horizon.py` | Forecasts a multi-week horizon with no observation from it |

## Use it on your own data

Point `--data-root` at a directory holding `<site>/<site>_hourly_weather.csv`
with the same columns as the released files: `datetime_utc` (or
`datetime_local`, Europe/Bucharest time), `active_power_kw`, and the seven
weather columns (`temp_c`, `humidity_pct`, `cloud_cover_pct`, `wind_speed_kmh`,
`ghi_wm2`, `dni_wm2`, `dhi_wm2`). Then declare the site:

```python
from pvforecast.config import SITES, Site

SITES["my_site"] = Site("my_site", latitude=45.0, longitude=23.3, capacity_kwp=50.0)
```

Coordinates drive the pvlib solar geometry, and rated capacity converts power to
capacity factor, so both have to be right for your installation.

## How it works

Every site is put on a complete hourly UTC grid: the row stamped `u` is the mean
output over `[u, u+1h)`, paired with the weather record valid at `u+1h`
(Open-Meteo radiation is averaged over the hour ending at its stamp) and with
the solar geometry at the middle of the hour. Hours missing from the inverter
export stay as gaps and are never filled.

The target is capacity factor, `active_power_kw / rated_kwp` clipped to [0, 1].
Features are 28 columns in six groups: solar geometry (5), weather (7), cyclic
time encodings (6), capacity-factor lags at 1, 2, 3 and 6 hours (4), rolling mean
and standard deviation (3), and physics-derived terms (3). `--feature-set`
selects `full` (28), `base` (25, without the physics terms), `core` (the 7 lag
and rolling columns) or `endogenous` (18, no weather).

`--weather` chooses where the weather columns come from: `nwp` (default), the
Open-Meteo forecast issued 24 hours before each hour ends, which a live system
holds before the hour starts; `reanalysis`, the retrospective record of the
weather that occurred, kept only as a reference because it is not available
operationally; or `none`.

Rolling windows close one hour before the hour being predicted and need a
complete window, so no input contains the value being predicted and no feature
reaches into a gap. Training rows carry an exponential recency weight, 0.92 per
month with an 18-month cutoff. Night rows stay in the model input so the
sequence models see contiguous 24-hour windows, and are dropped from the
metrics.

Evaluation is rolling-origin: train on months 1 to N, predict month N+1, slide
forward, starting once six months of history exist. Nothing is split at random.
Metrics are MAE and RMSE in kW, MAPE, nRMSE as a percentage of rated capacity,
and R2, over daytime hours where solar elevation exceeds 5 degrees; all models
and baselines are scored on the same hours. Smart persistence, the reference for
the skill scores, keeps the clear-sky index of the previous hour,
`CF(t-1) * GHIcs(t) / GHIcs(t-1)`.

Past the end of the recorded history there are no observations to build the lag
and rolling features from, so they are seeded from the month-and-hour
climatology. Solar geometry is computed exactly for any future timestamp.
Predictions are never fed back into the features.

## Layout

| Module | Contents |
|---|---|
| `config.py` | Site description, registry, seed control |
| `dataset.py` | CSV loading, UTC alignment, month indexing, calendar restriction |
| `weather.py` | Archived forecasts, reanalysis and no-weather input configurations |
| `features.py` | Target, feature groups, feature construction, recency weights |
| `models.py` | XGBoost, LightGBM, warm-start variants |
| `deep.py` | LSTM, CNN-LSTM, attention-LSTM, Transformer, N-BEATS |
| `baselines.py` | Naive, day and smart persistence |
| `evaluation.py` | Metrics, rolling-origin and fixed-holdout evaluation |
| `ensemble.py` | Averaging, inverse-error weighting, ridge stacking |
| `forecast.py` | Operational forecaster, climatology, energy aggregation |
| `cli.py` | Argument plumbing shared by the scripts |

A model is a factory returning `model_fn(X_train, y_train, w_train)`, which
returns `predict_fn(X_test, timestamps)`. Anything with that signature can be
evaluated without registering it.

`data/nwp/` holds the archived Open-Meteo forecasts used in the paper, at lead
times of 0 to 7 days, stamped by valid time in UTC. Both sites share them
because they fall in the same weather-model grid cell (`metadata.json`). They
are kept here because the Open-Meteo archive only retains model runs for seven
days, so they cannot be fetched again. `fetch_forecast_weather.py` downloads the
equivalent files for another site or period, within that window.
`build_dataset_v11.py` builds the UTC-aligned hourly files of the dataset from
hourly files stamped in local time, and copies the archived forecasts beside
them.
