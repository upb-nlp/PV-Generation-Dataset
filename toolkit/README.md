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

## Forecast the next two weeks

Run from the repository root, so `--data-root .` finds `site_a/` and `site_b/`:

```
python toolkit/scripts/forecast.py --data-root . --site site_a --days 14
```

This trains on the site's whole history and writes hourly and daily forecasts to
`results/forecast/`. Weather past the end of the record falls back to the site's
month-by-hour climatology. Supply a real forecast instead with `--weather
file.csv`, where the file has a timestamp column plus any of `temp_c`,
`humidity_pct`, `cloud_cover_pct`, `wind_speed_kmh`, `ghi_wm2`, `dni_wm2`,
`dhi_wm2`.

## Run the experiments

Each script runs one experiment and writes a table to `results/tables/`, with
per-month results beside it.

| Script | What it does |
|---|---|
| `benchmark.py` | Compares the seven models against the persistence baselines |
| `training_window.py` | Varies how many months of history the model is given |
| `ablation.py` | Removes one feature group at a time |
| `ensemble.py` | Averaging, inverse-error weighting, ridge stacking |
| `cross_site.py` | Ranks the models at both sites over an aligned window |
| `importance.py` | Gain importance per feature |
| `deployment_horizon.py` | Forecasts a multi-week horizon with no observation from it |
| `nwp_lead_time.py` | Varies the age of the weather input |

```
python toolkit/scripts/benchmark.py --data-root . --site site_a
```

Every script takes `--data-root`, `--site`, `--feature-set`, `--start`, `--end`
(months as `YYYY-MM`), `--results-dir` and `--seed`. The boosting models and the
baselines run in about a minute on a CPU; the deep models take considerably
longer.

## Use it on your own data

Point `--data-root` at a directory holding `<site>/<site>_hourly_weather.csv`
with the same columns as the released files: `datetime_local`,
`active_power_kw`, and the seven weather columns (`temp_c`, `humidity_pct`,
`cloud_cover_pct`, `wind_speed_kmh`, `ghi_wm2`, `dni_wm2`, `dhi_wm2`). Then
declare the site:

```python
from pvforecast.config import SITES, Site

SITES["my_site"] = Site("my_site", latitude=45.0, longitude=23.3, capacity_kwp=50.0)
```

Coordinates drive the pvlib solar geometry, and rated capacity converts power to
capacity factor, so both have to be right for your installation.

## How it works

The target is capacity factor, `active_power_kw / rated_kwp` clipped to [0, 1].
Features are 28 columns in six groups: solar geometry (5), weather (7), cyclic
time encodings (6), capacity-factor lags at 1, 2, 3 and 6 hours (4), rolling mean
and standard deviation (3), and physics-derived terms (3). `--feature-set`
selects `full` (28), `base` (25, without the physics terms) or `core` (the 7 lag
and rolling columns).

Rolling windows close one hour before the hour being predicted, so every input
is available at prediction time. Training rows carry an exponential recency
weight, 0.92 per month with an 18-month cutoff. Night rows stay in the model
input so the sequence models see contiguous 24-hour windows, and are dropped
from the metrics.

Evaluation is rolling-origin: train on months 1 to N, predict month N+1, slide
forward, starting once six months of history exist. Nothing is split at random.
Metrics are MAE and RMSE in kW, MAPE, nRMSE as a percentage of rated capacity,
and R2, over daytime hours where solar elevation exceeds 5 degrees.

Past the end of the recorded history there are no observations to build the lag
and rolling features from, so they are seeded from the month-and-hour
climatology. Solar geometry is computed exactly for any future timestamp.
Predictions are never fed back into the features.

## Layout

| Module | Contents |
|---|---|
| `config.py` | Site description, registry, seed control |
| `dataset.py` | CSV loading, month indexing, calendar restriction |
| `features.py` | Target, feature groups, feature construction, recency weights |
| `models.py` | XGBoost, LightGBM, warm-start variants |
| `deep.py` | LSTM, CNN-LSTM, attention-LSTM, Transformer, N-BEATS |
| `baselines.py` | Naive, day-ahead and clear-sky-scaled persistence |
| `evaluation.py` | Metrics, rolling-origin and fixed-holdout evaluation |
| `ensemble.py` | Averaging, inverse-error weighting, ridge stacking |
| `forecast.py` | Operational forecaster, climatology, energy aggregation |
| `cli.py` | Argument plumbing shared by the scripts |

A model is a factory returning `model_fn(X_train, y_train, w_train)`, which
returns `predict_fn(X_test, timestamps)`. Anything with that signature can be
evaluated without registering it.

`data/nwp/` holds archived weather forecasts for Site A at lead times of 0, 1, 3,
5 and 7 days, used by `nwp_lead_time.py`. They are kept because the Open-Meteo
archive only retains model runs for seven days, so they cannot be fetched again.
`fetch_forecast_weather.py` downloads the equivalent files for another site or
period, within that window.
