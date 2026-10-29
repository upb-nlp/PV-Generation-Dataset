# Two-Site Solar PV Inverter & Weather Dataset (Romania)

Hourly and 5-minute AC power generation data from two grid-connected,
50 kW solar PV systems in southwestern Romania, each
equipped with a Huawei SUN2000 string inverter, joined with matched
historical weather (Open-Meteo) and solar position. Built for research on
short-term PV power forecasting; released as a companion dataset to a
dissertation studying gradient boosting vs. deep learning architectures for
single- and cross-site solar generation prediction.

The repository also ships the forecasting toolkit used to produce those
results. See [Forecasting toolkit](#forecasting-toolkit).

## What's in here

Each site has its own folder with two layers. The archived weather
forecasts and the toolkit sit alongside them:

```
site_a/
├── site_a_hourly_weather.csv   ← recommended starting point
└── raw/
    ├── site_a_2025-01.xlsx     ← original 5-minute inverter export (anonymized, trimmed)
    ├── site_a_2025-02.xlsx
    └── ...
site_b/
├── site_b_hourly_weather.csv
└── raw/
    └── ...
archived_forecasts/
├── site_a_and_b_lead0.csv      ← Open-Meteo forecasts issued 0 to 7 days ahead
└── ...
toolkit/
├── pvforecast/                 forecasting library
├── scripts/                    experiment and reproduction scripts
├── data/nwp/                   archived weather forecasts used by the toolkit
└── requirements.txt
```

| | Site A | Site B |
|---|---|---|
| Coverage | 2025-01-04 → 2026-06-30 | 2025-05-01 → 2026-05-31 |
| Months | 18 | 13 |
| Inverter | Huawei SUN2000, 50KTL-M0 | Huawei SUN2000, 50K-M0 |
| Rated capacity | 50 kW (inverter nominal AC rating) | 50 kW (inverter nominal AC rating) |
| Location | Southwestern Romania (≈45.0°N, 23.3°E, see [Anonymization](#anonymization)) | Same region |
| Climate | Sub-Carpathian temperate: hot/moderately sunny summers, cold/frequently overcast winters | Same |

Site A and Site B are two independent, unaffiliated installations in the
same climate region, which is useful for testing whether a model or finding
trained on one site generalizes to another.

### Layer 1: `<site>_hourly_weather.csv` (recommended)

One row per UTC hour, on a complete hourly grid. Each row holds the inverter
output averaged from the 5-minute readings of the hour `[datetime_utc,
datetime_utc + 1 h)`, joined with the hourly historical weather (Open-Meteo
Archive API) and the solar position (computed with
[pvlib](https://pvlib-python.readthedocs.io/)) of the same interval, at the
anonymized site coordinates. The weather columns hold the Open-Meteo record
valid at the end of the hour, whose radiation values are averages over that
hour, and the solar position is computed at the midpoint of the hour.

| Column | Unit | Description |
|---|---|---|
| `datetime_utc` | timestamp | Start of the hour in UTC; the row covers [datetime_utc, datetime_utc + 1 h) |
| `datetime_local` | timestamp | Start of the same hour in Europe/Bucharest time, with its UTC offset |
| `active_power_kw` | kW | AC power delivered to the grid, the primary target variable |
| `dc_input_power_kw` | kW | Total DC input power from the PV strings |
| `daily_energy_kwh` | kWh | Cumulative energy generated since midnight (resets daily) |
| `cumulative_energy_kwh` | kWh | Lifetime cumulative energy yield of the inverter |
| `inverter_temp_c` | °C | Internal inverter temperature |
| `grid_freq_hz` | Hz | Grid frequency at the point of connection |
| `power_factor` | — | AC power factor |
| `efficiency_pct` | % | Inverter conversion efficiency |
| `inverter_status` | text (English) | Inverter status, e.g. `Standby: no sunlight`, `Connected to grid` (see the [full list](#inverter-status-values)) |
| `is_generating` | 0/1 | 1 if any 5-minute reading in that hour had `active_power_kw > 0` |
| `temp_c` | °C | Ambient air temperature (Open-Meteo, 2 m) |
| `humidity_pct` | % | Relative humidity |
| `cloud_cover_pct` | % | Total cloud cover |
| `wind_speed_kmh` | km/h | Wind speed at 10 m |
| `ghi_wm2` | W/m² | Global horizontal irradiance |
| `dni_wm2` | W/m² | Direct normal irradiance |
| `dhi_wm2` | W/m² | Diffuse horizontal irradiance |
| `precip_mm` | mm | Hourly precipitation |
| `solar_elevation_deg` | ° | Sun elevation at the midpoint of the hour, clipped at zero |
| `solar_azimuth_deg` | ° | Sun azimuth at the midpoint of the hour (0° = north, 90° = east) |
| `capacity_factor` | 0–1 | `active_power_kw / 50.0`, clipped to [0, 1], the standard normalized target for cross-site comparison |

Night-time hours are included with `active_power_kw = 0` (the inverter
genuinely reports zero, not missing data). Hours missing from the inverter
exports, around the daylight-saving changes at the ends of March and October,
are present with empty measurement columns, never zeros. Filter on
`solar_elevation_deg` if you want daytime-only evaluation, which is the
convention used in the accompanying research (elevation > 5°).

### Layer 2: `raw/<site>_YYYY-MM.xlsx`

The original Huawei FusionSolar 5-minute export for each calendar month,
anonymized and trimmed. Trimmed from ~114 columns
down to the 13 that carry real information (see
[Anonymization](#anonymization); most of the discarded columns are
per-string diagnostics that are >98% null on a 6-MPPT inverter, or
administrative fields).

The row layout matches the original vendor format. All column headers,
metadata labels, and status text are in English (see the header table
below). A parser written for the raw vendor export column names will not
match these headers as-is; start from the already-clean
`hourly_weather.csv` if you want the simplest interface.

- Row 1: `Time range:` (calendar period covered by the export)
- Row 2: `Export date/time:` (when the file was generated, not a data value)
- Row 3: legend / footnote text
- Row 4: column headers
- Row 5 onward: one row per 5-minute reading

| Header | Meaning |
|---|---|
| `Site Name` | Site display name, anonymized to `Site A` / `Site B` |
| `Installer` | Installer/management company, anonymized to `Installer` |
| `Device Name` | Logger/device identifier, anonymized with the model substring preserved |
| `Start Date/Time` | Reading start timestamp |
| `Active Power (kW)` | AC active power |
| `Total Input Power (kW)` | Total DC input power |
| `Daily Energy (kWh)` | Daily energy (resets at midnight) |
| `Total PV Yield (kWh)` | Lifetime cumulative energy |
| `Internal Temperature (°C)` | Internal inverter temperature |
| `Grid Frequency (Hz)` | Grid frequency |
| `Power Factor` | Power factor |
| `Conversion Efficiency (%)` | Conversion efficiency |
| `Inverter Status` | Inverter status string |

#### Inverter status values

Eight distinct inverter status strings occur across both sites:

- Connected to grid
- OFF: unexpected shutdown
- Standby: detecting sunlight
- Standby: detecting insulation resistance
- Standby: detecting grid power
- Standby: initializing
- Standby: no sunlight
- Starting up

**Known data quirks** (inherited from the source logger, not artifacts of
anonymization):
- `-0.01` / `-0.1` sentinel values in some raw exports indicate a
  disconnected string channel. They are already absent from the trimmed
  columns above, but worth knowing if you go back to a fuller export.
- Some months have DST timezone suffixes (`DST`/`EEST`) appended to
  timestamps; strip them before parsing with `pandas.to_datetime`.
- Site A's first data point starts 2025-01-04 rather than 2025-01-01; the
  first 3 days of January 2025 were not present in the original export.

### Archived weather forecasts: `archived_forecasts/`

Open-Meteo forecasts from its previous-runs archive, as they were issued 0 to 7
days before each hour (`site_a_and_b_lead<k>.csv` holds the run issued `k` days
ahead), stamped by `valid_time_utc` with the same seven weather columns as the
hourly files. Unlike the historical weather in the hourly files, these were
available before the hour they describe, so they can be used as inputs to an
operational forecast. Both sites share them because they fall in the same
weather-model grid cell. They are kept because the archive retains model runs
for only seven days, so they cannot be downloaded again.

## Quick start

```python
import pandas as pd

df = pd.read_csv("site_a/site_a_hourly_weather.csv", parse_dates=["datetime_utc"])
df = df.set_index("datetime_utc")

# Daytime-only rows, the convention used in the accompanying research
daytime = df[df["solar_elevation_deg"] > 5]
print(daytime[["active_power_kw", "capacity_factor", "ghi_wm2"]].describe())
```

## Forecasting toolkit

`toolkit/` holds the code behind the published experiments. It forecasts
future generation from a recorded inverter history, and re-runs each
experiment in the accompanying paper. It works on the two sites here or on
any hourly history with the same columns.

### Install and forecast

```
pip install -r toolkit/requirements.txt
python toolkit/scripts/forecast.py --data-root . --site site_a --days 14
```

Run from the repository root, so `--data-root .` finds `site_a/` and
`site_b/`. This trains on the site's whole history and writes hourly and
daily forecasts to `results/forecast/`. Weather beyond the end of the record
falls back to the site's month-by-hour climatology; supply a real forecast
with `--weather file.csv` instead.

### Reproducing the paper

```
python toolkit/scripts/tune.py --data-root . --results-dir results/revision
python toolkit/scripts/revision_runner.py --data-root .
python toolkit/scripts/operational.py --data-root .
python toolkit/scripts/revision_report.py --data-root .
python toolkit/scripts/latex_tables.py
python toolkit/scripts/figures.py
```

`tune.py` selects every model's configuration on one validation month,
`revision_runner.py` runs the walk-forward experiments with three seeds,
`operational.py` evaluates fixed-origin 14-day forecasts, and the last three
scripts build the tables, statistical tests and figures.
[`toolkit/README.md`](toolkit/README.md) maps each script to the tables and
figures of the paper.

PyTorch is needed only for the five deep models. Without it, the gradient
boosting models, the persistence baselines and the operational forecaster
all still run.

### Running single experiments

These scripts each run one experiment and write a table to
`results/tables/`, with per-month results beside it.

| Script | What it does |
|---|---|
| `benchmark.py` | Compares the seven models against the persistence baselines |
| `training_window.py` | Varies how many months of history the model is given |
| `ablation.py` | Removes one feature group at a time |
| `ensemble.py` | Averaging, inverse-error weighting, ridge stacking |
| `cross_site.py` | Ranks the models at both sites over an aligned window |
| `importance.py` | Gain importance per feature |
| `deployment_horizon.py` | Forecasts a multi-week horizon with no observation from it |

```
python toolkit/scripts/benchmark.py --data-root . --site site_a
```

Every script takes `--data-root`, `--site`, `--feature-set`, `--weather`,
`--start`, `--end` (months as `YYYY-MM`), `--results-dir` and `--seed`. The boosting
models and the baselines run in about a minute on a CPU; the deep models
take considerably longer.

### Method in brief

The target is `capacity_factor`. Features are 28 columns in six groups:
solar geometry (5), weather (7), cyclic time encodings (6), capacity-factor
lags at 1, 2, 3 and 6 hours (4), rolling mean and standard deviation (3),
and physics-derived terms (3). By default the weather columns come from the
forecast issued 24 hours ahead (`toolkit/data/nwp/`), and rolling windows close
one hour before the hour being predicted, so every input is available at
prediction time.
Evaluation is rolling-origin, never random: train on months 1 to N, predict
month N+1, slide forward.

[`toolkit/README.md`](toolkit/README.md) documents the feature sets, the
module layout, how to register your own site, and how to plug in a
different model.

## Anonymization

This dataset describes real, privately owned solar PV systems. The
following was removed or generalized before release:

- Owner and site names were replaced with generic `Site A` / `Site B` labels in every row of every file.
- The installer company name was replaced with a generic `Installer` placeholder.
- Logger and device identifiers were rebuilt to keep only the inverter model number, which is a hardware spec rather than an identifier, and drop the rest.
- Inverter serial numbers were removed entirely. They appeared only in the original file names, which have been renamed.
- Exact GPS coordinates were rounded to 1 decimal place (~11 km resolution) in this README and in the solar-position calculation. That is precise enough for solar geometry (elevation and azimuth) but not precise enough to identify an address.
- Document metadata was checked: the original files' embedded author and editor properties contained no personal information, only the generating software's name, so no further action was needed.

All measurement data (power, energy, temperature, weather) is unmodified;
the hourly files give each timestamp in UTC alongside local time. Every file was scanned line-by-line after processing to confirm
no owner name, company name, street name, or serial number survived.

## Known limitations

- Both sites are in the same climate region and inverter class, so this
  dataset alone cannot establish whether findings generalize to different
  climates, orientations, or system sizes.
- Site B has 5 fewer months of history than Site A (13 vs. 18).
- The weather columns of the hourly files are reanalysis data (Open-Meteo
  Archive), not on-site measurement, so expect some divergence from true local
  microclimate, especially cloud cover. They describe the weather that occurred
  and are published days later; for operational inputs use
  `archived_forecasts/`.
- No irradiance sensor ground-truth is available; `ghi_wm2`/`dni_wm2`/`dhi_wm2`
  are Open-Meteo estimates.
- The raw `.xlsx` files use English column headers, so a parser written for
  the original vendor export headers will not work as-is (see Layer 2 above).

## License

Released under **CC-BY-4.0** (Creative Commons Attribution 4.0
International). You may use, share, and adapt this data for any purpose,
including commercially, as long as you give appropriate credit. See
[`LICENSE`](LICENSE) for the full text.

## Citing this dataset

```
Linca, N.-R., Rebedea, T.-E., Ruseti, S., and Dascalu, M. (2026).
Two-Site Solar PV Inverter & Weather Dataset (Romania). GitHub repository.
https://github.com/upb-nlp/PV-Generation-Dataset
```

A companion research paper describes the short-term PV forecasting experiments
built on this dataset and on the toolkit released here (gradient boosting vs.
deep learning, small-window and cross-site evaluation). Citation details will
be added here once it is published.
