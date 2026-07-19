# Two-Site Solar PV Inverter & Weather Dataset (Romania)

Hourly and 5-minute AC power generation data from two grid-connected,
50 kWp-class solar PV systems in southwestern Romania, each
equipped with a Huawei SUN2000 string inverter, joined with matched
historical weather (Open-Meteo) and solar position. Built for research on
short-term PV power forecasting; released as a companion dataset to a
dissertation studying gradient boosting vs. deep learning architectures for
single- and cross-site solar generation prediction.

All original Romanian column headers, status labels, and metadata text have
been translated to English (see [Translation](#translation) below); no
Romanian-language content remains anywhere in the dataset.

## What's in here

Each site has its own folder with two layers:

```
site_a/
├── site_a_hourly_weather.csv   ← recommended starting point
└── raw/
    ├── site_a_2025-01.xlsx     ← original 5-minute inverter export (anonymized, trimmed, translated)
    ├── site_a_2025-02.xlsx
    └── ...
site_b/
├── site_b_hourly_weather.csv
└── raw/
    └── ...
```

| | Site A | Site B |
|---|---|---|
| Coverage | 2025-01-04 → 2026-06-30 | 2025-05-01 → 2026-05-31 |
| Months | 18 | 13 |
| Inverter | Huawei SUN2000, 50KTL-M0 | Huawei SUN2000, 50K-M0 |
| Rated capacity | 50 kWp | 50 kWp |
| Location | Southwestern Romania (≈45.0°N, 23.3°E, see [Anonymization](#anonymization)) | Same region |
| Climate | Sub-Carpathian temperate: hot/moderately sunny summers, cold/frequently overcast winters | Same |

Site A and Site B are two independent, unaffiliated installations in the
same climate region, which is useful for testing whether a model or finding
trained on one site generalizes to another.

### Layer 1: `<site>_hourly_weather.csv` (recommended)

One row per hour. Inverter output resampled from 5-minute readings, left-joined
with hourly historical weather (Open-Meteo Archive API) and solar position
(computed with [pvlib](https://pvlib-python.readthedocs.io/)) at the
anonymized site coordinates.

| Column | Unit | Description |
|---|---|---|
| `datetime_local` | timestamp | Local time (Europe/Bucharest, DST-naive as exported by the inverter logger) |
| `active_power_kw` | kW | AC power delivered to the grid, the primary target variable |
| `dc_input_power_kw` | kW | Total DC input power from the PV strings |
| `daily_energy_kwh` | kWh | Cumulative energy generated since midnight (resets daily) |
| `cumulative_energy_kwh` | kWh | Lifetime cumulative energy yield of the inverter |
| `inverter_temp_c` | °C | Internal inverter temperature |
| `grid_freq_hz` | Hz | Grid frequency at the point of connection |
| `power_factor` | — | AC power factor |
| `efficiency_pct` | % | Inverter conversion efficiency |
| `inverter_status` | text (English) | Inverter status, e.g. `Standby: no sunlight`, `Connected to grid` (see the [full translation table](#status-translation)) |
| `is_generating` | 0/1 | 1 if any 5-minute reading in that hour had `active_power_kw > 0` |
| `temp_c` | °C | Ambient air temperature (Open-Meteo, 2 m) |
| `humidity_pct` | % | Relative humidity |
| `cloud_cover_pct` | % | Total cloud cover |
| `wind_speed_kmh` | km/h | Wind speed at 10 m |
| `ghi_wm2` | W/m² | Global horizontal irradiance |
| `dni_wm2` | W/m² | Direct normal irradiance |
| `dhi_wm2` | W/m² | Diffuse horizontal irradiance |
| `precip_mm` | mm | Hourly precipitation |
| `solar_elevation_deg` | ° | Sun elevation angle above the horizon (negative = sun below horizon) |
| `solar_azimuth_deg` | ° | Sun azimuth angle (0° = north, 90° = east) |
| `capacity_factor` | 0–1 | `active_power_kw / 50.0`, clipped to [0, 1], the standard normalized target for cross-site comparison |

Night-time hours are included with `active_power_kw = 0` (the inverter
genuinely reports zero, not missing data). Filter on `solar_elevation_deg`
if you want daytime-only evaluation, which is the convention used in the
accompanying research (elevation > 5°).

### Layer 2: `raw/<site>_YYYY-MM.xlsx`

The original Huawei FusionSolar 5-minute export for each calendar month,
anonymized, trimmed, and translated to English. Trimmed from ~114 columns
down to the 13 that carry real information (see
[Anonymization](#anonymization); most of the discarded columns are
per-string diagnostics that are >98% null on a 6-MPPT inverter, or
administrative fields).

The row layout matches the original vendor format, but **all header names,
metadata labels, and status text have been translated from Romanian to
English**. This file will not work as-is with a parser written against the
original Romanian FusionSolar export column names. If you need to reproduce
that exact pipeline, rename the columns back per the table below, or start
from the already-English `hourly_weather.csv` instead.

- Row 1: `Time range:` (calendar period covered by the export)
- Row 2: `Export date/time:` (when the file was generated, not a data value)
- Row 3: legend / footnote text
- Row 4: column headers
- Row 5 onward: one row per 5-minute reading

| English header (this file) | Original Romanian header | Meaning |
|---|---|---|
| `Site Name` | `Nume locație` | Site display name, anonymized to `Site A` / `Site B` |
| `Installer` | `Domeniu de management` | Installer/management company, anonymized to `Installer` |
| `Device Name` | `Nume dispozitiv` | Logger/device identifier, anonymized with the model substring preserved |
| `Start Date/Time` | `Dată și oră de început` | Reading start timestamp |
| `Active Power (kW)` | `Putere activă(kW)` | AC active power |
| `Total Input Power (kW)` | `Putere totală de intrare(kW)` | Total DC input power |
| `Daily Energy (kWh)` | `Energie zilnică(kWh)` | Daily energy (resets at midnight) |
| `Total PV Yield (kWh)` | `Randament fotovoltaic total(kWh)` | Lifetime cumulative energy |
| `Internal Temperature (°C)` | `Temperatura internă(℃)` | Internal inverter temperature |
| `Grid Frequency (Hz)` | `Frecvența rețelei(Hz)` | Grid frequency |
| `Power Factor` | `Factor de putere` | Power factor |
| `Conversion Efficiency (%)` | `Eficiența conversiei(%)` | Conversion efficiency |
| `Inverter Status` | `Starea invertorului` | Inverter status string |

#### Status translation

Eight distinct inverter status strings occur across both sites; all are
translated as follows.

| English (this file) | Original Romanian |
|---|---|
| Connected to grid | Conectată la rețea |
| OFF: unexpected shutdown | OFF: oprire neașteptată |
| Standby: detecting sunlight | Standby: detectarea luminii solare |
| Standby: detecting insulation resistance | Standby: detectarea rezistenței de izolație |
| Standby: detecting grid power | Standby: detectarea rețelei de alimentare |
| Standby: initializing | Standby: inițializare |
| Standby: no sunlight | Standby: lipsă lumină solară |
| Starting up | În curs de pornire |

**Known data quirks** (inherited from the source logger, not artifacts of
anonymization or translation):
- `-0.01` / `-0.1` sentinel values in some raw exports indicate a
  disconnected string channel. They are already absent from the trimmed
  columns above, but worth knowing if you go back to a fuller export.
- Some months have DST timezone suffixes (`DST`/`EEST`) appended to
  timestamps; strip them before parsing with `pandas.to_datetime`.
- Site A's first data point starts 2025-01-04 rather than 2025-01-01; the
  first 3 days of January 2025 were not present in the original export.

## Quick start

```python
import pandas as pd

df = pd.read_csv("site_a/site_a_hourly_weather.csv", parse_dates=["datetime_local"])
df = df.set_index("datetime_local")

# Daytime-only rows, the convention used in the accompanying research
daytime = df[df["solar_elevation_deg"] > 5]
print(daytime[["active_power_kw", "capacity_factor", "ghi_wm2"]].describe())
```

## Anonymization

This dataset describes real, privately owned solar PV systems. The
following was removed or generalized before release:

- Owner and site names were replaced with generic `Site A` / `Site B` labels in every row of every file.
- The installer company name was replaced with a generic `Installer` placeholder.
- Logger and device identifiers were rebuilt to keep only the inverter model number, which is a hardware spec rather than an identifier, and drop the rest.
- Inverter serial numbers were removed entirely. They appeared only in the original file names, which have been renamed.
- Exact GPS coordinates were rounded to 1 decimal place (~11 km resolution) in this README and in the solar-position calculation. That is precise enough for solar geometry (elevation and azimuth) but not precise enough to identify an address.
- Document metadata was checked: the original files' embedded author and editor properties contained no personal information, only the generating software's name, so no further action was needed.

All measurement data (power, energy, temperature, weather, timestamps) is
unmodified. Every file was scanned line-by-line after processing to confirm
no owner name, company name, street name, or serial number survived.

## Translation

Every Romanian word in the original vendor export (column headers,
row-1/row-2 metadata labels, the row-3 legend text, and all 8 inverter
status strings) has been translated to English (see the tables above).
The dataset was scanned afterward for any remaining Romanian-alphabet
diacritics (ă, â, î, ș, ț) to confirm nothing was missed.

## Known limitations

- Both sites are in the same climate region and inverter class, so this
  dataset alone cannot establish whether findings generalize to different
  climates, orientations, or system sizes.
- Site B has 5 fewer months of history than Site A (13 vs. 18).
- Weather is reanalysis data (Open-Meteo Archive), not on-site measurement,
  so expect some divergence from true local microclimate, especially cloud cover.
- No irradiance sensor ground-truth is available; `ghi_wm2`/`dni_wm2`/`dhi_wm2`
  are Open-Meteo estimates.
- The translated raw `.xlsx` files are not drop-in compatible with a parser
  written against the original Romanian column names (see Layer 2 above).

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
built on this dataset (gradient boosting vs. deep learning, small-window and
cross-site evaluation). Citation details will be added here once it is published.
