# Vegetation Trend Analysis

Per-pixel long-term vegetation trend from MODIS EVI time series, using a
wavelet decomposition and Sen's slope, with an optional climate detrend
against ERA5 precipitation and temperature.

The repository contains the core algorithm and a sample dataset of 2,000
cropland pixels (1,000 in Argentina, 1,000 in the USA) to run it on.

## Method

For each pixel:

1. **QA and interpolation.** EVI observations that fail the QA filter are
   dropped, and the series is linearly interpolated onto the analysis time grid.
2. **Cropland check.** Seasonal EVI peaks and troughs are detected per
   agricultural year (starting at `start_doy_agro`). The pixel is treated as
   cropland (`is_agro = 1`) when the mean seasonal maximum is above
   `max_evi_threshold` and the mean seasonal minimum is below
   `min_evi_threshold`. Only cropland pixels go through the next steps.
3. **Long-term component.** A discrete wavelet transform (`db4`, `level`
   levels) splits the series; the approximation (lowest-frequency) component
   is the long-term EVI signal `lt_evi`.
4. **Trend.** Sen's slope of `lt_evi` (Mann-Kendall test, via
   `komanawa-kendall-stats`) gives `lt_evi_slope`.
5. **Climate detrend** (`--detrend true`). Precipitation and temperature go
   through the same wavelet decomposition. `lt_evi` is then modelled as a
   function of their long-term components (`gam_current`: GAM with one spline
   per variable; `linear_regression`: linear model with time as an extra
   covariate). Sen's slope of the residuals gives `residuals_slope`: the EVI
   trend not explained by climate.

## Repository layout

```
process_parquet_wavelets_slope.py   entry point: runs the algorithm on a Parquet file
utils/
  series_analysis_20.py             per-pixel algorithm (series_analysis)
  dwt_deconstruction.py             wavelet decomposition
  peak_detection.py                 seasonal peak detection per agricultural year
  plot_pixel_summary.py             per-pixel summary plot
dataset/
  samples.parquet                   input time series, 2,000 pixels
  reference.csv                     reference results for those pixels
```

## Setup

Requires Python 3.11+ (tested with 3.12). Dependency versions are pinned in
`requirements.txt` to the ones used to produce `dataset/reference.csv`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data sources

All input series were extracted with Google Earth Engine.

### Vegetation: MODIS MOD13Q1 (EVI)

| | |
|---|---|
| Product | MODIS/Terra Vegetation Indices 16-Day L3 Global 250 m SIN Grid, Collection 6.1 |
| Earth Engine collection | `MODIS/061/MOD13Q1` |
| Bands used | `EVI` (scale factor 0.0001: stored values are EVI ×10,000) and `DetailedQA` |
| Resolution | 250 m, 16-day composites, MODIS sinusoidal projection |
| In this dataset | `evi_original`: the `EVI` value of each composite. `qa_ok`: `True` when bits 0–1 of `DetailedQA` (MODLAND QA) are `00`, i.e. "VI produced with good quality". Observations with `qa_ok = False` are dropped before interpolation. `lat` / `lon`: center of the 250 m pixel, reprojected to WGS84 |

### Climate: ERA5-Land (precipitation and temperature)

| | |
|---|---|
| Product | ERA5-Land daily aggregated |
| Earth Engine collection | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| Bands used | `total_precipitation_sum` (m) and `temperature_2m` (K) |
| Resolution | 0.1° (~9 km), daily |
| In this dataset | Both variables were aggregated to the MODIS composite dates at extraction time, and each MODIS pixel takes the ERA5-Land cell that contains its center. `pp_raw`: precipitation of each composite window divided by the window length in days (m/day). `temp_raw_celsius`: temperature converted to °C (K − 273). Both are given on the analysis time grid |

### Sample selection

`dataset/samples.parquet` holds 1,000 pixels per region, sampled at random
from a larger set of cropland pixels in the study areas:

| Region | Latitude | Longitude | Export tiles (`tile` column) |
|---|---|---|---|
| `arg` (Argentina) | 39.3° S – 28.0° S | 65.1° W – 56.8° W | 21 |
| `usa` (United States) | 31.1° N – 47.7° N | 103.0° W – 82.3° W | 33 |

`dataset/reference.csv` holds the results of running the original
raster-based pipeline on those same pixels.

### Citation and attribution

If you use this dataset, cite the source products:

- Didan, K. (2021). *MODIS/Terra Vegetation Indices 16-Day L3 Global 250m SIN
  Grid V061*. NASA EOSDIS Land Processes Distributed Active Archive Center.
  https://doi.org/10.5067/MODIS/MOD13Q1.061
- Muñoz Sabater, J. (2019). *ERA5-Land hourly data from 1950 to present*.
  Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
  https://doi.org/10.24381/cds.e2161bac

The ERA5-Land-derived series (`pp_raw`, `temp_raw_celsius`) contain modified
Copernicus Climate Change Service information (2026). Neither the European
Commission nor ECMWF is responsible for any use that may be made of the
Copernicus information or data it contains.

MODIS data are distributed by NASA's LP DAAC with no restrictions on reuse.

## Dataset

### `dataset/samples.parquet`

Long format: one row per pixel, series and date.

| Column | Description |
|---|---|
| `region` | `arg` or `usa` |
| `tile` | Source MODIS tile |
| `pixel_id` | `row_col` within the tile. Repeats across tiles: a pixel is identified by `region` + `tile` + `pixel_id` |
| `row`, `col` | Pixel position within the tile |
| `lat`, `lon` | Pixel center (WGS84) |
| `series` | Series name (see below) |
| `x_days` | Days since the first `evi_original` date |
| `date` | `YYYY-MM-DD` |
| `value` | Series value |
| `qa_ok` | `evi_original` only: `True` if the observation passes the QA filter |

Series per pixel:

| `series` | Content |
|---|---|
| `evi_original` | Raw MODIS MOD13Q1 EVI (scaled ×10,000) at its native 16-day dates, 2015-01 to 2026-06 |
| `evi_interpolated` | Defines the analysis time grid (230 dates) |
| `pp_raw` | ERA5-Land total precipitation on the analysis grid, as a rate per day (m/day) |
| `temp_raw_celsius` | ERA5-Land 2 m temperature on the analysis grid, in °C |

All pixels in the dataset are cropland pixels.

### `dataset/reference.csv`

One row per pixel with the results of the original run (`lt_evi_slope`,
`pp_slope`, `temp_slope`, `residuals_slope`, their intercepts `*_A` and
Mann-Kendall significance flags `*_significant`). Running the script with the
parameters below reproduces them.

## Usage

```bash
python process_parquet_wavelets_slope.py \
  --input dataset/samples.parquet \
  --region arg \
  --detrend true \
  --params "$(cat <<'EOF'
{"level": 5, "start_doy_agro": 183, "first_date": "2015-07-01", "last_date": "2025-12-31",
 "nan_values": -10000, "min_evi_threshold": 2600, "max_evi_threshold": 5700,
 "summer_crop_season": {"min": 289, "max": 97}, "summer_peak_modis_dates": {"min": 337, "max": 81},
 "new_period": 0, "satellite": "modis", "detrend_model": "gam_current"}
EOF
)" \
  --output output/slope_arg.parquet \
  --workers 8
```

| Argument | Description |
|---|---|
| `--input` | Input Parquet file |
| `--region` | Region to process. Required when the file has more than one region, since each region uses its own parameters |
| `--params` | Algorithm parameters, as JSON (see below) |
| `--detrend` | `true` / `false`: run the climate detrend (default `true`) |
| `--output` | Output file, `.parquet` or `.csv` |
| `--workers` | Number of parallel processes (default 1) |
| `--pixel-ids` | Process only these pixels: comma-separated `tile:pixel_id`, or `pixel_id` to match it in every tile |
| `--plot-dir` | Save a summary plot per pixel in this directory |

### Plotting a single pixel

```bash
python process_parquet_wavelets_slope.py \
  --input dataset/samples.parquet --region arg --params '<ARG params>' \
  --output output/pixel.csv \
  --pixel-ids output_slope_energy_MOD13Q1_EVI_0000000000-0000000000:920_682 \
  --plot-dir output/plots
```

The plot has three panels: original and interpolated EVI with the seasonal
peaks of each agricultural year; `lt_evi` and its trend; and the detrend
(`lt_evi`, long-term precipitation and temperature, and the residual, each
with its trend line).

A VS Code debug configuration for this run is in `.vscode/launch.json`.

## Parameters

| Key | Description |
|---|---|
| `level` | Wavelet decomposition level |
| `start_doy_agro` | Day of year on which the agricultural year starts |
| `first_date`, `last_date` | Analysis period (`YYYY-MM-DD`) |
| `nan_values` | No-data value(s) of the source EVI |
| `min_evi_threshold`, `max_evi_threshold` | Cropland check thresholds on the mean seasonal minimum / maximum (EVI ×10,000) |
| `summer_crop_season` | `{min, max}` day-of-year window of the summer crop season, used for `lt_evi_slope_evi_summer_mean`. If `min > max`, the window wraps around the new year |
| `summer_peak_modis_dates` | Required by the parameter validation; not used by the current algorithm |
| `new_period` | Must be `0`: use the native MODIS dates as the analysis grid |
| `satellite` | Must be `modis` |
| `detrend_model` | `gam_current` (default) or `linear_regression` |

Parameters used for the sample dataset:

| Key | ARG | USA |
|---|---|---|
| `level` | 5 | 5 |
| `start_doy_agro` | 183 | 1 |
| `first_date` | 2015-07-01 | 2016-01-01 |
| `last_date` | 2025-12-31 | 2025-12-31 |
| `nan_values` | -10000 | -10000 |
| `min_evi_threshold` | 2600 | 2600 |
| `max_evi_threshold` | 5700 | 5500 |
| `summer_crop_season` | `{"min": 289, "max": 97}` | `{"min": 152, "max": 243}` |
| `summer_peak_modis_dates` | `{"min": 337, "max": 81}` | `{"min": 152, "max": 243}` |
| `new_period` | 0 | 0 |
| `satellite` | modis | modis |
| `detrend_model` | gam_current | gam_current |

USA as JSON:

```json
{"level": 5, "start_doy_agro": 1, "first_date": "2016-01-01", "last_date": "2025-12-31",
 "nan_values": -10000, "min_evi_threshold": 2600, "max_evi_threshold": 5500,
 "summer_crop_season": {"min": 152, "max": 243}, "summer_peak_modis_dates": {"min": 152, "max": 243},
 "new_period": 0, "satellite": "modis", "detrend_model": "gam_current"}
```

## Output

One row per pixel: `region`, `tile`, `pixel_id`, `row`, `col`, `lat`, `lon`,
plus:

| Column | Description |
|---|---|
| `lt_evi_slope` | Sen's slope of the long-term EVI component, per day, ×100 |
| `is_agro` | 1 if the pixel passed the cropland check, 0 otherwise |
| `evi_mean` | Mean interpolated EVI (0–1 scale) |
| `lt_evi_slope_evi_summer_mean` | `lt_evi_slope` × mean EVI within `summer_crop_season` (0–1 scale) |
| `pp_slope` | Sen's slope of the long-term precipitation component, per day, ×100 |
| `temp_slope` | Sen's slope of the long-term temperature component, per day, ×100 |
| `residuals_slope` | Sen's slope of the detrend residuals, per day, ×100 |
| `residuals_mean` | Mean of the detrend residuals |
| `temp_coef`, `pp_coef` | Linear coefficients of temperature and precipitation (`linear_regression` only) |

Values that could not be computed (for example, all trend columns of a
non-cropland pixel, or the coefficients under `gam_current`) are `NaN`.
