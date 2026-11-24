"""
Per-pixel long-term EVI trend (wavelets + Sen slope) and climate detrend
(PP/TEMP), computed from a Parquet file of time series.

Runs series_analysis() on each pixel, reading the series from a
long-format Parquet file and writing one row per pixel.

Input Parquet (e.g. dataset/samples.parquet), one row per pixel/series/date:
    region, tile, pixel_id, row, col, lat, lon   pixel identification
    series    evi_original | evi_interpolated | pp_raw | temp_raw_celsius
    x_days    days since the first evi_original date
    date      YYYY-MM-DD
    value     series value
    qa_ok     evi_original only: True if the observation passes the QA filter

  - evi_original:     raw EVI at the sensor's native dates.
  - evi_interpolated: defines the analysis time grid (x_p_new).
  - pp_raw:           ERA5-Land precipitation on the analysis grid (m/day).
  - temp_raw_celsius: ERA5-Land temperature on the analysis grid, in C.

Output (.parquet or .csv): one row per pixel with the identification
columns + OUTPUT_COLUMNS. Values that could not be computed are NaN.

With --plot-dir, it also saves the per-pixel analysis summary plot
(original + interpolated EVI, lt_evi + trend, detrend with PP/TEMP); only
for isAgro pixels, which are the ones that reach the trend analysis.
--pixel-ids restricts the run to a few points.

Examples:
    python process_parquet_wavelets_slope.py \\
        --input dataset/samples.parquet --region arg --detrend true \\
        --params '{"level": 5, "start_doy_agro": 183, ...}' \\
        --output tmp/slope_arg.parquet --workers 8

    # plot two points
    python process_parquet_wavelets_slope.py ... --output tmp/points.csv \\
        --pixel-ids output_slope_energy_MOD13Q1_EVI_0000000000-0000000000:858_703,899_775 \\
        --plot-dir tmp/plots
"""
import argparse
import datetime
import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display; plots are written to files

import numpy as np
import pandas as pd

from utils.series_analysis_20 import series_analysis

PIXEL_KEYS = ["region", "tile", "pixel_id", "row", "col", "lat", "lon"]

# Same order as the values returned by series_analysis(type_results="raster_slope").
OUTPUT_COLUMNS = [
    "lt_evi_slope",
    "is_agro",
    "evi_mean",
    "lt_evi_slope_evi_summer_mean",
    "pp_slope",
    "temp_slope",
    "residuals_slope",
    "residuals_mean",
    "temp_coef",
    "pp_coef",
]

REQUIRED_CONFIG_KEYS = (
    "level",
    "start_doy_agro",
    "first_date",
    "last_date",
    "nan_values",
    "min_evi_threshold",
    "max_evi_threshold",
    "summer_crop_season",
    "summer_peak_modis_dates",
    "new_period",
    "satellite",
)
ALLOWED_DETREND_MODELS = {"gam_current", "linear_regression"}

# series_analysis() flags "not computed" with -40000 (and -40001 when there is no EVI).
NO_DATA_THRESHOLD = -39000


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_params(value):
    try:
        params = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid JSON in --params: {exc}") from exc
    if not isinstance(params, dict):
        raise argparse.ArgumentTypeError("--params must be a JSON object.")

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in params]
    unexpected = [k for k in params if k not in REQUIRED_CONFIG_KEYS and k != "detrend_model"]
    if missing or unexpected:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if unexpected:
            parts.append("unexpected or misspelled: " + ", ".join(unexpected))
        raise argparse.ArgumentTypeError("Error in --params: " + " | ".join(parts))

    params.setdefault("detrend_model", "gam_current")
    if params["detrend_model"] not in ALLOWED_DETREND_MODELS:
        raise argparse.ArgumentTypeError(
            "detrend_model must be one of: " + ", ".join(sorted(ALLOWED_DETREND_MODELS)))
    if params["satellite"] != "modis" or params["new_period"] != 0:
        # The PP/TEMP series in the Parquet are already on the native MODIS
        # grid (new_period=0); resampling to another period would change
        # the result with respect to the original raster run.
        raise argparse.ArgumentTypeError("This script only supports satellite='modis' with new_period=0.")
    return params


def to_yyyymmdd(date_str):
    return date_str.replace("-", "")


def summer_mask(dates_periodic, season):
    doy = np.array([datetime.datetime.strptime(d, "%Y%m%d").timetuple().tm_yday for d in dates_periodic])
    if season["max"] >= season["min"]:
        return (doy >= season["min"]) & (doy <= season["max"])
    return (doy <= season["min"]) | (doy >= season["max"])


def series_on_grid(pix, name, x_grid, offset=0.0):
    """Climate series aligned to x_grid by x_days (NaN where there is no data)."""
    s = pix[pix.series == name]
    out = np.full(len(x_grid), np.nan, dtype=np.float32)
    if s.empty:
        return None
    pos = pd.Series(np.arange(len(x_grid)), index=x_grid)
    idx = pos.reindex(s.x_days.to_numpy())
    ok = idx.notna().to_numpy()
    out[idx[ok].astype(int).to_numpy()] = s.value.to_numpy()[ok] + offset
    return out


def build_pixel_task(pix, config):
    """Builds the series_analysis() arguments for one pixel."""
    evi_o = pix[pix.series == "evi_original"].sort_values("x_days")
    evi_i = pix[pix.series == "evi_interpolated"].sort_values("x_days")

    x_old = evi_o.x_days.to_numpy()
    dates = [to_yyyymmdd(d) for d in evi_o.date]

    # Analysis grid: the evi_interpolated one, clipped to [first_date, last_date].
    first, last = to_yyyymmdd(config["first_date"]), to_yyyymmdd(config["last_date"])
    dates_periodic = np.array([to_yyyymmdd(d) for d in evi_i.date])
    keep = (dates_periodic >= first) & (dates_periodic <= last)
    x_p_new = evi_i.x_days.to_numpy()[keep]
    dates_periodic = dates_periodic[keep]

    run_config = dict(config)
    # new_period=0: native grid; get_slope() needs the actual spacing.
    run_config["new_period"] = float(np.median(np.diff(x_old)))

    # QA: series_analysis (MODIS) drops observations with (qa & 0b11) != 0.
    qa_ok = evi_o.qa_ok.fillna(False).to_numpy(dtype=bool)
    clouds = np.where(qa_ok, 0, 1).astype(np.int16)

    detrend = config["detrend"]
    return dict(
        x_old=x_old,
        x_p_new=x_p_new,
        dates=dates,
        dates_periodic=dates_periodic,
        evi=evi_o.value.to_numpy(dtype=np.float32),
        clouds=clouds,
        config=run_config,
        summer_dates_mask=summer_mask(dates_periodic, config["summer_crop_season"]),
        pp_periodic=series_on_grid(pix, "pp_raw", x_p_new) if detrend else None,
        temp_periodic=series_on_grid(pix, "temp_raw_celsius", x_p_new, offset=273.0) if detrend else None,
    )


def plot_stem(plot_dir, key):
    """Plot file prefix; series_analysis appends '_summary.jpg'."""
    tile = key["tile"].removeprefix("output_slope_energy_")
    return str(Path(plot_dir) / f"{key['region']}_{tile}_{key['pixel_id']}")


def process_pixel(task):
    key, inputs, plot_dir = task
    result = series_analysis(
        inputs["x_old"],
        inputs["x_p_new"],
        inputs["dates"],
        inputs["dates_periodic"],
        inputs["evi"],
        inputs["clouds"],
        inputs["config"],
        type_analysis=["slope"],
        type_results="raster_slope",
        x=key["row"],
        y=key["col"],
        summer_dates_mask=inputs["summer_dates_mask"],
        pp_periodic=inputs["pp_periodic"],
        temp_periodic=inputs["temp_periodic"],
        satellite=inputs["config"]["satellite"],
        plot_path=plot_stem(plot_dir, key) if plot_dir else None,
        location_label=f"{key['region'].upper()} {key['pixel_id']} - Lat {key['lat']:.5f}, Lon {key['lon']:.5f}",
    )
    values = [np.nan if v <= NO_DATA_THRESHOLD else float(v) for v in result]
    return {**key, **dict(zip(OUTPUT_COLUMNS, values))}


def parse_pixel_ids(value):
    """Comma-separated 'tile:pixel_id' or 'pixel_id' -> [(tile|None, pixel_id)]."""
    keys = []
    for item in value.split(","):
        tile, _, pixel_id = item.strip().rpartition(":")
        keys.append((tile or None, pixel_id))
    return keys


def load_input(path, region=None, pixel_ids=None):
    filters = [("region", "==", region)] if region else []
    if pixel_ids:
        filters.append(("pixel_id", "in", sorted({p for _, p in pixel_ids})))
    df = pd.read_parquet(path, filters=filters or None)
    if df.empty:
        raise SystemExit(f"No data in {path} for the requested filters.")
    if not region and df.region.nunique() > 1:
        raise SystemExit("The Parquet has several regions: pass --region (each one takes its own --params).")
    for c in ("region", "tile", "series"):
        df[c] = df[c].astype(str)

    if pixel_ids:
        # pixel_id (row_col) repeats across tiles: if a tile was given, filter by the pair.
        wanted = pd.Series(False, index=df.index)
        for tile, pixel_id in pixel_ids:
            match = df.pixel_id == pixel_id
            if tile:
                match &= df.tile == tile
            wanted |= match
        df = df[wanted]
        found = df[["tile", "pixel_id"]].drop_duplicates()
        print(f"--pixel-ids: {len(found)} pixels found")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Long-format Parquet of per-pixel time series.")
    ap.add_argument("--region", help="Region to process ('region' column).")
    ap.add_argument("--params", required=True, type=parse_params, help="JSON with the algorithm parameters.")
    ap.add_argument("--detrend", type=parse_bool, default=True, help="Detrend EVI with PP and TEMP.")
    ap.add_argument("--output", required=True, help="Output file (.parquet or .csv).")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--pixel-ids", type=parse_pixel_ids,
                    help="Process only these pixels: 'tile:pixel_id' or 'pixel_id' (all tiles), comma-separated.")
    ap.add_argument("--plot-dir", help="If given, saves the summary plot of each pixel in this directory.")
    args = ap.parse_args()

    config = dict(args.params, detrend=args.detrend)
    df = load_input(args.input, args.region, args.pixel_ids)
    if args.plot_dir:
        Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    tasks = []
    for values, pix in df.groupby(PIXEL_KEYS, sort=True):
        tasks.append((dict(zip(PIXEL_KEYS, values)), build_pixel_task(pix, config), args.plot_dir))
    print(f"Pixels to process: {len(tasks)}")

    if args.workers > 1:
        with Pool(args.workers) as pool:
            rows = pool.map(process_pixel, tasks, chunksize=max(1, len(tasks) // (args.workers * 4)))
    else:
        rows = [process_pixel(t) for t in tasks]

    out = pd.DataFrame(rows, columns=PIXEL_KEYS + OUTPUT_COLUMNS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".csv":
        out.to_csv(output, index=False)
    else:
        out.to_parquet(output, index=False)
    print(f"Output: {output} ({len(out)} pixels, {int(out.lt_evi_slope.notna().sum())} with trend)")
    if args.plot_dir:
        n_plots = sum(Path(f"{plot_stem(args.plot_dir, r)}_summary.jpg").exists() for r in rows)
        print(f"Plots: {n_plots} in {args.plot_dir}")


if __name__ == "__main__":
    main()
