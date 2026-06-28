"""
ERA5-Land Daily Tmax & Tmin Download Script
For St. John's, NL — 1985 to 2024

Downloads two files:
  era5_tmax_stjohns.nc  — daily maximum 2m temperature
  era5_tmin_stjohns.nc  — daily minimum 2m temperature
"""

import cdsapi
import xarray as xr
from pathlib import Path

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

# Bounding box around St. John's, NL
# Format: [lat_max, lon_min, lat_min, lon_max]  (North, West, South, East)
AREA = [47.8, -53.5, 47.2, -52.5]

YEAR_START = 1985
YEAR_END = 2024

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "derived-era5-land-daily-statistics"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────


def all_days():
    return [f"{d:02d}" for d in range(1, 32)]


def all_months():
    return [f"{m:02d}" for m in range(1, 13)]

# ─────────────────────────────────────────────
# DOWNLOAD — one year at a time to avoid timeouts
# ─────────────────────────────────────────────


client = cdsapi.Client()

for statistic, label, outfile in [
    ("daily_maximum", "Tmax", "era5_tmax_stjohns.nc"),
    ("daily_minimum", "Tmin", "era5_tmin_stjohns.nc"),
]:
    merged_path = OUTPUT_DIR / outfile
    if merged_path.exists():
        print(f"\n{label}: {merged_path} already exists, skipping.")
        continue

    yearly_files = []

    for year in range(YEAR_START, YEAR_END + 1):
        outpath = OUTPUT_DIR / f"era5_{label.lower()}_{year}_tmp.nc"
        yearly_files.append(outpath)

        if outpath.exists():
            print(f"  {year} {label} already downloaded, skipping.")
            continue

        print(f"Downloading {label} for {year}...")
        request = {
            "variable":        "2m_temperature",
            "year":            str(year),
            "month":           all_months(),
            "day":             all_days(),
            "daily_statistic": statistic,
            "time_zone":       "utc+00:00",
            "frequency":       "1_hourly",
            "area":            AREA,
            "format":          "netcdf",
        }

        client.retrieve(DATASET, request, str(outpath))
        print(f"  Saved: {outpath}")

    # Merge yearly files into one using xarray
    print(f"\nMerging {label} yearly files into {outfile}...")
    ds_list = [xr.open_dataset(f) for f in yearly_files if f.exists()]
    if ds_list:
        merged = xr.concat(ds_list, dim="time").sortby("time")
        merged.to_netcdf(merged_path)
        print(f"  Saved: {merged_path}")

        # Clean up yearly temp files
        for f in yearly_files:
            if f.exists():
                f.unlink()
    else:
        print(f"  WARNING: No files to merge for {label}")

print("\nAll done!")
print("Output files:")
print("  era5_tmax_stjohns.nc")
print("  era5_tmin_stjohns.nc")
print("\nThese are ready to use directly with heatwave_detection.py")
print("Set TMAX_FILE = 'era5_tmax_stjohns.nc' \
      and TMIN_FILE = 'era5_tmin_stjohns.nc'")
print("Variable name will be 't2m' — update TMAX_VAR and TMIN_VAR accordingly")
