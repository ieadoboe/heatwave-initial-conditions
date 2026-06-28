"""
Extend the St. John's daily Tmax/Tmin archive forward (2025 → latest available).

The existing merged files (data/era5_tmax_stjohns.nc, data/era5_tmin_stjohns.nc)
cover 1985–2024 in a "block-diagonal" layout: t2m(time=40, valid_time=14610, …),
where each yearly `time` row is finite only on its own days. download_era5_tmax_tmin.py
SKIPS when the merged file exists, so it cannot append. This script does:

  1. download only the NEW months (default 2025-01 → today) from CDS, one month
     per request, into data/_era5_extend/ (restartable: existing temp files skip);
  2. collapse the existing merged file to a clean valid_time-only series and
     concat the new months onto it (dedup + sort by valid_time);
  3. back up the original merged file once (→ .bak) and write the extended file.

The clean valid_time layout is still read correctly by identify_heatwaves.py's
load_available (it collapses any extra axis on load).

ERA5-Land daily statistics lag real time by ~2–3 months, so the most recent
months will not exist yet — the script tries chronologically and stops at the
first month CDS can't serve. Re-run later to pick up newly published months.

Requires: cdsapi + ~/.cdsapirc with your CDS API key. Hits the CDS queue —
run it yourself (each month is usually a few minutes in the queue).

Usage:
  python scripts/extend_era5_daily.py                 # 2025-01 → today
  python scripts/extend_era5_daily.py --start 2025-01 --end 2026-03
"""

import argparse
import datetime as dt
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

# ─────────────────────────────────────────────
# SETTINGS (match download_era5_tmax_tmin.py)
# ─────────────────────────────────────────────

AREA = [47.8, -53.5, 47.2, -52.5]          # [N, W, S, E] around St. John's, NL
DATASET = "derived-era5-land-daily-statistics"

STATS = [
    ("daily_maximum", "tmax", "era5_tmax_stjohns.nc"),
    ("daily_minimum", "tmin", "era5_tmin_stjohns.nc"),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TMP_DIR = DATA_DIR / "_era5_extend"

KEEP_DIMS = {"valid_time", "latitude", "longitude", "number"}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def all_days():
    return [f"{d:02d}" for d in range(1, 32)]


def parse_ym(s: str) -> tuple[int, int]:
    y, m = s.split("-")
    return int(y), int(m)


def month_range(start: tuple[int, int], end: tuple[int, int]):
    """Inclusive chronological list of (year, month) from start to end."""
    y, m = start
    out = []
    while (y, m) <= end:
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def download_month(client, statistic, label, year, month) -> Path:
    """Download one month of a daily statistic to a temp file; skip if present."""
    out = TMP_DIR / f"era5_{label}_{year:04d}-{month:02d}.nc"
    if out.exists() and out.stat().st_size > 0:
        print(f"    {year:04d}-{month:02d} {label}: temp exists, skip")
        return out
    print(f"    {year:04d}-{month:02d} {label}: requesting …")
    client.retrieve(
        DATASET,
        {
            "variable": "2m_temperature",
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": all_days(),
            "daily_statistic": statistic,
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": AREA,
            "format": "netcdf",
        },
        str(out),
    )
    return out


def to_daily_series(path: Path) -> xr.DataArray:
    """Open a t2m file and return a clean DataArray with dims
    (valid_time, latitude, longitude), collapsing any spurious extra axis
    (e.g. the merged file's per-year `time`). Loaded into memory so the file
    handle is released before we overwrite it."""
    with xr.open_dataset(path) as ds:
        da = ds["t2m"]
        if "valid_time" not in da.dims and "time" in da.dims:
            da = da.rename({"time": "valid_time"})
        extra = [d for d in da.dims if d not in KEEP_DIMS]
        if extra:
            da = da.mean(dim=extra, skipna=True)   # one finite value per day
        return da.load()


def dedup_sort(da: xr.DataArray) -> xr.DataArray:
    da = da.sortby("valid_time")
    vt = da["valid_time"].values
    _, first_idx = np.unique(vt, return_index=True)
    return da.isel(valid_time=np.sort(first_idx))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────


def main(start: tuple[int, int], end: tuple[int, int]) -> None:
    import cdsapi  # imported here so the file is inspectable without cdsapi

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    todo = month_range(start, end)
    print(f"Target months: {start[0]}-{start[1]:02d} → {end[0]}-{end[1]:02d} "
          f"({len(todo)} months). ERA5-Land lags ~2–3 months; the run will stop "
          f"at the first month CDS can't serve.\n")

    for statistic, label, outfile in STATS:
        merged_path = DATA_DIR / outfile
        bak_path = merged_path.with_suffix(merged_path.suffix + ".bak")
        print(f"── {label.upper()}  ({outfile}) "
              f"{'─' * max(0, 40 - len(outfile))}")

        if not merged_path.exists():
            print(f"  {merged_path} missing — run download_era5_tmax_tmin.py "
                  f"first to build the 1985–2024 base. Skipping {label}.")
            continue

        # 1. download new months, chronologically, stopping at the first gap.
        new_files: list[Path] = []
        for year, month in todo:
            try:
                new_files.append(
                    download_month(client, statistic, label, year, month))
            except Exception as e:  # noqa: BLE001
                print(f"    {year:04d}-{month:02d} {label}: not available / "
                      f"failed ({type(e).__name__}: {e}). Stopping forward scan.")
                # Drop any half-written file so a rerun retries it.
                p = TMP_DIR / f"era5_{label}_{year:04d}-{month:02d}.nc"
                if p.exists() and p.stat().st_size == 0:
                    p.unlink()
                break

        if not new_files:
            print(f"  No new {label} months retrieved — nothing to append.\n")
            continue

        # 2. collapse existing + new to a clean valid_time series and merge.
        existing = to_daily_series(merged_path)
        n_before = existing.sizes["valid_time"]
        pieces = [existing] + [to_daily_series(p) for p in new_files]
        combined = dedup_sort(xr.concat(pieces, dim="valid_time"))
        combined.attrs = existing.attrs
        n_after = combined.sizes["valid_time"]

        if n_after <= n_before:
            print(f"  No new days beyond {str(existing.valid_time.max().values)[:10]}; "
                  f"file left unchanged.\n")
            continue

        # 3. back up the true original once, then write the extended file.
        if not bak_path.exists():
            shutil.copy2(merged_path, bak_path)
            print(f"  Backed up original → {bak_path.name}")
        combined.to_dataset(name="t2m").to_netcdf(merged_path)
        print(f"  {label}: {n_before} → {n_after} days "
              f"(now …{str(combined.valid_time.max().values)[:10]}). "
              f"Wrote {merged_path}\n")

    print("Done. Re-run later to pick up newly published months. "
          "Temp files kept in data/_era5_extend/ (safe to delete).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2025-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default=None,
                    help="last month, YYYY-MM (default: current month)")
    args = ap.parse_args()

    start = parse_ym(args.start)
    if args.end:
        end = parse_ym(args.end)
    else:
        today = dt.date.today()
        end = (today.year, today.month)

    if start > end:
        sys.exit(f"start {start} is after end {end}")
    main(start, end)
