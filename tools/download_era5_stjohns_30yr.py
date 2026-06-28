"""
Download 30 years of hourly ERA5 pressure-level data for the St. John's region,
same domain/variables/levels as download_era5_stjohns.py, but spanning a long
climatological window.

Strategy:
  - One CDS request per (year, month). Keeps each request well under CDS's
    ~120k-fields-per-request soft cap (5 vars * 4 levels * 31 days * 24h ≈ 15k).
  - Files saved to ./era5_30yr/era5_stjohns_pl_YYYY-MM.nc
  - Restartable: months whose output file already exists are skipped.
  - Per-month try/except so one bad month doesn't kill the whole archive.

Requirements:
  pip install cdsapi
  ~/.cdsapirc must contain your CDS API key.

Typical runtime: each monthly request usually takes a few minutes in the CDS
queue. 360 months ⇒ expect this to span hours/days depending on CDS load.
Run it in a tmux/screen on a machine with reliable network.

Usage:
  python download_era5_stjohns_30yr.py               # 1995-2024
  python download_era5_stjohns_30yr.py 2000 2010     # custom inclusive range
"""

import os
import sys
import time
import traceback

import cdsapi

# ----- Region (matches download_era5_stjohns.py) -----
anchor_lat = 49.75
anchor_lon_360 = 305.5
grid_points = 16
res = 0.25

anchor_lon = ((anchor_lon_360 + 180) % 360) - 180  # -54.5
north = anchor_lat
south = anchor_lat - grid_points * res             # 45.75
west = anchor_lon                                  # -54.5
east = anchor_lon + grid_points * res              # -50.5

# ----- Variables / levels -----
VARIABLES = [
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "geopotential",
]
PRESSURE_LEVELS = ["1000", "850", "700", "500"]

# ----- Time window -----
DEFAULT_START_YEAR = 1995
DEFAULT_END_YEAR = 2024  # inclusive; 30 years total

OUT_DIR = "era5_30yr"


def request_month(client: cdsapi.Client, year: int, month: int, out_path: str) -> None:
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": VARIABLES,
            "pressure_level": PRESSURE_LEVELS,
            "year": f"{year:04d}",
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "area": [north, west, south, east],
            "format": "netcdf",
        },
        out_path,
    )


def main(start_year: int, end_year: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Region: N={north}, W={west}, S={south}, E={east}")
    print(f"Years: {start_year}-{end_year} (inclusive), monthly chunks")
    print(f"Output: ./{OUT_DIR}/")

    client = cdsapi.Client()
    failures: list[tuple[int, int, str]] = []
    total = (end_year - start_year + 1) * 12
    done = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            done += 1
            out_path = os.path.join(OUT_DIR, f"era5_stjohns_pl_{year:04d}-{month:02d}.nc")
            tag = f"[{done:>3}/{total}] {year:04d}-{month:02d}"

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"{tag}  skip (exists)")
                continue

            t0 = time.time()
            try:
                request_month(client, year, month, out_path)
                dt = time.time() - t0
                size_mb = os.path.getsize(out_path) / 1e6
                print(f"{tag}  ok  ({size_mb:.1f} MB, {dt:.0f}s)")
            except Exception as e:  # noqa: BLE001
                # Remove a half-written file so the rerun can re-request it.
                if os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                msg = f"{type(e).__name__}: {e}"
                print(f"{tag}  FAIL  {msg}")
                traceback.print_exc()
                failures.append((year, month, msg))

    print()
    print(f"Done. {total - len(failures)}/{total} months retrieved.")
    if failures:
        print(f"{len(failures)} months failed -- rerun the script to retry:")
        for y, m, msg in failures:
            print(f"  {y:04d}-{m:02d}  {msg}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        s, e = DEFAULT_START_YEAR, DEFAULT_END_YEAR
    elif len(sys.argv) == 3:
        s, e = int(sys.argv[1]), int(sys.argv[2])
    else:
        print("usage: python download_era5_stjohns_30yr.py [start_year end_year]")
        sys.exit(2)
    main(s, e)
