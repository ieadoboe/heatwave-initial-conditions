"""Download daily-maximum 2 m temperature over Europe for heat-wave detection.

    python scripts/download_era5_europe_daily.py --probe    # validate the request
    python scripts/download_era5_europe_daily.py            # baseline + target
    python scripts/download_era5_europe_daily.py --years 2026

Uses the CDS `derived-era5-single-levels-daily-statistics` catalogue entry,
which aggregates the hourly reanalysis to a daily statistic SERVER-SIDE and
subsets by area. That matters: the equivalent from ARCO-ERA5 would mean
reading ~110,000 global hourly fields, because its zarr is chunked one hour
per chunk, so a Europe-only request still transfers the whole globe.

Years are requested in batches (5 at a time by default) and split into one
netCDF per year in data/era5_europe/, about 25 MB each, so the full 1991-2020
baseline plus 2026 is under a gigabyte. Batching is what makes this practical:
the CDS queue wait dominates the transfer, and a request observed on
2026-09-09 waited 93 minutes to run for 4. Existing years are skipped, so an
interrupted run resumes and costs nothing.

Run --probe first. It asks for a single day and fails in seconds if the
request schema is wrong, instead of after the real requests have queued.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DATASET = "derived-era5-single-levels-daily-statistics"

# [North, West, South, East] — the conventional European domain.
AREA = [72, -25, 35, 45]

BASELINE_YEARS = range(1991, 2021)   # WMO 1991-2020 normal
TARGET_YEARS = [2026]
MONTHS = ["05", "06", "07", "08", "09"]

OUT_DIR = REPO / "data" / "era5_europe"


def request_for(years, months=None, days=None) -> dict:
    """The CDS request for one or more years of daily-maximum 2 m temperature."""
    if isinstance(years, int):
        years = [years]
    return {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": [str(y) for y in years],
        "month": months or MONTHS,
        "day": days or [f"{d:02d}" for d in range(1, 32)],
        "daily_statistic": "daily_maximum",
        "time_zone": "utc+00:00",
        "frequency": "1_hourly",
        "area": AREA,
        "data_format": "netcdf",
    }


def consecutive_runs(years):
    """Split a year list into runs of consecutive years.

    CDS warns that "selection of non-consecutive dates is significantly
    slower" for the daily-statistics entries, so a batch must never straddle
    a gap. The default selection has one: the 1991-2020 baseline and the 2026
    target, which naively chunked would pair 2020 with 2026."""
    runs, current = [], []
    for y in sorted(years):
        if current and y != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(y)
    if current:
        runs.append(current)
    return runs


def _is_too_large(exc: Exception) -> bool:
    """True when CDS refused a request for being over its cost limit.

    That refusal is immediate rather than queued, so probing for the largest
    acceptable batch is cheap. Matched on the message because the client
    raises a plain HTTPError for it."""
    text = str(exc).lower()
    return ("too large" in text or "cost limit" in text
            or "reduce your selection" in text)


def year_path(out_dir: Path, year: int) -> Path:
    return out_dir / f"europe_tmax_{year}.nc"


def download_batch(client, years, out_dir: Path) -> None:
    """Fetch several years in ONE request, then split into per-year files.

    Batching matters because the CDS queue wait dominates: a request observed
    on 2026-09-09 sat 93 minutes in the queue and then ran in 4. One request
    per year would mean 31 of those waits. The per-year split keeps the resume
    logic and the detection driver, which both address files by year."""
    import xarray as xr

    label = f"{years[0]}-{years[-1]}" if len(years) > 1 else str(years[0])
    print(f"  {label}: requesting {len(years)} year(s) in one request ...",
          flush=True)
    tmp = out_dir / f"_batch_{label}.nc"
    client.retrieve(DATASET, request_for(years), str(tmp))

    ds = xr.open_dataset(tmp)
    time_name = next(c for c in ("valid_time", "time") if c in ds.dims)
    for year in years:
        part = ds.sel({time_name: ds[time_name].dt.year == year})
        if part.sizes[time_name] == 0:
            print(f"    {year}: WARNING nothing returned")
            continue
        out = year_path(out_dir, year)
        part.to_netcdf(out)
        print(f"    {year}: {out.name} ({out.stat().st_size / 1e6:.0f} MB)",
              flush=True)
    ds.close()
    tmp.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="Years to fetch (default: 1991-2020 plus 2026)")
    parser.add_argument("--probe", action="store_true",
                        help="Fetch a single day to validate the request, "
                             "then stop")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-years", type=int, default=5,
                        help="Years per CDS request (default 5). The queue "
                             "wait dominates the transfer, so batching is "
                             "what makes this finish in hours not days.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import cdsapi
    client = cdsapi.Client()

    if args.probe:
        probe = out_dir / "_probe.nc"
        print(f"Probing {DATASET} with a single day ...")
        client.retrieve(DATASET, request_for(2026, months=["07"], days=["15"]),
                        str(probe))
        import xarray as xr
        ds = xr.open_dataset(probe)
        print(f"\nOK. Variables: {list(ds.data_vars)}")
        print(f"Dims: {dict(ds.sizes)}")
        print(f"Coords: {list(ds.coords)}")
        print(f"\nProbe file kept at {probe} — delete it when you are done.")
        return

    years = args.years if args.years else list(BASELINE_YEARS) + TARGET_YEARS
    if not args.overwrite:
        have = [y for y in years if year_path(out_dir, y).exists()]
        years = [y for y in years if y not in have]
        if have:
            print(f"Already downloaded: {len(have)} year(s) "
                  f"({have[0]}-{have[-1]})")
    if not years:
        print("Nothing left to download.")
        return

    batch = max(1, args.batch_years)
    print(f"Downloading {len(years)} year(s) of daily-max 2m temperature, "
          f"up to {batch} per request")
    print(f"  domain {AREA[0]}N {AREA[3]}E to {AREA[2]}N {AREA[1]}E, "
          f"months {'-'.join(MONTHS)}")
    print(f"  -> {out_dir}\n", flush=True)

    # CDS caps a request by "cost", and the cap sits somewhere below five
    # years of daily fields. Rather than hard-code a guess, halve on refusal
    # and keep the size that worked for the remaining batches.
    for run in consecutive_runs(years):
        i = 0
        while i < len(run):
            chunk = run[i:i + batch]
            try:
                download_batch(client, chunk, out_dir)
            except Exception as exc:
                if not _is_too_large(exc) or len(chunk) == 1:
                    raise
                batch = max(1, len(chunk) // 2)
                print(f"  refused as too large; retrying {batch} year(s) "
                      f"per request", flush=True)
                continue
            i += len(chunk)

    total = sum(f.stat().st_size for f in out_dir.glob("europe_tmax_*.nc"))
    print(f"\nDone. {total / 1e9:.2f} GB in {out_dir}")


if __name__ == "__main__":
    main()
