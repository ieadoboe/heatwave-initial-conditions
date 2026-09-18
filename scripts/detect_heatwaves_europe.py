"""Catalogue the 2026 European heat waves from ERA5 daily maxima.

    python scripts/detect_heatwaves_europe.py
    python scripts/detect_heatwaves_europe.py --year 2025 --min-area 100000

Needs the downloads from scripts/download_era5_europe_daily.py. Applies
Perkins & Alexander CTX90pct against a 1991-2020 calendar-day baseline, labels
the connected space-time blobs, and writes the ranked catalogue plus a map.

Runs on a laptop: numpy, pandas, xarray, scipy and matplotlib only, no jax.

Outputs (project convention: data/ and plots/):
  data/heatwave_events_europe_<year>.csv
  data/heatwave_threshold_europe.nc      cached percentile climatology
  plots/heatwave_europe_<year>.pdf
"""

import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402


def _submodule(name):
    """Import one heatwave_ic submodule without running the package __init__.

    That __init__ eagerly imports model.py and data.py, which pull in jax,
    neuralgcm and dinosaur. Detection needs none of them: numpy, pandas,
    xarray, scipy and matplotlib are enough, so this runs on a laptop with no
    GPU stack installed."""
    if "heatwave_ic" not in sys.modules:
        pkg = types.ModuleType("heatwave_ic")
        pkg.__path__ = [str(REPO / "heatwave_ic")]
        sys.modules["heatwave_ic"] = pkg
    full = f"heatwave_ic.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, REPO / "heatwave_ic" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod


D = _submodule("detect")


def footprint_koppen(det, event_id: int, zones) -> str:
    """Most common land Koppen class among the cells an event ever covered."""
    from collections import Counter
    hit = det.labels == event_id
    _, ii, jj = np.nonzero(hit)
    cells = set(zip(ii.tolist(), jj.tolist()))
    counts = Counter(zones.koppen_class(det.lat[i], det.lon[j])
                     for i, j in cells)
    counts.pop("ocean", None)
    return counts.most_common(1)[0][0] if counts else "ocean"

# HEATWAVE_ROOT re-roots outputs, as elsewhere in this project.
ROOT = Path(os.environ.get("HEATWAVE_ROOT", REPO))
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
ERA5_DIR = DATA / "era5_europe"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026,
                        help="Year to catalogue (default 2026)")
    parser.add_argument("--baseline", nargs=2, type=int, default=[1991, 2020],
                        metavar=("FIRST", "LAST"))
    parser.add_argument("--percentile", type=float, default=0.90)
    parser.add_argument("--min-days", type=int, default=3,
                        help="Consecutive days to qualify (CTX90pct: 3)")
    parser.add_argument("--min-area", type=float, default=50_000.0,
                        help="Minimum extent in km2 (default 50000, about "
                             "half of Portugal). Applied per day before "
                             "labelling, so scattered exceedances cannot "
                             "bridge separate events, and again to each "
                             "event's peak extent.")
    parser.add_argument("--data-dir", default=str(ERA5_DIR))
    parser.add_argument("--rebuild-threshold", action="store_true",
                        help="Recompute the percentile climatology even if the "
                             "cached one exists")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    first, last = args.baseline
    target_path = data_dir / f"europe_tmax_{args.year}.nc"
    if not target_path.exists():
        sys.exit(f"missing {target_path}\n"
                 f"run: python scripts/download_era5_europe_daily.py "
                 f"--years {args.year}")

    print(f"Target year {args.year}: {target_path.name}")
    target = D.open_daily_tmax([target_path])
    print(f"  {target.sizes['time']} days, "
          f"{target.sizes['latitude']}x{target.sizes['longitude']} grid, "
          f"{float(target.min()):.1f} to {float(target.max()):.1f} C")

    thresh_path = DATA / (f"heatwave_threshold_europe_"
                          f"{first}_{last}_p{int(args.percentile * 100)}.nc")
    if thresh_path.exists() and not args.rebuild_threshold:
        print(f"\nThreshold: reusing {thresh_path.name}")
        threshold = xr.open_dataarray(thresh_path).load()
    else:
        baseline_paths = [data_dir / f"europe_tmax_{y}.nc"
                          for y in range(first, last + 1)]
        missing = [p.name for p in baseline_paths if not p.exists()]
        if missing:
            sys.exit(f"missing {len(missing)} baseline year(s): "
                     f"{', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}\n"
                     f"run: python scripts/download_era5_europe_daily.py")
        print(f"\nBaseline {first}-{last}: reading {len(baseline_paths)} files ...")
        baseline = D.open_daily_tmax(baseline_paths)
        print(f"  {baseline.sizes['time']} days "
              f"({baseline.nbytes / 1e6:.0f} MB in memory)")
        print(f"  computing the calendar-day {args.percentile:.0%} percentile "
              f"on a +/-7 day window ...")
        threshold = D.calendar_day_percentile(baseline, q=args.percentile)
        threshold.to_netcdf(thresh_path)
        print(f"  cached -> {thresh_path}")
        del baseline

    print("\nDetecting ...")
    land = D.land_mask_for(target["latitude"].values, target["longitude"].values)
    print(f"  land cells: {land.sum()} of {land.size} "
          f"({100 * land.mean():.0f}%)")
    det = D.detect(target, threshold, land_mask=land,
                   min_days=args.min_days, min_area_km2=args.min_area)
    print(f"  {det.mask.sum()} cell-days in a spell of >= {args.min_days} days")
    print(f"  {det.n_events} connected blobs before the area filter")

    events = D.event_table(det, target, min_area_km2=args.min_area)
    if events.empty:
        sys.exit(f"\nNo events above {args.min_area:,.0f} km2. "
                 f"Lower --min-area to see smaller ones.")

    # Tag each event with the modal Koppen class over its land footprint. The
    # centroid alone is wrong for an event wrapped around a coast: one
    # spanning Italy and the Balkans has its centroid in the Adriatic.
    zones = _submodule("zones")
    events["koppen"] = [
        footprint_koppen(det, int(eid), zones) for eid in events.event_id]
    events["zone"] = [zones.GROUP_NAMES.get(zones.koppen_group(k), "")
                      for k in events.koppen]

    DATA.mkdir(parents=True, exist_ok=True)
    out_csv = DATA / f"heatwave_events_europe_{args.year}.csv"
    events.to_csv(out_csv, index=False)
    print(f"\n{len(events)} events above {args.min_area:,.0f} km2 "
          f"-> {out_csv}\n")
    cols = ["rank", "start", "peak", "end", "duration_days", "peak_area_km2",
            "centroid_lat", "centroid_lon", "peak_tmax_C", "max_anom_C",
            "koppen", "severity_C_km2_day"]
    print(events[cols].head(15).to_string(index=False))

    hw_days = (det.labels > 0).sum(axis=0)
    plots = _submodule("plots")
    fig, _ = plots.plot_europe_heatwaves(
        hw_days, det.lat, det.lon, events=events, year=args.year,
        save=PLOTS / f"heatwave_europe_{args.year}.pdf")
    fig.savefig(PLOTS / f"heatwave_europe_{args.year}.png", dpi=150,
                bbox_inches="tight")


if __name__ == "__main__":
    main()
