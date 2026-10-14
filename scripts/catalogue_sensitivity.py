"""How much of the Europe 2026 catalogue survives other detection settings.

    python scripts/catalogue_sensitivity.py
    python scripts/catalogue_sensitivity.py --top 8 --skip-runs

Reruns scripts/detect_heatwaves_europe.py for every combination of minimum
area (25k, 50k, 100k km2) and percentile (90th, 95th), then matches the top
events of the reference catalogue, data/heatwave_events_europe_<year>.csv,
into each run. Event ids do not carry across runs (the labeller numbers blobs
in scan order), so events are matched by overlapping dates and the nearest
centroid instead.

Outputs (project convention):
  data/heatwave_events_europe_<year>_a<area>k_p<pct>.csv   one per setting
  data/catalogue_sensitivity_europe_<year>.csv             the comparison
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HEATWAVE_ROOT", REPO))
DATA = ROOT / "data"

AREAS_KM2 = (25_000, 50_000, 100_000)
PERCENTILES = (0.90, 0.95)


def tag_for(area_km2: float, pct: float) -> str:
    return f"_a{int(area_km2 / 1000)}k_p{int(round(pct * 100))}"


def match_events(ref: pd.DataFrame, run: pd.DataFrame,
                 max_dist_deg: float = 10.0) -> pd.Series:
    """For each reference event, the index in `run` of its counterpart, or -1.

    A counterpart must overlap the reference in time. Among those, the one
    with the most overlapping days wins, nearest centroid as tie-break, and
    it must lie within max_dist_deg of the reference centroid. Frames must
    carry start, end, centroid_lat and centroid_lon; dates as datetimes.

    ponytail: date-plus-centroid matching, not cell-day overlap, so a run
    that splits one reference event into two reports only the larger half.
    Swap in overlap of the label grids if that ever decides a result."""
    out = []
    for _, e in ref.iterrows():
        overlap = ((np.minimum(run.end, e.end) - np.maximum(run.start, e.start))
                   .dt.days + 1).clip(lower=0)
        dlat = run.centroid_lat - e.centroid_lat
        dlon = (run.centroid_lon - e.centroid_lon) * np.cos(np.radians(e.centroid_lat))
        dist = np.hypot(dlat, dlon)
        ok = (overlap > 0) & (dist <= max_dist_deg)
        if not ok.any():
            out.append(-1)
            continue
        score = pd.DataFrame({"overlap": overlap[ok], "dist": dist[ok]})
        best = score.sort_values(["overlap", "dist"],
                                 ascending=[False, True]).index[0]
        out.append(int(best))
    return pd.Series(out, index=ref.index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--top", type=int, default=6,
                        help="How many reference events to track (default 6)")
    parser.add_argument("--skip-runs", action="store_true",
                        help="Only rebuild the comparison from existing CSVs")
    args = parser.parse_args()

    dates = ["start", "peak", "end"]
    ref_path = DATA / f"heatwave_events_europe_{args.year}.csv"
    if not ref_path.exists():
        sys.exit(f"missing {ref_path}; run scripts/detect_heatwaves_europe.py")
    ref = pd.read_csv(ref_path, parse_dates=dates).head(args.top)

    rows = []
    for pct in PERCENTILES:
        for area in AREAS_KM2:
            tag = tag_for(area, pct)
            path = DATA / f"heatwave_events_europe_{args.year}{tag}.csv"
            if not path.exists() and not args.skip_runs:
                print(f"\n=== min-area {area:,} km2, percentile {pct:.0%} ===")
                subprocess.run(
                    [sys.executable, str(REPO / "scripts" / "detect_heatwaves_europe.py"),
                     "--year", str(args.year), "--min-area", str(area),
                     "--percentile", str(pct), "--tag", tag],
                    check=True, env=os.environ)
            if not path.exists():
                print(f"skipping {path.name}: not found")
                continue
            run = pd.read_csv(path, parse_dates=dates)
            hit = match_events(ref, run)
            for (i, e), j in zip(ref.iterrows(), hit):
                m = run.loc[j] if j >= 0 else None
                rows.append({
                    "min_area_km2": area, "percentile": pct,
                    "n_events": len(run), "ref_rank": int(e["rank"]),
                    "survives": j >= 0,
                    "rank": int(m["rank"]) if m is not None else np.nan,
                    "start": m.start.date() if m is not None else pd.NaT,
                    "end": m.end.date() if m is not None else pd.NaT,
                    "duration_days": m.duration_days if m is not None else np.nan,
                    "peak_area_km2": m.peak_area_km2 if m is not None else np.nan,
                    "severity_C_km2_day": (m.severity_C_km2_day
                                           if m is not None else np.nan),
                })

    table = pd.DataFrame(rows)
    out = DATA / f"catalogue_sensitivity_europe_{args.year}.csv"
    table.to_csv(out, index=False)
    print(f"\n-> {out}\n")

    counts = table.groupby(["percentile", "min_area_km2"]).n_events.first()
    print("events per setting:\n" + counts.to_string() + "\n")
    label = table.apply(
        lambda r: (f"{int(r['rank'])} ({int(r.duration_days)} d)"
                   if r.survives else "-"), axis=1)
    pivot = (table.assign(cell=label)
             .pivot(index="ref_rank", columns=["percentile", "min_area_km2"],
                    values="cell"))
    print("reference rank -> rank (duration) under each setting:\n"
          + pivot.to_string())


if __name__ == "__main__":
    main()
