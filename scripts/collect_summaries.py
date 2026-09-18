"""Combine the per-event summary tables a SLURM array leaves behind.

    python scripts/collect_summaries.py

Each array task writes its own data/atlas_summary_<event>.csv and
data/ensemble_summary_<event>.csv, because seven tasks running at once would
otherwise overwrite each other's rows in one shared file. This concatenates
them in atlas order, writes data/atlas_summary.csv and
data/ensemble_summary.csv, and merges the ensemble metrics into the atlas
table so one file carries the storyline gain and the natural-variability
yardstick for every zone.

Safe to re-run: it reads only the per-event files and rewrites the combined
ones. Pass --persist-dir to copy the two combined tables somewhere durable.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

# Inlined rather than imported from heatwave_ic.config, so this stays a
# pandas-only script that also runs on a laptop without jax installed.
DATA_DIR = os.path.join(os.environ.get("HEATWAVE_ROOT", ""), "data")

ATLAS_ORDER = ["pnw_jun2021", "stjohns_aug2025", "moscow_jul2010",
               "japan_jul2018", "sahel_apr2024", "brazil_nov2023",
               "siberia_jun2020"]

METRIC_COLS = ["n_members", "ens_mean_peak_C", "ens_max_peak_C",
               "ens_peak_spread_C", "gain_vs_ens_max_C", "gain_over_spread"]


def _combine(data_dir: Path, stem: str) -> pd.DataFrame | None:
    """Concatenate data_dir/<stem>_<event>.csv in atlas order."""
    frames = []
    for event in ATLAS_ORDER:
        path = data_dir / f"{stem}_{event}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
        else:
            print(f"  missing: {path.name}")
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    print(f"  combined {len(frames)} file(s) -> {len(df)} row(s)")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist-dir", default=None,
                        help="Also copy the combined tables here")
    args = parser.parse_args()

    data_dir = Path(DATA_DIR)
    print(f"Reading per-event summaries from {data_dir.resolve()}")

    print("\natlas:")
    atlas = _combine(data_dir, "atlas_summary")
    print("\nensembles:")
    ens = _combine(data_dir, "ensemble_summary")

    if atlas is None and ens is None:
        sys.exit("No per-event summary files found. Did the arrays run?")

    written = []
    if ens is not None:
        out = data_dir / "ensemble_summary.csv"
        ens.to_csv(out, index=False)
        written.append(out)
    if atlas is not None:
        if ens is not None and "event" in ens.columns:
            atlas = atlas.drop(
                columns=[c for c in METRIC_COLS if c in atlas.columns])
            metrics = ens[["event"] + [c for c in METRIC_COLS if c in ens.columns]]
            atlas = atlas.merge(metrics, on="event", how="left")
        out = data_dir / "atlas_summary.csv"
        atlas.to_csv(out, index=False)
        written.append(out)
        print(f"\n{atlas.to_string(index=False)}")

    for path in written:
        print(f"\nwrote {path}")
        if args.persist_dir:
            dest = Path(args.persist_dir)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest / path.name)
            print(f"  copied to {dest / path.name}")


if __name__ == "__main__":
    main()
