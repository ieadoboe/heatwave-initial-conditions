"""How stable is the top of the catalogue, summer by summer, 1991-2026?

    python scripts/catalogue_stability_allyears.py
    python scripts/catalogue_stability_allyears.py --years 2003 2022 --jobs 2
    python scripts/catalogue_stability_allyears.py --skip-runs

For every summer, reruns scripts/detect_heatwaves_europe.py under every
combination of minimum area (25k, 50k, 100k km2), percentile (90th, 95th) and
threshold window (7, 15, 31 days), always with a leave-one-out baseline: a
summer inside 1991-2020 is judged against the other 29 years. The reference
setting is 50k km2, 90th percentile, 15 days. Its top events are matched into
every other setting with scripts/catalogue_sensitivity.py's matcher, and each
summer is summarized by how many of them change rank, which parameter moved
them, and how crowded its leading events were.

Leave-one-out thresholds are about 36 MB each and only one summer uses them,
so each is deleted once that summer's runs finish.

Outputs:
  data/allyears/heatwave_events_europe_<year>_a<k>k_p<pct>_w<days>.csv
  data/allyears/catalogue_sensitivity_europe_<year>.csv
  data/catalogue_stability_allyears.csv          one row per summer
  (merge_settings counts settings in which two reference events became one)
  plots/detection/catalogue_stability_allyears.pdf
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HEATWAVE_ROOT", REPO))
DATA = ROOT / "data"
OUT = DATA / "allyears"
PLOTS = ROOT / "plots" / "detection"


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script("catalogue_sensitivity")
DH = _script("detect_heatwaves_europe")

AREAS_KM2 = (25_000, 50_000, 100_000)
PERCENTILES = (0.90, 0.95)
WINDOWS = (7, 15, 31)
REFERENCE = (50_000, 0.90, 15)
BASELINE = (1991, 2020)
DATES = ["start", "peak", "end"]


def labels_for(csv: Path) -> Path:
    return DH.labels_path_for(csv)


def tag_for(area_km2: float, pct: float, window: int) -> str:
    return S.tag_for(area_km2, pct) + f"_w{window}"


def footprint(path: Path) -> pd.DataFrame:
    """One row per event cell-day: a flat (time, lat, lon) index and the
    event_id it belongs to, from detect_heatwaves_europe.py --save-labels."""
    z = np.load(path)
    cell = np.ravel_multi_index((z["t"], z["i"], z["j"]), tuple(z["shape"]))
    return pd.DataFrame({"cell": cell, "event_id": z["event_id"]})


def match_by_overlap(ref: pd.DataFrame, ref_fp: pd.DataFrame,
                     run: pd.DataFrame, run_fp: pd.DataFrame) -> pd.DataFrame:
    """Match every reference event to the run event sharing most cell-days.

    Returns one row per row of `ref`: run_index (row of `run`, or -1 when no
    cell-day is shared), shared (cell-days in common), n_pieces (run events
    it overlaps, so >1 means it split) and merged (another reference event
    has the same best match, so the run joined them).

    Shared cells, unlike dates and centroids, cannot mistake a vanished
    event for a merged one: a reference event counts as merged only if its
    own cells lie inside the joint run event. Under a stricter setting (higher
    percentile, larger area) cells are only removed, so a true merge is
    impossible there and none is reported."""
    if not len(run) or "event_id" not in run:   # the setting found nothing
        return pd.DataFrame({"run_index": -1, "shared": 0, "n_pieces": 0,
                             "merged": False}, index=ref.index)
    both = ref_fp.merge(run_fp, on="cell", suffixes=("_ref", "_run"))
    counts = (both.groupby(["event_id_ref", "event_id_run"]).size()
              .rename("shared").reset_index())
    best = (counts.sort_values("shared", ascending=False, kind="stable")
            .drop_duplicates("event_id_ref").set_index("event_id_ref"))
    pieces = counts.groupby("event_id_ref").size()
    row_of = pd.Series(np.arange(len(run)), index=run.event_id.to_numpy())

    out = pd.DataFrame(index=ref.index)
    ids = ref.event_id.to_numpy()
    run_id = best.event_id_run.reindex(ids)
    out["run_index"] = [int(row_of[r]) if pd.notna(r) else -1 for r in run_id]
    out["shared"] = best.shared.reindex(ids).fillna(0).astype(int).to_numpy()
    out["n_pieces"] = pieces.reindex(ids).fillna(0).astype(int).to_numpy()
    hit = out.run_index[out.run_index >= 0]
    out["merged"] = out.run_index.isin(hit[hit.duplicated(keep=False)])
    return out


def changed_parameter(area, pct, window) -> str | None:
    """Which single parameter this setting changes from the reference, or
    None when it changes more than one (or none)."""
    diff = [name for name, v, r in zip(("area", "percentile", "window"),
                                       (area, pct, window), REFERENCE)
            if v != r]
    return diff[0] if len(diff) == 1 else None


def crowding(top: pd.DataFrame, max_gap_days: int = 3,
             max_dist_deg: float = 10.0) -> int:
    """Pairs of leading events that sit close in both time and place.

    Close in time: they overlap, or one starts within max_gap_days of the
    other ending. Close in place: centroids within max_dist_deg, longitude
    scaled by latitude as in the matcher.

    ponytail: dates and centroids only, not footprint overlap. Two events on
    either side of a large one can count as close without sharing a cell.
    Use the label grids if this measure ends up carrying the result."""
    n = 0
    for (_, a), (_, b) in combinations(top.iterrows(), 2):
        gap = (max(a.start, b.start) - min(a.end, b.end)).days
        dlat = a.centroid_lat - b.centroid_lat
        dlon = (a.centroid_lon - b.centroid_lon) * np.cos(
            np.radians(0.5 * (a.centroid_lat + b.centroid_lat)))
        if gap <= max_gap_days and np.hypot(dlat, dlon) <= max_dist_deg:
            n += 1
    return n


def run_year(year: int, skip_runs: bool) -> None:
    """Every setting for one summer, then drop its leave-one-out thresholds."""
    for area, pct, window in product(AREAS_KM2, PERCENTILES, WINDOWS):
        tag = tag_for(area, pct, window)
        csv = OUT / f"heatwave_events_europe_{year}{tag}.csv"
        if skip_runs or (csv.exists() and labels_for(csv).exists()):
            continue
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "detect_heatwaves_europe.py"),
             "--year", str(year), "--min-area", str(area),
             "--percentile", str(pct), "--window", str(window),
             "--leave-one-out", "--no-plot", "--save-labels",
             "--out-dir", str(OUT),
             "--tag", tag],
            check=True, env=os.environ, stdout=subprocess.DEVNULL)
    for f in DATA.glob(f"heatwave_threshold_europe_*_loo{year}_*.nc"):
        f.unlink()
    print(f"{year}: done", flush=True)


def compare_year(year: int, top: int) -> tuple[pd.DataFrame, dict]:
    """Per-setting matches of the reference top events, and the summary row."""
    def load(area, pct, window):
        path = OUT / f"heatwave_events_europe_{year}{tag_for(area, pct, window)}.csv"
        if not path.exists():
            return None
        run = pd.read_csv(path)
        for col in DATES:
            run[col] = pd.to_datetime(run[col])
        return run

    def fp(area, pct, window):
        return footprint(labels_for(
            OUT / f"heatwave_events_europe_{year}{tag_for(area, pct, window)}.csv"))

    ref_all = load(*REFERENCE)
    if ref_all is None:
        return pd.DataFrame(), {}
    ref = ref_all.head(top)
    ref_fp = fp(*REFERENCE)

    rows, moved, merge_settings, merged_ranks = [], {}, 0, set()
    for area, pct, window in product(AREAS_KM2, PERCENTILES, WINDOWS):
        if (area, pct, window) == REFERENCE:
            continue
        run = load(area, pct, window)
        if run is None:
            continue
        # Match ALL reference events, so a leader that absorbs a lower-ranked
        # event still shows as merged.
        m = match_by_overlap(ref_all, ref_fp, run, fp(area, pct, window))
        top_m = m.iloc[:len(ref)]
        merge_settings += bool(top_m.merged.any())
        merged_ranks.update(int(r) for r in ref["rank"][top_m.merged.to_numpy()])
        param = changed_parameter(area, pct, window)
        for (_, e), j, merged in zip(ref.iterrows(), top_m.run_index,
                                     top_m.merged):
            new_rank = int(run.loc[j, "rank"]) if j >= 0 else np.nan
            did_move = j < 0 or new_rank != e["rank"]
            rows.append({"year": year, "min_area_km2": area, "percentile": pct,
                         "window_days": window, "changed": param or "several",
                         "n_events": len(run), "ref_rank": int(e["rank"]),
                         "rank": new_rank, "moved": did_move,
                         "merged": bool(merged)})
            if did_move:
                moved.setdefault(param or "several", set()).add(int(e["rank"]))

    every = set().union(*moved.values()) if moved else set()
    summary = {
        "year": year,
        "in_baseline": BASELINE[0] <= year <= BASELINE[1],
        "n_events": len(ref_all),
        "n_ref": len(ref),
        "top1_moved": 1 in every,
        "n_moved_any": len(every),
        "n_moved_area": len(moved.get("area", ())),
        "n_moved_percentile": len(moved.get("percentile", ())),
        "n_moved_window": len(moved.get("window", ())),
        "highest_rank_moved": min(every) if every else np.nan,
        "merge_settings": merge_settings,
        "n_top_merged": len(merged_ranks),
        "crowding": crowding(ref),
    }
    return pd.DataFrame(rows), summary


def plot(summary: pd.DataFrame, full: pd.DataFrame, rho: float, p: float,
         top: int, path: Path) -> None:
    """Left: every summer. Right: only summers with `top` events, the ones
    the correlation is computed on."""
    sys.path.insert(0, str(REPO / "scripts"))
    from common import mpl_apply
    mpl_apply()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    x = summary.year.values
    width = 0.27
    for k, (col, label, colour) in enumerate([
            ("n_moved_area", "minimum area", "#1D4E89"),
            ("n_moved_percentile", "percentile", "#B5470F"),
            ("n_moved_window", "window", "#7FB069")]):
        ax1.bar(x + (k - 1) * width, summary[col], width, label=label,
                color=colour)
    ax1.set_xlabel("summer")
    ax1.set_ylabel(f"top-{top} events that change rank")
    ax1.legend(title="changing only the", frameon=False, fontsize=8)

    # Integer on integer: size each point by how many summers share it.
    pts = full.groupby(["crowding", "n_moved_any"]).year.agg(list)
    ax2.scatter([c for c, _ in pts.index], [m for _, m in pts.index],
                s=[25 * len(v) for v in pts], color="#1B1F2A", alpha=0.8)
    for (c, m), years in pts.items():
        named = [y for y in years if y in (2003, 2022, 2026)]
        if named:
            ax2.annotate(", ".join(map(str, named)), (c, m), xytext=(5, 3),
                         textcoords="offset points", fontsize=8)
    ax2.set_xlabel(f"crowded pairs among the top {top}")
    ax2.set_ylabel(f"top-{top} events that change rank (any setting)")
    ax2.set_title(rf"Spearman $\rho$ = {rho:.2f}, $p$ = {p:.2g}"
                  if np.isfinite(rho) else "Spearman undefined", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"-> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int,
                        default=list(range(1991, 2027)))
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=3,
                        help="Summers run at once; each needs about 2 GB "
                             "while it builds a threshold")
    parser.add_argument("--skip-runs", action="store_true",
                        help="Only rebuild the comparison from existing CSVs")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if not args.skip_runs:
        # Summers outside the baseline share one threshold per setting. Build
        # those with a single summer first, so parallel runs never race to
        # write the same cache file.
        # ponytail: assumes every shared summer has the same month span (part
        # of the cache key), true for the April-October downloads. Group the
        # shared summers by span if a download ever differs.
        years = list(args.years)
        shared = [y for y in years if not BASELINE[0] <= y <= BASELINE[1]]
        if shared:
            run_year(shared[0], False)
            years.remove(shared[0])
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(lambda y: run_year(y, False), years))

    tables, rows = [], []
    for year in args.years:
        table, summary = compare_year(year, args.top)
        if not summary:
            print(f"{year}: no reference catalogue, skipped")
            continue
        table.to_csv(OUT / f"catalogue_sensitivity_europe_{year}.csv",
                     index=False)
        tables.append(table)
        rows.append(summary)

    summary = pd.DataFrame(rows)
    out = DATA / "catalogue_stability_allyears.csv"
    summary.to_csv(out, index=False)
    print(f"\n-> {out}\n")
    print(summary.to_string(index=False))

    if summary.empty:
        sys.exit("no summers had a reference catalogue")
    # Summers with fewer than `top` events cap both quantities, which would
    # correlate them through the event count; test only full summers.
    full = summary[summary.n_ref == args.top]
    rho = p = np.nan
    if len(full) > 2 and full.crowding.nunique() > 1 and \
            full.n_moved_any.nunique() > 1:
        from scipy.stats import spearmanr
        rho, p = spearmanr(full.crowding, full.n_moved_any)
    print(f"\nSpearman over {len(full)} summers with {args.top} events, "
          f"crowding vs events moved: rho={rho:.2f} p={p:.3g}")
    print(f"summers where number 1 moved: {int(summary.top1_moved.sum())} "
          f"of {len(summary)}")
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot(summary, full, rho, p, args.top, PLOTS / "catalogue_stability_allyears.pdf")


if __name__ == "__main__":
    main()
