"""Every detected European heat wave, 1991-2026, with a default storyline set-up.

    python scripts/storyline_candidates.py
    python scripts/storyline_candidates.py --make-config 2003_r01

Reads the reference catalogue of every summer from the all-summers sweep
(scripts/catalogue_stability_allyears.py: 50k km2, 90th percentile, 15-day
window, leave-one-out baseline) and proposes a storyline set-up for each event:

  target point   where the event was most unusual on its peak day
  target window  the 5 days centred on the peak day
  start (init)   6 days before the peak, as in six of the seven configs

and a robustness label from the sweep: "holds" when every other setting
detects the event as one event of its own (found, and never merged with
another), "depends on settings" otherwise. Events are matched across settings
by the cell-days they share. max_date_shift_days says how far its start or
end moves across the settings that find it.

These are PROPOSALS, not checked set-ups. --make-config writes one event's
proposal as configs/europe_<id>.yaml for a person to review before running;
only then does it belong in data/storyline_events.csv.

Outputs:
  data/storyline_candidates_1991_2026.csv
  configs/europe_<id>.yaml           with --make-config
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HEATWAVE_ROOT", REPO))
DATA = ROOT / "data"


def _script(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AY = _script("catalogue_stability_allyears")
S = AY.S

LEAD_DAYS = 6
WINDOW_DAYS = 5
CATALOGUE = "a50k_p90_w15_loo"
TEMPLATE = REPO / "configs" / "moscow_jul2010.yaml"  # W&DL's released settings


def storyline_setup(peak) -> dict:
    """Default dates for an event peaking on `peak`."""
    peak = pd.Timestamp(peak)
    half = WINDOW_DAYS // 2
    init = peak - pd.Timedelta(days=LEAD_DAYS)
    window_end = peak + pd.Timedelta(days=half)
    return {
        "init_date": init.date(),
        "lead_days": LEAD_DAYS,
        "target_window_start": (peak - pd.Timedelta(days=half)).date(),
        "target_window_end": window_end.date(),
        # The run must reach the end of the window's last day.
        "evol_days": (window_end + pd.Timedelta(days=1) - init).days,
    }


def robustness(ref: pd.DataFrame, ref_fp: pd.DataFrame,
               runs: list[tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """Per reference event, across the other settings (each a run table and
    its footprint): how often it is found, keeps its rank, is merged with
    another reference event or split into pieces, and how far its dates
    move. Matching is by shared cell-days (AY.match_by_overlap)."""
    n = len(ref)
    found, same_rank, merged, split, shift = (np.zeros(n, int) for _ in range(5))
    for run, run_fp in runs:
        m = AY.match_by_overlap(ref, ref_fp, run, run_fp)
        for i, (j, is_merged, pieces) in enumerate(
                zip(m.run_index, m.merged, m.n_pieces)):
            if j < 0:
                continue
            found[i] += 1
            same_rank[i] += int(run["rank"].iloc[j]) == int(ref["rank"].iloc[i])
            merged[i] += bool(is_merged)
            split[i] += pieces > 1
            d = max(abs((run.start.iloc[j] - ref.start.iloc[i]).days),
                    abs((run.end.iloc[j] - ref.end.iloc[i]).days))
            shift[i] = max(shift[i], d)
    holds = (found == len(runs)) & (merged == 0)
    return pd.DataFrame({
        "n_settings": len(runs),
        "n_settings_found": found,
        "n_settings_same_rank": same_rank,
        "n_settings_merged": merged,
        "n_settings_split": split,
        "max_date_shift_days": shift,
        "robustness": np.where(holds, "holds", "depends on settings"),
    })


def load(year: int, tag: str) -> pd.DataFrame | None:
    path = AY.OUT / f"heatwave_events_europe_{year}{tag}.csv"
    if not path.exists():
        return None
    run = pd.read_csv(path)
    for col in AY.DATES:
        run[col] = pd.to_datetime(run[col])
    return run


def refresh_reference(year: int) -> None:
    """Rerun a reference catalogue made before the detector recorded each
    event's peak-day target point."""
    area, pct, window = AY.REFERENCE
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "detect_heatwaves_europe.py"),
         "--year", str(year), "--min-area", str(area),
         "--percentile", str(pct), "--window", str(window),
         "--leave-one-out", "--no-plot", "--save-labels",
         "--out-dir", str(AY.OUT),
         "--tag", AY.tag_for(area, pct, window)],
        check=True, env=os.environ, stdout=subprocess.DEVNULL)
    for f in DATA.glob(f"heatwave_threshold_europe_*_loo{year}_*.nc"):
        f.unlink()


def candidates_for(year: int) -> pd.DataFrame:
    ref_tag = AY.tag_for(*AY.REFERENCE)
    ref = load(year, ref_tag)
    if ref is None:
        return pd.DataFrame()
    ref_csv = AY.OUT / f"heatwave_events_europe_{year}{ref_tag}.csv"
    if len(ref) and ("peak_day_anom_lat" not in ref.columns
                     or not AY.labels_for(ref_csv).exists()):
        print(f"{year}: refreshing the reference catalogue")
        refresh_reference(year)
        ref = load(year, ref_tag)
    if not len(ref):
        return pd.DataFrame()

    def with_fp(tag):
        run = load(year, tag)
        lab = AY.labels_for(AY.OUT / f"heatwave_events_europe_{year}{tag}.csv")
        return (run, AY.footprint(lab)) if run is not None and lab.exists() else None

    others = [with_fp(AY.tag_for(a, p, w))
              for a, p, w in AY.product(AY.AREAS_KM2, AY.PERCENTILES, AY.WINDOWS)
              if (a, p, w) != AY.REFERENCE]
    rob = robustness(ref, AY.footprint(AY.labels_for(ref_csv)),
                     [o for o in others if o is not None])

    setup = pd.DataFrame([storyline_setup(p) for p in ref.peak])
    out = pd.DataFrame({
        "event_id": [f"{year}_r{r:02d}" for r in ref["rank"]],
        "year": year, "catalogue": CATALOGUE, "rank": ref["rank"],
        "start": ref.start.dt.date, "peak": ref.peak.dt.date,
        "end": ref.end.dt.date, "duration_days": ref.duration_days,
        "peak_area_km2": ref.peak_area_km2,
        "severity_C_km2_day": ref.severity_C_km2_day,
        "koppen": ref.get("koppen", ""),
        "censored_start": ref.censored_start, "censored_end": ref.censored_end,
        "centroid_lat": ref.centroid_lat, "centroid_lon": ref.centroid_lon,
        "target_lat": ref.peak_day_anom_lat,
        "target_lon": ref.peak_day_anom_lon,
    })
    return pd.concat([out, setup, rob], axis=1)


def make_config(row: pd.Series, out_dir: Path = REPO / "configs") -> Path:
    """Write one candidate as a config for review, from the W&DL template.

    The config's event dates are the target window, because the loader
    requires the run to reach event.end; the detected dates are kept beside
    them for reference."""
    cfg = yaml.safe_load(TEMPLATE.read_text())
    name = f"europe_{row['event_id']}"
    cfg["event"] = {
        "name": name,
        "zone": row["koppen"],
        "target_lat": float(row["target_lat"]),
        "target_lon": float(row["target_lon"]),
        "start": str(row["target_window_start"]),
        "peak": str(row["peak"]),
        "end": str(row["target_window_end"]),
        "detected_start": str(row["start"]),
        "detected_end": str(row["end"]),
        "robustness": row["robustness"],
    }
    cfg["run"]["init_date"] = str(row["init_date"])
    cfg["run"]["evol_days"] = int(row["evol_days"])
    cfg["run"].pop("lead_days", None)
    cfg["loss"]["target_window_days"] = WINDOW_DAYS
    path = Path(out_dir) / f"{name}.yaml"
    if path.exists():
        sys.exit(f"{path} already exists; it may hold someone's edits. "
                 f"Delete it first to regenerate.")
    header = (f"# PROPOSED storyline for detected event {row['event_id']} "
              f"({row['start']} to {row['end']}, rank {row['rank']} of {row['year']}).\n"
              f"# Made by scripts/storyline_candidates.py from "
              f"{TEMPLATE.name}. Review before running.\n")
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int,
                        default=list(range(1991, 2027)))
    parser.add_argument("--make-config", metavar="EVENT_ID",
                        help="Write configs/europe_<EVENT_ID>.yaml from the "
                             "existing table, e.g. 2003_r01")
    args = parser.parse_args()
    out = DATA / "storyline_candidates_1991_2026.csv"

    if args.make_config:
        table = pd.read_csv(out)
        hit = table[table.event_id == args.make_config]
        if hit.empty:
            sys.exit(f"{args.make_config} is not in {out.name}")
        print(f"-> {make_config(hit.iloc[0])}  (review before running)")
        return

    parts = [candidates_for(y) for y in args.years]
    table = pd.concat([p for p in parts if len(p)], ignore_index=True)
    table.to_csv(out, index=False)
    print(f"-> {out}: {len(table)} events in {table.year.nunique()} summers, "
          f"{(table.robustness == 'holds').sum()} hold under every setting")


if __name__ == "__main__":
    main()
