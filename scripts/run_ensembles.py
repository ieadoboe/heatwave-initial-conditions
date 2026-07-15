"""Run the 75-member stochastic-ensemble baselines for every atlas event.

    python scripts/run_ensembles.py                       # all events, resume-aware
    python scripts/run_ensembles.py --configs configs/pnw_jun2021.yaml
    python scripts/run_ensembles.py --members 25          # cheaper spread estimate

Needs each event's completed optimization run (storyline.csv in its run dir)
— run scripts/run_atlas.py first. Per event: rebuild the IC if missing ->
unroll N stochastic members -> save ensemble.csv + the ensemble figure.
Events with an existing >=N-member ensemble.csv are skipped unless --rerun.

Writes data/ensemble_summary.csv and MERGES the ensemble metrics into
data/atlas_summary.csv (columns: n_members, ens_mean_peak_C, ens_max_peak_C,
ens_peak_spread_C, gain_vs_ens_max_C, gain_over_spread).

RUNTIME: needs a GPU and GCS access — run on Colab, not the local CPU venv.
"""

import argparse
import gc
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from heatwave_ic import load_config, describe  # noqa: E402
from heatwave_ic.ensemble import run_event_ensemble  # noqa: E402

ATLAS_ORDER = [
    "configs/pnw_jun2021.yaml",
    "configs/stjohns_aug2025.yaml",
    "configs/moscow_jul2010.yaml",
    "configs/japan_jul2018.yaml",
    "configs/sahel_apr2024.yaml",
    "configs/brazil_nov2023.yaml",
    "configs/siberia_jun2020.yaml",
]

METRIC_COLS = ["n_members", "ens_mean_peak_C", "ens_max_peak_C",
               "ens_peak_spread_C", "gain_vs_ens_max_C", "gain_over_spread"]


def _sync_tree(src, dst):
    src, dst = Path(src), Path(dst)
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", f"{src}/", str(dst)], check=False)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=ATLAS_ORDER)
    parser.add_argument("--members", type=int, default=75,
                        help="Ensemble size (default 75, matching W&DL)")
    parser.add_argument("--rerun", action="store_true",
                        help="Recompute even if an ensemble.csv exists")
    parser.add_argument("--persist-dir", default=None,
                        help="Durable dir (e.g. mounted Drive folder): run "
                             "dirs are restored from it at start and synced "
                             "back after every event")
    args = parser.parse_args()

    persist = Path(args.persist_dir) if args.persist_dir else None
    if persist:
        print(f"Restoring runs from {persist} ...")
        _sync_tree(persist / "opt_runs", "data/opt_runs")

    def sync_back():
        if persist:
            _sync_tree("data/opt_runs", persist / "opt_runs")
            _sync_tree("plots", persist / "plots")

    from heatwave_ic.model import load_model
    models = {}
    rows = []
    for path in args.configs:
        cfg = load_config(path)
        name = cfg["event"]["name"]
        print(f"\n{'=' * 70}\n{describe(cfg)}\n{'=' * 70}")
        model_name = cfg["model_name"]
        if model_name not in models:
            print(f"Loading model {model_name} ...")
            models[model_name] = load_model(model_name)
        try:
            summary = run_event_ensemble(cfg, models[model_name],
                                         n_members=args.members,
                                         skip_existing=not args.rerun)
        except Exception as exc:
            traceback.print_exc()
            summary = {"event": name, "status": f"FAILED: {exc}"}
        rows.append(summary)
        print(f"-> {summary.get('status')}  "
              f"gain_vs_ens_max={summary.get('gain_vs_ens_max_C', '—')} C  "
              f"gain/spread={summary.get('gain_over_spread', '—')}")
        sync_back()
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    df = pd.DataFrame(rows)
    out = Path("data/ensemble_summary.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)

    # Merge the metrics into the atlas summary.
    atlas_path = Path("data/atlas_summary.csv")
    if atlas_path.exists() and "event" in df.columns:
        atlas = pd.read_csv(atlas_path)
        atlas = atlas.drop(columns=[c for c in METRIC_COLS if c in atlas.columns])
        metrics = df[["event"] + [c for c in METRIC_COLS if c in df.columns]]
        atlas = atlas.merge(metrics, on="event", how="left")
        atlas.to_csv(atlas_path, index=False)
        if persist:
            shutil.copy2(atlas_path, persist / "atlas_summary.csv")
        print(f"\nEnsemble metrics merged into {atlas_path}")
        print(atlas.to_string(index=False))
    else:
        print(f"\nEnsemble summary -> {out}")
        print(df.to_string(index=False))
    if persist:
        shutil.copy2(out, persist / "ensemble_summary.csv")


if __name__ == "__main__":
    main()
