"""Run the Direction A atlas: the full pipeline over every event config.

    python scripts/run_atlas.py                        # all 7 events, resume-aware
    python scripts/run_atlas.py --configs configs/pnw_jun2021.yaml
    python scripts/run_atlas.py --rerun                # ignore existing runs

Per event: build IC (if needed) -> optimize -> save outputs -> storyline
evaluation -> figures. Events that already have a completed run dir are
skipped (their storyline gain is read back) unless --rerun is given; a
failure in one event does not stop the sweep. Writes the cross-zone summary
to data/atlas_summary.csv.

RUNTIME: needs a GPU and GCS access — run on Colab, not the local CPU venv.
The PNW validation run comes first by default: if it does not reproduce
W&DL's +3.7 C storyline, fix that before trusting the rest.
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
from heatwave_ic.pipeline import run_event  # noqa: E402

# Validation event first; then the in-hand maritime event; then new zones.
ATLAS_ORDER = [
    "configs/pnw_jun2021.yaml",
    "configs/stjohns_aug2025.yaml",
    "configs/moscow_jul2010.yaml",
    "configs/japan_jul2018.yaml",
    "configs/sahel_apr2024.yaml",
    "configs/brazil_nov2023.yaml",
    "configs/siberia_jun2020.yaml",
]


def _sync_tree(src, dst):
    """Mirror src into dst (rsync when available, else copytree)."""
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
    parser.add_argument("--configs", nargs="*", default=ATLAS_ORDER,
                        help="Event YAMLs to run (default: the full atlas)")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-run events even if a completed run dir exists")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the storyline evaluation/figures")
    parser.add_argument("--persist-dir", default=None,
                        help="Durable directory (e.g. a mounted Google Drive "
                             "folder): completed runs are restored from it at "
                             "start and synced back after every event, so the "
                             "atlas resumes across Colab sessions. IC zarrs "
                             "are NOT persisted (too large) — they rebuild "
                             "automatically when missing.")
    args = parser.parse_args()

    persist = Path(args.persist_dir) if args.persist_dir else None
    if persist:
        print(f"Restoring completed runs from {persist} ...")
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
            summary = run_event(cfg, models[model_name],
                                skip_existing=not args.rerun,
                                evaluate=not args.no_eval)
        except Exception as exc:
            traceback.print_exc()
            summary = {"event": name, "status": f"FAILED: {exc}"}
        rows.append(summary)
        print(f"-> {summary.get('status')}  "
              f"gain={summary.get('storyline_gain_C', '—')} C")
        sync_back()

        # Each event compiles its own unroll length; drop caches between
        # events to keep GPU memory bounded.
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass
        gc.collect()

    df = pd.DataFrame(rows)
    out = Path("data/atlas_summary.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    if persist:
        shutil.copy2(out, persist / "atlas_summary.csv")
    print(f"\n{'=' * 70}\nAtlas summary -> {out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
