"""Build the ARCO-ERA5 initial-condition zarrs for the atlas events.

    python scripts/build_ics.py                          # all 7 events
    python scripts/build_ics.py --configs configs/pnw_jun2021.yaml
    python scripts/build_ics.py --overwrite               # force a rebuild

Downloads and regrid-prepares nothing that needs a GPU: this is the network
half of the pipeline, split out so a cluster can run it as a cheap CPU job
before the GPU jobs start. Each store holds the three hourly snapshots at
init_date that the optimizer and the ensembles read, so the download is a few
GB per event rather than the ~50 GB of a full 2-day hourly window.

An existing store is reused when it is valid AND starts at the config's
init_date; a store written before the forcing-shift ordering fix starts 24 h
late and is rebuilt automatically.
"""

import argparse
import gc
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heatwave_ic import load_config, describe  # noqa: E402
from heatwave_ic.data import build_ic_zarr  # noqa: E402

ATLAS_ORDER = [
    "configs/pnw_jun2021.yaml",
    "configs/stjohns_aug2025.yaml",
    "configs/moscow_jul2010.yaml",
    "configs/japan_jul2018.yaml",
    "configs/sahel_apr2024.yaml",
    "configs/brazil_nov2023.yaml",
    "configs/siberia_jun2020.yaml",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=ATLAS_ORDER,
                        help="Event YAMLs to build (default: the full atlas)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild even when a valid store exists")
    parser.add_argument("--snapshots", type=int, default=3,
                        help="Hourly snapshots to keep from init_date "
                             "(default 3; 0 keeps the whole 2-day window)")
    args = parser.parse_args()

    from heatwave_ic.model import load_model
    models = {}
    failures = []
    for path in args.configs:
        cfg = load_config(path)
        name = cfg["event"]["name"]
        print(f"\n{'=' * 70}\n{describe(cfg)}\n{'=' * 70}")
        model_name = cfg["model_name"]
        if model_name not in models:
            print(f"Loading model {model_name} ...")
            models[model_name] = load_model(model_name)
        try:
            out = build_ic_zarr(models[model_name], cfg,
                                overwrite=args.overwrite,
                                snapshots=args.snapshots or None)
            print(f"-> {out}")
        except Exception as exc:
            traceback.print_exc()
            failures.append((name, exc))
        gc.collect()

    if failures:
        print(f"\n{len(failures)} event(s) failed:")
        for name, exc in failures:
            print(f"  {name}: {exc}")
        sys.exit(1)
    print(f"\nAll {len(args.configs)} IC stores ready.")


if __name__ == "__main__":
    main()
