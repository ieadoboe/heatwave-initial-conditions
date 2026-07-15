"""Full pipeline for ONE event config (thin CLI over heatwave_ic.pipeline).

    python scripts/optimize_event.py --config configs/stjohns_aug2025.yaml --build-ic
    python scripts/optimize_event.py --config configs/stjohns_aug2025.yaml

Does: IC build (if needed) -> optimize -> save outputs -> storyline
evaluation -> figures. For the whole event set use scripts/run_atlas.py.

RUNTIME: needs a GPU and GCS access — run on Colab, not the local CPU venv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heatwave_ic import load_config, describe, load_model, build_ic_zarr  # noqa: E402
from heatwave_ic.pipeline import run_event  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to event YAML")
    parser.add_argument("--build-ic", action="store_true",
                        help="Build the ERA5 IC zarr from ARCO-ERA5, then exit")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the storyline evaluation/figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(describe(cfg))

    if args.build_ic:
        build_ic_zarr(load_model(cfg["model_name"]), cfg)
        return

    summary = run_event(cfg, evaluate=not args.no_eval)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
