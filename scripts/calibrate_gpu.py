"""Time one event on this GPU, so the batch walltimes are measured not guessed.

    python scripts/calibrate_gpu.py --config configs/siberia_jun2020.yaml

Runs a handful of optimizer iterations and a couple of ensemble members, then
extrapolates both to the full 75 and prints the walltime to put in the sbatch
scripts. Pick the LONGEST event (evol_days 11: pnw, brazil, siberia); the
shorter ones finish inside the same limit.

The first iteration and the first member each pay for XLA compilation, which
the later ones reuse, so they are timed and reported separately.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from heatwave_ic import load_config, describe  # noqa: E402
from heatwave_ic.data import build_ic_zarr, load_ic_on_model_grid  # noqa: E402
from heatwave_ic.evaluate import box_t1000_trajectory  # noqa: E402
from heatwave_ic.optimize import (encode_initial_state, optimize_event,  # noqa: E402
                                  target_indices)


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/siberia_jun2020.yaml")
    parser.add_argument("--iters", type=int, default=6,
                        help="Optimizer iterations to time (default 6)")
    parser.add_argument("--members", type=int, default=3,
                        help="Ensemble members to time (default 3)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(describe(cfg))

    import jax
    print(f"\njax {jax.__version__}  devices: {jax.devices()}")

    from heatwave_ic.model import load_model
    t = time.perf_counter()
    model = load_model(cfg["model_name"])
    print(f"model loaded in {time.perf_counter() - t:.1f} s")

    build_ic_zarr(model, cfg)
    eval_era5 = load_ic_on_model_grid(model, cfg["paths"]["ic_zarr"])

    # --- optimizer ---------------------------------------------------------
    marks = []
    t_start = time.perf_counter()
    optimize_event(model, eval_era5, cfg, iterations=args.iters, progress=False,
                   on_step=lambda *a: marks.append(time.perf_counter()))
    steps = np.diff(marks)
    per_iter = float(np.median(steps)) if len(steps) else float("nan")
    print(f"\noptimizer: {len(marks)} iterations timed")
    print(f"  compilation + first iteration: {marks[0] - t_start:.1f} s")
    print(f"  steady-state per iteration:    {per_iter:.1f} s")

    iters_total = int(cfg["optimizer"]["iteration_number"])
    opt_estimate = per_iter * iters_total + 300  # + model load, IC, saving
    print(f"  -> {iters_total} iterations ≈ {_hms(opt_estimate)}")

    # --- ensemble members --------------------------------------------------
    lat_i, lon_i = target_indices(eval_era5, cfg["event"]["target_lat"],
                                  cfg["event"]["target_lon_east"])
    outer_steps = int(round(cfg["run"]["evol_days"] * 24))
    base_seed = int(cfg["run"]["rng_seed"])
    member_times = []
    for i in range(args.members):
        t = time.perf_counter()
        state, forcings = encode_initial_state(model, eval_era5,
                                               rng_seed=base_seed + i)
        box_t1000_trajectory(model, state, forcings, outer_steps,
                             lat_i, lon_i, cfg["run"]["init_date"])
        member_times.append(time.perf_counter() - t)
        print(f"  member {i:03d}: {member_times[-1]:.1f} s")

    per_member = float(np.median(member_times[1:] or member_times))
    ens_estimate = per_member * 75 + 300
    print(f"\nensemble: steady-state per member {per_member:.1f} s")
    print(f"  -> 75 members ≈ {_hms(ens_estimate)}")

    print("\nSuggested sbatch walltimes (estimate + 50% margin):")
    print(f"  02_run_atlas.sbatch      --time={_hms(opt_estimate * 1.5)}")
    print(f"  03_run_ensembles.sbatch  --time={_hms(ens_estimate * 1.5)}")
    print("\nCheck peak GPU memory in another shell with:")
    print("  nvidia-smi --query-gpu=memory.used,memory.total --format=csv")
    print("If it is close to the slice size, move to --gpus=h100_3g.40gb:1.")


if __name__ == "__main__":
    main()
