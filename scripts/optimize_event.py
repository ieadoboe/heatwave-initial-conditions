"""Thin CLI over the heatwave_ic package: IC optimization for any event config.

    python scripts/optimize_event.py --config configs/stjohns_aug2025.yaml --build-ic
    python scripts/optimize_event.py --config configs/stjohns_aug2025.yaml

Supersedes the monolithic scripts/stjohns_optimize.py (same numerics, now in
heatwave_ic/). RUNTIME: needs a GPU and GCS access — run on Colab, not the
local CPU venv.

Outputs (under paths.output_dir/<run-name>/): optimized.nc, original.nc,
losses/box_T_K/reg .npy, optimized/original state fields .npy, plus
plots/<event>_opt_loss.pdf.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heatwave_ic import (  # noqa: E402
    load_config, describe, load_model, build_ic_zarr, load_ic_on_model_grid,
    encode_initial_state, target_indices, optimize_ic,
    make_run_dir, save_losses, save_state_fields, save_trajectory_nc, plots,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to event YAML")
    parser.add_argument("--build-ic", action="store_true",
                        help="Build the ERA5 IC zarr from ARCO-ERA5, then exit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(describe(cfg))
    model = load_model(cfg["model_name"])

    if args.build_ic:
        build_ic_zarr(model, cfg)
        return

    event, run, loss, opt = cfg["event"], cfg["run"], cfg["loss"], cfg["optimizer"]
    eval_era5 = load_ic_on_model_grid(model, cfg["paths"]["ic_zarr"])
    initial_state, all_forcings = encode_initial_state(
        model, eval_era5, rng_seed=run["rng_seed"]
    )
    lat_i, lon_i = target_indices(
        eval_era5, event["target_lat"], event["target_lon_east"]
    )
    outer_steps = int(round(run["evol_days"] * 24))
    window_steps = int(round(loss["target_window_days"] * 24))

    out_dir = make_run_dir(cfg)
    print(f"Run dir: {out_dir}")
    save_every = int(run["save_every_iters"])

    def on_step(step, loss_val, box_T, reg, get_state):
        if save_every and step % save_every == 0:
            save_trajectory_nc(
                model, get_state(), all_forcings, outer_steps,
                Path(out_dir) / f"optimized_step{step}.nc",
                variables=["temperature", "geopotential"],
            )

    result = optimize_ic(
        model, initial_state, all_forcings,
        outer_steps=outer_steps, window_steps=window_steps,
        lat_i=lat_i, lon_i=lon_i,
        beta=loss["beta"], lam=loss["lambda"], t_ref=loss["T_ref"],
        weights=loss["lambda_weights"],
        learning_rate=opt["learning_rate"],
        iterations=opt["iteration_number"],
        on_step=on_step,
    )
    print("Training complete.")

    save_losses(result, out_dir)
    save_trajectory_nc(model, result["optimized_state"], all_forcings,
                       outer_steps, Path(out_dir) / "optimized.nc")
    save_state_fields(model, result["optimized_state"], out_dir, "opt")
    save_trajectory_nc(model, result["initial_state"], all_forcings,
                       outer_steps, Path(out_dir) / "original.nc")
    save_state_fields(model, result["initial_state"], out_dir, "original")
    print("Trajectories saved!")

    name = event["name"]
    plots.plot_loss(
        result, title=f"{name} IC optimization --- loss",
        save=Path(cfg["paths"]["plots_dir"]) / f"{name}_opt_loss.pdf",
    )


if __name__ == "__main__":
    main()
