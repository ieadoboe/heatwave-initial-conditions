"""The whole per-event Direction A pipeline in one call.

run_event(cfg) does: IC build (if needed) -> regrid -> encode -> optimize ->
save run outputs -> storyline evaluation -> plots -> Köppen tags, and returns
a one-row summary dict. scripts/optimize_event.py runs one event through it;
scripts/run_atlas.py sweeps every event config and collects the atlas table.

RUNTIME: needs a GPU and GCS access — run on Colab, not the local CPU venv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from heatwave_ic.data import build_ic_zarr, load_ic_on_model_grid
from heatwave_ic.evaluate import box_t1000_trajectory
from heatwave_ic.optimize import encode_initial_state, optimize_ic, target_indices
from heatwave_ic.outputs import (make_run_dir, save_losses, save_state_fields,
                                 save_trajectory_nc)
from heatwave_ic.zones import classify_event


def run_event(cfg: dict, model=None, *, skip_existing: bool = False,
              evaluate: bool = True, progress: bool = True) -> dict:
    """Run the full pipeline for one resolved event config.

    model: pass a preloaded model to reuse it across events (the atlas runner
    does this); None loads cfg['model_name'].
    skip_existing: if the run dir already holds optimized.nc, skip the run and
    summarise from the saved storyline instead (resume support).
    evaluate: unroll original vs optimized trajectories, save storyline.csv
    and the loss/storyline figures, and report the storyline gain.
    """
    event, run = cfg["event"], cfg["run"]
    loss_cfg, opt_cfg = cfg["loss"], cfg["optimizer"]
    name = event["name"]

    koppen = classify_event(cfg)
    summary = {
        "event": name,
        "zone": event.get("zone", ""),
        "koppen_point": koppen["koppen_point"],
        "koppen_cell": koppen["koppen_model_cell"],
        "init_date": str(run["init_date"]),
        "evol_days": run["evol_days"],
    }

    out_dir = Path(make_run_dir(cfg))
    summary["run_dir"] = str(out_dir)
    if skip_existing and (out_dir / "optimized.nc").exists():
        summary["status"] = "skipped (existing run)"
        storyline = out_dir / "storyline.csv"
        if storyline.exists():
            df = pd.read_csv(storyline, index_col=0)
            summary["storyline_gain_C"] = round(
                float(df["optimized_C"].max() - df["unperturbed_C"].max()), 2)
        return summary

    if model is None:
        from heatwave_ic.model import load_model
        model = load_model(cfg["model_name"])

    build_ic_zarr(model, cfg)
    eval_era5 = load_ic_on_model_grid(model, cfg["paths"]["ic_zarr"])
    initial_state, all_forcings = encode_initial_state(
        model, eval_era5, rng_seed=run["rng_seed"])
    lat_i, lon_i = target_indices(
        eval_era5, event["target_lat"], event["target_lon_east"])
    outer_steps = int(round(run["evol_days"] * 24))
    window_steps = int(round(loss_cfg["target_window_days"] * 24))

    save_every = int(run.get("save_every_iters", 0))

    def on_step(step, loss_val, box_T, reg, get_state):
        if save_every and step % save_every == 0:
            save_trajectory_nc(model, get_state(), all_forcings, outer_steps,
                               out_dir / f"optimized_step{step}.nc",
                               variables=["temperature", "geopotential"])

    result = optimize_ic(
        model, initial_state, all_forcings,
        outer_steps=outer_steps, window_steps=window_steps,
        lat_i=lat_i, lon_i=lon_i,
        beta=loss_cfg["beta"], lam=loss_cfg["lambda"], t_ref=loss_cfg["T_ref"],
        weights=loss_cfg["lambda_weights"],
        learning_rate=opt_cfg["learning_rate"],
        iterations=opt_cfg["iteration_number"],
        on_step=on_step, progress=progress,
    )

    save_losses(result, out_dir)
    save_trajectory_nc(model, result["optimized_state"], all_forcings,
                       outer_steps, out_dir / "optimized.nc")
    save_state_fields(model, result["optimized_state"], out_dir, "opt")
    save_trajectory_nc(model, result["initial_state"], all_forcings,
                       outer_steps, out_dir / "original.nc")
    save_state_fields(model, result["initial_state"], out_dir, "original")

    summary.update(
        status="ok",
        final_loss=round(float(result["losses"][-1]), 4),
        final_box_T_C=round(float(result["box_T_K"][-1]) - 273.15, 2),
        final_reg=float(result["reg"][-1]),
    )

    if evaluate:
        from heatwave_ic import plots  # deferred: pulls in matplotlib/common

        traj_ori = box_t1000_trajectory(
            model, result["initial_state"], all_forcings, outer_steps,
            lat_i, lon_i, run["init_date"])
        traj_opt = box_t1000_trajectory(
            model, result["optimized_state"], all_forcings, outer_steps,
            lat_i, lon_i, run["init_date"])
        pd.DataFrame({"unperturbed_C": traj_ori, "optimized_C": traj_opt}
                     ).to_csv(out_dir / "storyline.csv")
        summary["storyline_gain_C"] = round(
            float(traj_opt.max() - traj_ori.max()), 2)

        plots_dir = Path(cfg["paths"]["plots_dir"])
        plots.plot_loss(result, title=f"{name} IC optimization --- loss",
                        save=plots_dir / f"{name}_opt_loss.pdf")
        plots.plot_storyline(traj_ori, traj_opt, event["start"], event["end"],
                             title=f"{name} storyline",
                             save=plots_dir / f"{name}_storyline.pdf")
        import matplotlib.pyplot as plt
        plt.close("all")

    return summary
