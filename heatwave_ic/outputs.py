"""Run directories and on-disk outputs (data/ per project convention)."""

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def make_run_dir(cfg: dict, params: dict | None = None) -> str:
    """data/opt_runs/{event}_lr..._it..._lam..._b..._d.../ — the run name
    encodes the hyperparameters actually used (pass result['params'] when
    they were overridden)."""
    loss, opt, run = cfg["loss"], cfg["optimizer"], cfg["run"]
    p = {
        "learning_rate": opt["learning_rate"],
        "iterations": opt["iteration_number"],
        "lam": loss["lambda"],
        "beta": loss["beta"],
        "evol_days": run["evol_days"],
    }
    if params:
        p.update({k: params[k] for k in p if k in params})
    parts = [
        cfg["event"]["name"],
        f"lr{float(p['learning_rate']):.0e}",
        f"it{p['iterations']}",
        f"lam{p['lam']}",
        f"b{p['beta']}",
        f"d{p['evol_days']:.0f}",
    ]
    name = "_".join(str(x).replace(".", "p").replace("-", "m") for x in parts)
    path = os.path.join(cfg["paths"]["output_dir"], name)
    os.makedirs(path, exist_ok=True)
    return path


def save_losses(result: dict, out_dir: str | Path) -> None:
    np.save(str(Path(out_dir) / "losses"), result["losses"])
    np.save(str(Path(out_dir) / "box_T_K"), result["box_T_K"])
    np.save(str(Path(out_dir) / "reg"), result["reg"])


def save_state_fields(model, state, out_dir: str | Path, tag: str) -> None:
    """Surface pressure, vorticity and divergence of a model state → .npy."""
    out_dir = Path(out_dir)
    h = model.model_coords.horizontal
    sp = model.from_nondim_units(
        jnp.squeeze(jnp.exp(h.to_nodal(state.state.log_surface_pressure)), axis=0),
        "kg / (meter s**2)",
    )
    vort = model.from_nondim_units(h.to_nodal(state.state.vorticity), "1/s")
    div = model.from_nondim_units(h.to_nodal(state.state.divergence), "1/s")
    np.save(str(out_dir / f"log_surface_pressure_{tag}"), sp)
    np.save(str(out_dir / f"vorticity_{tag}"), vort)
    np.save(str(out_dir / f"divergence_{tag}"), div)


def save_trajectory_nc(model, state, all_forcings, steps: int,
                       path: str | Path, variables: list[str] | None = None) -> None:
    """Unroll a state and write the trajectory to netCDF."""
    _, preds = model.unroll(state, all_forcings, steps=steps)
    ds = model.data_to_xarray(preds, times=np.arange(steps))
    if variables:
        ds = ds[variables]
    ds.to_netcdf(str(path))
