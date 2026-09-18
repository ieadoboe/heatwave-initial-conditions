"""Run directories and on-disk outputs (data/ per project convention)."""

import numbers
import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from heatwave_ic.optimize import WEIGHT_KEYS

# What netCDF accepts as an attribute value. Note bool is deliberately absent:
# xarray's validator lets it through because Python bools are ints, but the
# netCDF4 writer then rejects the b1 dtype, so booleans are converted instead.
_NC_ATTR_TYPES = (str, bytes, numbers.Number, np.ndarray, np.number)
_NC_SEQ_TYPES = (list, tuple)


def make_run_dir(cfg: dict, params: dict | None = None) -> str:
    """data/opt_runs/{event}_i{init}_lr..._it..._lam..._b..._w..._d.../

    The run name encodes everything the result depends on: the init date,
    the optimizer and loss hyperparameters, the per-variable weights in
    WEIGHT_KEYS order, and the unroll length. Two configs that differ in any
    of these get different directories, so the resume logic in pipeline.py
    (skip when optimized.nc exists) cannot hand one config another's run.
    Pass result['params'] when the hyperparameters were overridden."""
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

    def tok(prefix, value):
        return prefix + str(value).replace(".", "p").replace("-", "m")

    init = np.datetime_as_string(np.datetime64(run["init_date"]), unit="D")
    weights = "x".join(tok("", f"{float(loss['lambda_weights'][k]):g}")
                       for k in WEIGHT_KEYS)
    name = "_".join([
        cfg["event"]["name"],
        "i" + init.replace("-", ""),
        tok("lr", f"{float(p['learning_rate']):.0e}"),
        tok("it", p["iterations"]),
        tok("lam", p["lam"]),
        tok("b", p["beta"]),
        "w" + weights,
        tok("d", f"{float(p['evol_days']):.0f}"),
    ])
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


def _nc_safe(value):
    """A netCDF-writable form of an attribute value, or None to drop it."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)          # netCDF has no boolean attribute type
    if isinstance(value, _NC_ATTR_TYPES):
        return value
    if isinstance(value, _NC_SEQ_TYPES):
        items = [_nc_safe(x) for x in value]
        if items and all(x is not None for x in items):
            return items
    return None


def drop_unwritable_attrs(ds):
    """Remove attributes netCDF cannot serialize, on the dataset and on every
    variable.

    dinosaur tags a decoded dataset with attrs such as
    basis_as_jax_arrays=None, and xarray refuses to write a None-valued attr
    ("Invalid value for attr ... its value must be of one of the following
    types"). Older xarray let it through, which is why this only shows up off
    Colab. Nothing downstream reads these, so dropping them loses nothing."""
    def keep(attrs):
        out = {}
        for k, v in attrs.items():
            v = _nc_safe(v)
            if v is not None:
                out[k] = v
        return out

    ds = ds.copy(deep=False)
    ds.attrs = keep(ds.attrs)
    for name, var in ds.variables.items():
        dropped = keep(var.attrs)
        if len(dropped) != len(var.attrs):
            ds[name].attrs = dropped
    return ds


def save_trajectory_nc(model, state, all_forcings, steps: int,
                       path: str | Path, variables: list[str] | None = None) -> None:
    """Unroll a state and write the trajectory to netCDF."""
    _, preds = model.unroll(state, all_forcings, steps=steps)
    ds = model.data_to_xarray(preds, times=np.arange(steps))
    if variables:
        ds = ds[variables]
    drop_unwritable_attrs(ds).to_netcdf(str(path))
