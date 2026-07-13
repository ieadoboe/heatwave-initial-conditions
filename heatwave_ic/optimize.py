"""Differentiable IC optimization (the W&DL storyline method).

Numerics are a faithful port of scripts/stjohns_optimize.py (which fixed the
cloud-liquid regularization bug in Tim's NeuralGCM_example.py:208) with the
hyperparameters passed as traced arguments, notebook-style, so sweeps reuse
one compiled loss instead of re-jitting per hyperparameter set.

Loss (Eq 3 of the paper, with the code's sqrt):
    L = (beta * T_ref) / sqrt(mean(T_box, last window))
        + lambda * sum_i  w_i * mean((dx_i)^2) / mean(x0_i)^2

NOTE: the reference scale mean(x0_i)^2 is fragile for near-zero-mean fields
(vorticity, divergence, temperature anomaly) — kept to match Tim; switching to
std/abs-mean would be a deliberate deviation (docs/meeting_notes.md).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from tqdm import tqdm

# Order matters: it defines the layout of the weight vector passed to the loss.
WEIGHT_KEYS = [
    "log_surface_pressure",
    "divergence",
    "vorticity",
    "temperature_variation",
    "specific_humidity",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
]
STATE_FIELDS = {
    "log_surface_pressure", "divergence", "vorticity", "temperature_variation",
}


def extract_non_diff(state):
    """Split a model state into (differentiable part, non-diff components)."""
    components = (state.randomness, state.state.sim_time, state.memory)
    new_inner = state.state.replace(sim_time=None)
    return state.replace(state=new_inner, randomness=None, memory=None), components


def reconstruct_full_state(state_wo, non_diff):
    randomness, sim_time, memory = non_diff
    sim_time = jax.lax.stop_gradient(sim_time)
    new_inner = state_wo.state.replace(sim_time=sim_time)
    return state_wo.replace(randomness=randomness, state=new_inner, memory=memory)


def encode_initial_state(model, eval_era5, rng_seed: int = 42):
    """Encode the first snapshot of a regridded IC dataset.
    Returns (initial_state, all_forcings) ready for unroll/optimization."""
    inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
    state = model.encode(inputs, input_forcings, jax.random.key(int(rng_seed)))
    all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))
    return state, all_forcings


def target_indices(eval_era5, target_lat: float, target_lon_east: float):
    """(lat_i, lon_i) of the model grid cell nearest the target point.
    target_lon_east is in the 0-360 convention (cfg event.target_lon_east)."""
    lat_i = int(np.abs(eval_era5.latitude.values - target_lat).argmin())
    lon_i = int(np.abs(eval_era5.longitude.values - target_lon_east).argmin())
    return lat_i, lon_i


def _field(inner, key):
    return getattr(inner, key) if key in STATE_FIELDS else inner.tracers[key]


def make_loss_fn(model, all_forcings, outer_steps: int, window_steps: int,
                 lat_i: int, lon_i: int, box_halfwidth: int = 2):
    """Build compute_loss(diff_state, non_diff, initial_diff_state,
    beta, lam, t_ref, wvec) -> (loss, (box_T_K, reg_total)).

    Returned un-jitted; optimize_ic wraps it in jit(value_and_grad(...)).
    The target is the box-mean of the LAST `window_steps` hours of the
    surface-level (index -1) temperature in a (2*halfwidth)^2 box."""
    hw = box_halfwidth

    def compute_loss(diff_state, non_diff, initial_diff_state, beta, lam, t_ref, wvec):
        full_state = reconstruct_full_state(diff_state, non_diff)

        def scan_fn(carry, _):
            new_state, preds = model.unroll(
                carry, all_forcings, steps=1, start_with_input=True
            )
            return new_state, preds["temperature"]

        _, temp_traj = lax.scan(
            jax.checkpoint(scan_fn), full_state, None, length=outer_steps
        )

        i0, d0 = initial_diff_state.state, diff_state.state
        reg_total = 0.0
        for k, key in enumerate(WEIGHT_KEYS):
            cur, init = _field(d0, key), _field(i0, key)
            # Reference scale = (mean of the initial field)^2, as in Tim's code.
            reg_total = reg_total + wvec[k] * jnp.mean((cur - init) ** 2) / (
                jnp.mean(init) ** 2
            )

        box_T = jnp.mean(
            temp_traj[-window_steps:, 0, -1,
                      lon_i - hw:lon_i + hw, lat_i - hw:lat_i + hw]
        )
        loss = (beta * t_ref) / jnp.sqrt(box_T) + lam * reg_total
        return loss, (box_T, reg_total)

    return compute_loss


def optimize_ic(model, initial_state, all_forcings, *,
                outer_steps: int, window_steps: int, lat_i: int, lon_i: int,
                beta: float, lam: float, t_ref: float, weights: dict,
                learning_rate: float, iterations: int,
                box_halfwidth: int = 2, on_step=None, progress: bool = True) -> dict:
    """Run Adam on the initial condition. Returns a dict with:
      losses, box_T_K, reg (np arrays per iteration),
      optimized_state, initial_state.

    on_step(step, loss, box_T_K, reg_total, get_state) is called every
    iteration; get_state() reconstructs the current full state on demand
    (use it to save intermediate trajectories)."""
    loss_fn = make_loss_fn(model, all_forcings, outer_steps, window_steps,
                           lat_i, lon_i, box_halfwidth)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    beta_j = jnp.float32(beta)
    lam_j = jnp.float32(lam)
    t_ref_j = jnp.float32(t_ref)
    wvec = jnp.asarray([float(weights[k]) for k in WEIGHT_KEYS], dtype=jnp.float32)

    optimizer = optax.adam(learning_rate=float(learning_rate))
    initial_diff_state, non_diff = extract_non_diff(initial_state)
    diff = initial_diff_state
    opt_state = optimizer.init(diff)

    losses, box_list, reg_list = [], [], []
    steps_iter = range(int(iterations))
    bar = tqdm(steps_iter, desc="Optimizing") if progress else steps_iter
    for step in bar:
        (loss, (box_T, reg)), grads = grad_fn(
            diff, non_diff, initial_diff_state, beta_j, lam_j, t_ref_j, wvec
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        diff = optax.apply_updates(diff, updates)
        full = reconstruct_full_state(diff, non_diff)
        _, non_diff = extract_non_diff(full)

        losses.append(float(loss))
        box_list.append(float(box_T))
        reg_list.append(float(reg))
        if progress:
            bar.set_description(f"loss {float(loss):.4f}  box T {float(box_T):.2f} K")
        if on_step is not None:
            _d, _n = diff, non_diff
            on_step(step, float(loss), float(box_T), float(reg),
                    lambda d=_d, n=_n: reconstruct_full_state(d, n))

    return {
        "losses": np.asarray(losses),
        "box_T_K": np.asarray(box_list),
        "reg": np.asarray(reg_list),
        "optimized_state": reconstruct_full_state(diff, non_diff),
        "initial_state": initial_state,
    }


def optimize_event(model, eval_era5, cfg: dict, *, on_step=None,
                   progress: bool = True, weights: dict | None = None,
                   **overrides) -> dict:
    """One-call optimization for a resolved event config.

    Hyperparameter overrides for sweeps (notebook-style), e.g.
        optimize_event(model, eval_era5, cfg, beta=20, iterations=30)
    Accepted overrides: beta, lam, t_ref, learning_rate, iterations,
    target_window_days, evol_days, rng_seed; `weights` merges into the
    config's lambda_weights.

    Returns optimize_ic's result dict, plus all_forcings / lat_i / lon_i /
    outer_steps (what downstream evaluation needs)."""
    loss_cfg, opt_cfg, run = cfg["loss"], cfg["optimizer"], cfg["run"]
    p = {
        "beta": loss_cfg["beta"],
        "lam": loss_cfg["lambda"],
        "t_ref": loss_cfg["T_ref"],
        "target_window_days": loss_cfg["target_window_days"],
        "learning_rate": opt_cfg["learning_rate"],
        "iterations": opt_cfg["iteration_number"],
        "evol_days": run["evol_days"],
        "rng_seed": run["rng_seed"],
    }
    unknown = set(overrides) - set(p)
    if unknown:
        raise TypeError(f"unknown override(s): {sorted(unknown)}")
    p.update(overrides)
    w = dict(loss_cfg["lambda_weights"])
    if weights:
        w.update(weights)

    initial_state, all_forcings = encode_initial_state(
        model, eval_era5, rng_seed=p["rng_seed"]
    )
    lat_i, lon_i = target_indices(
        eval_era5, cfg["event"]["target_lat"], cfg["event"]["target_lon_east"]
    )
    outer_steps = int(round(float(p["evol_days"]) * 24))
    window_steps = int(round(float(p["target_window_days"]) * 24))

    result = optimize_ic(
        model, initial_state, all_forcings,
        outer_steps=outer_steps, window_steps=window_steps,
        lat_i=lat_i, lon_i=lon_i,
        beta=p["beta"], lam=p["lam"], t_ref=p["t_ref"], weights=w,
        learning_rate=p["learning_rate"], iterations=p["iterations"],
        on_step=on_step, progress=progress,
    )
    result.update(all_forcings=all_forcings, lat_i=lat_i, lon_i=lon_i,
                  outer_steps=outer_steps, params=p)
    return result
