"""
Initial-condition optimization for the Aug 2025 St. John's, NL heat wave.

Adapted from Tim Whittaker's NeuralGCM_example.py (ExtremeStorylines, Zenodo
10.5281/zenodo.15649394) — the optimizer behind Whittaker & Di Luca (2026).
This is a faithful copy with the corrections/changes below; Tim's archived code
in tim_code/ is intentionally left untouched (the line-208 bug still lives there
— worth flagging to Tim).

  1. BUG FIX — reg_term for cloud LIQUID now uses
     specific_cloud_liquid_water_content. The original
     (NeuralGCM_example.py:208) copy-pasted specific_cloud_ICE, so cloud-liquid
     was effectively unregularized.
  2. PARAMETERIZED — values Tim hardcoded (each marked `# TODO` in his code) are
     now in the config: T_ref, the per-variable regularization weights (were
     ×100 / ×10 multipliers), the target-averaging window (was -5*24), and the
     intermediate-save interval (was step % 100).
  3. ST JOHN'S TARGET — target_lat / target_lon default to St. John's Intl A;
     the loss still averages a 4×4 grid box around the nearest model cell.
  4. IC BUILDER — `--build-ic` slices the model's input + forcing variables from
     ARCO-ERA5 at init_date and writes the IC zarr the run regrids (mirrors the
     pipeline in scripts/heatwave_leadtime_sweep.py).

Loss (unchanged form — Eq 3 of the paper, with the code's sqrt):
    L = (beta * T_ref) / sqrt(mean(T_box, last window))
        + lambda * sum_i  w_i * mean((dx_i)^2) / mean(x0_i)^2

NOTE on tuning (see docs/meeting_notes.md): T_ref is degenerate with beta (only
the product beta*T_ref matters); the reference scale mean(x0_i)^2 is fragile for
near-zero-mean fields (vorticity, divergence, temperature anomaly).

RUNTIME: needs a GPU and GCS access — run on Colab, not the local CPU venv.
UNTESTED locally.

Outputs (under output_dir): optimized.nc, original.nc, losses.npy, plus
optimized/original log-surface-pressure, vorticity, divergence .npy fields.
Plot: plots/stjohns_opt_loss.pdf
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import gcsfs
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import optax
import xarray
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
import neuralgcm

# Project plotting convention (CLAUDE.md).
sys.path.insert(0, str(Path(__file__).parent))
from common import mpl_apply  # noqa: E402

mpl_apply()

gcs = gcsfs.GCSFileSystem(token="anon")


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(config):
    """Load a NeuralGCM checkpoint and patch its decoder to also output surface
    pressure. The config-string surgery below is the one Tim used for the
    stochastic-precip model; it likely needs adjusting for a deterministic
    checkpoint (untested)."""
    model_name = config["model_name"]
    with gcs.open(f"gs://neuralgcm/models/{model_name}", "rb") as f:
        ckpt = pickle.load(f)

    new_inputs_to_units_mapping = {
        "u": "meter / second",
        "v": "meter / second",
        "t": "kelvin",
        "z": "m**2 s**-2",
        "sim_time": "dimensionless",
        "tracers": {
            "specific_humidity": "dimensionless",
            "specific_cloud_liquid_water_content": "dimensionless",
            "specific_cloud_ice_water_content": "dimensionless",
        },
        "diagnostics": {"surface_pressure": "kg / (meter s**2)"},
    }
    new_model_config_str = "\n".join([
        ckpt["model_config_str"],
        "DimensionalLearnedPrimitiveToWeatherbenchDecoder.inputs_to_units_mapping"
        f" = {new_inputs_to_units_mapping}",
        "DimensionalLearnedPrimitiveToWeatherbenchDecoder.diagnostics_module ="
        " @NodalModelDiagnosticsDecoder",
        "StochasticPhysicsParameterizationStep.diagnostics_module ="
        " @SurfacePressureDiagnostics",
    ])
    ckpt["model_config_str"] = new_model_config_str
    return neuralgcm.PressureLevelModel.from_checkpoint(ckpt)


def build_initial_condition(config):
    """Slice the model's input + forcing variables for init_date from ARCO-ERA5
    and write the IC zarr that main() regrids onto the model grid. Forcings
    (SST / sea-ice) are time-shifted 24 h, as the model expects."""
    model = load_model(config)
    init = config["init"]
    t0 = np.datetime64(init["init_date"])
    print(f"Opening ARCO-ERA5 and slicing the IC window at {t0} ...")
    full = xarray.open_zarr(
        init["era5_arco"], chunks=None, storage_options=dict(token="anon")
    )
    window = full[model.input_variables + model.forcing_variables].sel(
        time=slice(t0, t0 + np.timedelta64(2, "D"))
    )
    window = window.pipe(
        xarray_utils.selective_temporal_shift,
        variables=model.forcing_variables,
        time_shift="24 hours",
    )
    out = Path(init["init_cond"])
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Materialising {window.sizes['time']} snapshots -> {out} ...")
    window.compute().to_zarr(str(out), mode="w")
    print("IC built.")


def make_output_dir(config):
    opt, loss = config["optimizer"], config["loss"]
    parts = [
        f"lr{float(opt['learning_rate']):.0e}",
        f"it{opt['iteration_number']}",
        f"lam{loss['lambda']}",
        f"b{loss['beta']}",
        f"d{config['evol_days']}",
        "stjohns",
    ]
    name = "_".join(p.replace(".", "p").replace("-", "m") for p in parts)
    path = os.path.join(config["output_dir"], name)
    os.makedirs(path, exist_ok=True)
    return path


def main(config):
    model = load_model(config)
    output_dir = make_output_dir(config)

    # Load + regrid the IC onto the model grid.
    sliced_era5 = xarray.open_zarr(config["init"]["init_cond"], chunks=None)
    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=sliced_era5.sizes["latitude"],
        longitude_nodes=sliced_era5.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sliced_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sliced_era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )
    eval_era5 = xarray_utils.fill_nan_with_nearest(
        xarray_utils.regrid(sliced_era5, regridder)
    )

    # Initialize model state.
    inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
    rng_key = jax.random.key(int(config["rng_seed"]))
    initial_state = model.encode(inputs, input_forcings, rng_key)
    all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

    def extract_non_diff(state):
        components = (state.randomness, state.state.sim_time, state.memory)
        new_inner = state.state.replace(sim_time=None)
        return state.replace(state=new_inner, randomness=None, memory=None), components

    def reconstruct_full_state(state_wo, non_diff):
        randomness, sim_time, memory = non_diff
        sim_time = jax.lax.stop_gradient(sim_time)
        new_inner = state_wo.state.replace(sim_time=sim_time)
        return state_wo.replace(randomness=randomness, state=new_inner, memory=memory)

    # Target domain (longitude given in degrees WEST -> 0-360 convention).
    target_lat = float(config["target_lat"])
    target_lon_pos = (360 - float(config["target_lon"])) % 360
    lat_i = int(np.abs(eval_era5.latitude.values - target_lat).argmin())
    lon_i = int(np.abs(eval_era5.longitude.values - target_lon_pos).argmin())

    outer_steps = int(round(float(config["evol_days"]) * 24))
    window_steps = int(round(float(config["loss"]["target_window_days"]) * 24))
    beta = float(config["loss"]["beta"])
    lam = float(config["loss"]["lambda"])
    t_ref = float(config["loss"]["T_ref"])
    w = config["loss"]["lambda_weights"]

    @jax.jit
    def compute_loss(diff_state, non_diff, initial_diff_state):
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

        def reg(name, cur, init):
            # Reference scale = (mean of the initial field)^2, as in Tim's code.
            return w[name] * jnp.mean((cur - init) ** 2) / (jnp.mean(init) ** 2)

        reg_total = (
            reg("log_surface_pressure", d0.log_surface_pressure, i0.log_surface_pressure)
            + reg("divergence", d0.divergence, i0.divergence)
            + reg("vorticity", d0.vorticity, i0.vorticity)
            + reg("temperature_variation", d0.temperature_variation, i0.temperature_variation)
            + reg("specific_humidity",
                  d0.tracers["specific_humidity"], i0.tracers["specific_humidity"])
            + reg("specific_cloud_ice_water_content",
                  d0.tracers["specific_cloud_ice_water_content"],
                  i0.tracers["specific_cloud_ice_water_content"])
            # BUG FIX: cloud LIQUID (the original used ice here too).
            + reg("specific_cloud_liquid_water_content",
                  d0.tracers["specific_cloud_liquid_water_content"],
                  i0.tracers["specific_cloud_liquid_water_content"])
        )

        final_temp = jnp.mean(
            temp_traj[-window_steps:, 0, -1, lon_i - 2:lon_i + 2, lat_i - 2:lat_i + 2]
        )
        loss = (beta * t_ref) / jnp.sqrt(jnp.mean(final_temp)) + lam * reg_total
        return loss, final_temp

    optimizer = optax.adam(learning_rate=float(config["optimizer"]["learning_rate"]))

    @jax.jit
    def update_step(diff_state, opt_state, non_diff, initial_diff_state):
        (loss, temp), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            diff_state, non_diff, initial_diff_state
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        diff_state = optax.apply_updates(diff_state, updates)
        return diff_state, opt_state, loss, temp

    initial_diff_state, initial_non_diff = extract_non_diff(initial_state)
    opt_state = optimizer.init(initial_diff_state)
    current_diff, current_non_diff = initial_diff_state, initial_non_diff
    times = np.arange(outer_steps)
    losses = []

    save_every = int(config["save_every_iters"])
    n_iter = int(config["optimizer"]["iteration_number"])
    pbar = tqdm(range(n_iter), desc="Optimizing")
    for step in pbar:
        current_diff, opt_state, loss, temp = update_step(
            current_diff, opt_state, current_non_diff, initial_diff_state
        )
        losses.append(float(loss))
        full_state = reconstruct_full_state(current_diff, current_non_diff)
        _, current_non_diff = extract_non_diff(full_state)
        pbar.set_description(f"loss {float(loss):.4f}  mean T {float(temp):.4f}")

        if save_every and step % save_every == 0:
            st = reconstruct_full_state(current_diff, current_non_diff)
            _, preds = model.unroll(st, all_forcings, steps=outer_steps, start_with_input=True)
            ds = model.data_to_xarray(preds, times=times)[["temperature", "geopotential"]]
            ds.to_netcdf(f"{output_dir}/optimized_step{step}.nc")
            del ds, st, preds

    print("Training complete.")
    losses = np.array(losses)
    np.save(f"{output_dir}/losses", losses)

    def save_state_fields(state, tag):
        h = model.model_coords.horizontal
        sp = model.from_nondim_units(
            jnp.squeeze(jnp.exp(h.to_nodal(state.state.log_surface_pressure)), axis=0),
            "kg / (meter s**2)",
        )
        vort = model.from_nondim_units(h.to_nodal(state.state.vorticity), "1/s")
        div = model.from_nondim_units(h.to_nodal(state.state.divergence), "1/s")
        np.save(f"{output_dir}/log_surface_pressure_{tag}", sp)
        np.save(f"{output_dir}/vorticity_{tag}", vort)
        np.save(f"{output_dir}/divergence_{tag}", div)

    # Optimized trajectory + fields.
    optimized_state = reconstruct_full_state(current_diff, current_non_diff)
    _, preds = model.unroll(optimized_state, all_forcings, steps=outer_steps)
    model.data_to_xarray(preds, times=times).to_netcdf(f"{output_dir}/optimized.nc")
    save_state_fields(optimized_state, "opt")

    # Original (unperturbed) trajectory + fields.
    _, preds = model.unroll(initial_state, all_forcings, steps=outer_steps)
    model.data_to_xarray(preds, times=times).to_netcdf(f"{output_dir}/original.nc")
    save_state_fields(initial_state, "original")
    print("Trajectories saved!")

    # Loss-curve diagnostic (plots/ per CLAUDE.md).
    plots_dir = Path(config["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, marker="o", ms=3)
    ax.set(xlabel="iteration", ylabel="loss",
           title="St. John's IC optimization --- loss")
    ax.grid(alpha=0.3)
    fig.savefig(plots_dir / "stjohns_opt_loss.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Loss curve -> {plots_dir / 'stjohns_opt_loss.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--build-ic", action="store_true",
                        help="Build the ERA5 IC zarr from ARCO-ERA5, then exit")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.build_ic:
        build_initial_condition(cfg)
    else:
        main(cfg)
