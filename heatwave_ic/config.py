"""Event-config loading and resolution.

A config YAML describes ONE heat-wave event + the optimization hyperparameters
(see configs/stjohns_aug2025.yaml for the annotated schema). `load_config`
returns a plain dict with everything downstream code needs already resolved:

  - event dates parsed to np.datetime64
  - target_lon accepted as SIGNED degrees East (-52.71 == 52.71°W) and also
    exposed 0-360 as event.target_lon_east (the model-grid convention)
  - run.init_date  = explicit value, or event.peak - run.lead_days
  - run.evol_days  = explicit value, or (event.end - init_date) in days
  - run.lead_days  = (event.peak - init_date) in days (derived if init given)
  - paths.ic_zarr  defaults to data/era5_ic_{event.name}_{init_date}.zarr
"""

from pathlib import Path

import numpy as np
import yaml

# Kept here (not data.py) so configs load without the heavy jax/dinosaur deps.
ARCO_ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

_DAY = np.timedelta64(1, "D")


def load_config(path: str | Path) -> dict:
    """Read an event YAML and resolve derived fields. Returns a nested dict."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return resolve_config(raw)


def resolve_config(cfg: dict) -> dict:
    cfg = dict(cfg)  # shallow copy; nested blocks are edited in place below.

    event = cfg["event"]
    for key in ("start", "peak", "end"):
        event[key] = np.datetime64(str(event[key]))
    event["target_lat"] = float(event["target_lat"])
    event["target_lon"] = float(event["target_lon"])  # signed degrees East
    event["target_lon_east"] = event["target_lon"] % 360.0

    run = cfg.setdefault("run", {})
    if run.get("init_date") is not None:
        run["init_date"] = np.datetime64(str(run["init_date"]))
        run["lead_days"] = float((event["peak"] - run["init_date"]) / _DAY)
    else:
        lead = float(run["lead_days"])
        run["init_date"] = event["peak"] - np.timedelta64(int(round(lead)), "D")
        run["lead_days"] = lead
    if run.get("evol_days") is None:
        run["evol_days"] = float((event["end"] - run["init_date"]) / _DAY)
    run["evol_days"] = float(run["evol_days"])
    run.setdefault("rng_seed", 42)
    run.setdefault("save_every_iters", 0)  # 0 = no intermediate trajectories

    loss = cfg["loss"]
    if loss["target_window_days"] > run["evol_days"]:
        raise ValueError(
            f"target_window_days ({loss['target_window_days']}) exceeds "
            f"evol_days ({run['evol_days']})"
        )
    if run["init_date"] + np.timedelta64(int(round(run["evol_days"])), "D") < event["end"]:
        raise ValueError(
            "init_date + evol_days ends before event.end — the target window "
            "would miss the event"
        )

    paths = cfg.setdefault("paths", {})
    paths.setdefault("era5_arco", ARCO_ERA5_PATH)
    init_str = np.datetime_as_string(run["init_date"], unit="D")
    paths.setdefault("ic_zarr", f"data/era5_ic_{event['name']}_{init_str}.zarr")
    paths.setdefault("output_dir", "data/opt_runs")
    paths.setdefault("plots_dir", "plots")
    return cfg


def describe(cfg: dict) -> str:
    """One-paragraph summary of a resolved config (print it in notebooks)."""
    event, run, loss, opt = cfg["event"], cfg["run"], cfg["loss"], cfg["optimizer"]
    return (
        f"{event['name']}: target ({event['target_lat']:.3f}, "
        f"{event['target_lon']:.3f}°E), event {event['start']} → {event['end']} "
        f"(peak {event['peak']})\n"
        f"  init {run['init_date']} (lead {run['lead_days']:.0f} d), "
        f"evolve {run['evol_days']:.0f} d, target window = last "
        f"{loss['target_window_days']} d\n"
        f"  beta={loss['beta']} lambda={loss['lambda']} T_ref={loss['T_ref']} "
        f"lr={float(opt['learning_rate']):.0e} iters={opt['iteration_number']}\n"
        f"  model {cfg['model_name']}  IC {cfg['paths']['ic_zarr']}"
    )
