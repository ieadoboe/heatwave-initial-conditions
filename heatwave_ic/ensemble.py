"""Stochastic-physics ensemble baselines (the W&DL 75-member benchmark).

Each member encodes the SAME unperturbed initial condition with a different
random seed, so the ensemble samples NeuralGCM's learned stochastic physics —
the envelope of "how hot could this event have been from internal variability
alone". A storyline is convincing when the optimized-IC peak beats even the
hottest member (W&DL's +3.7 C is exactly this quantity), and the per-zone
ensemble spread is the natural-variability yardstick that makes gains
comparable across climate zones.

Members are forward-only unrolls (no gradients): ~75 members cost roughly a
quarter to a third of the 75-iteration optimization already spent per event.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from heatwave_ic.data import build_ic_zarr, load_ic_on_model_grid
from heatwave_ic.evaluate import box_t1000_trajectory
from heatwave_ic.optimize import encode_initial_state, target_indices
from heatwave_ic.outputs import make_run_dir


def run_ensemble(model, eval_era5, cfg: dict, n_members: int = 75,
                 progress: bool = True) -> pd.DataFrame:
    """Unroll n_members stochastic realizations of the unperturbed IC.
    Returns a DataFrame of hourly box-mean T1000 (°C) trajectories, one
    column per member. Member seeds are rng_seed + 0..n-1, so member_000
    reproduces the control run's seed."""
    event, run = cfg["event"], cfg["run"]
    lat_i, lon_i = target_indices(
        eval_era5, event["target_lat"], event["target_lon_east"])
    outer_steps = int(round(run["evol_days"] * 24))
    base_seed = int(run["rng_seed"])

    columns = {}
    members = range(n_members)
    bar = tqdm(members, desc=f"ensemble {event['name']}") if progress else members
    for i in bar:
        state, forcings = encode_initial_state(
            model, eval_era5, rng_seed=base_seed + i)
        columns[f"member_{i:03d}"] = box_t1000_trajectory(
            model, state, forcings, outer_steps, lat_i, lon_i, run["init_date"])
        del state, forcings
        gc.collect()
    return pd.DataFrame(columns)


def ensemble_metrics(ens: pd.DataFrame, opt_peak_C: float) -> dict:
    """Storyline-vs-ensemble metrics from member trajectories.
    Peaks are per-member maxima of the box-mean T1000 trajectory."""
    peaks = ens.max(axis=0)
    spread = float(peaks.std(ddof=1))
    return {
        "n_members": int(ens.shape[1]),
        "ens_mean_peak_C": round(float(peaks.mean()), 2),
        "ens_max_peak_C": round(float(peaks.max()), 2),
        "ens_peak_spread_C": round(spread, 2),
        # W&DL's quantity: optimized peak minus the HOTTEST member's peak.
        "gain_vs_ens_max_C": round(float(opt_peak_C - peaks.max()), 2),
        # Cross-zone-comparable: gain in units of the zone's own variability.
        "gain_over_spread": round(float((opt_peak_C - peaks.mean()) / spread), 2),
    }


def run_event_ensemble(cfg: dict, model=None, *, n_members: int = 75,
                       skip_existing: bool = True, progress: bool = True) -> dict:
    """Ensemble baseline for one event whose optimization has already run.

    Needs the run dir's storyline.csv (for the optimized peak). Saves
    ensemble.csv next to it, writes the ensemble figure, and returns a
    summary row with the ensemble metrics. skip_existing reuses a saved
    ensemble.csv when it already has >= n_members members."""
    event = cfg["event"]
    name = event["name"]
    out_dir = Path(make_run_dir(cfg))
    summary = {"event": name, "zone": event.get("zone", ""),
               "run_dir": str(out_dir)}

    storyline_path = out_dir / "storyline.csv"
    if not storyline_path.exists():
        summary["status"] = "no storyline.csv — run the optimization first"
        return summary
    storyline = pd.read_csv(storyline_path, index_col=0, parse_dates=True)
    opt_peak = float(storyline["optimized_C"].max())

    ens_path = out_dir / "ensemble.csv"
    if skip_existing and ens_path.exists():
        ens = pd.read_csv(ens_path, index_col=0, parse_dates=True)
        if ens.shape[1] >= n_members:
            summary.update(status="skipped (existing ensemble)",
                           **ensemble_metrics(ens.iloc[:, :n_members], opt_peak))
            return summary

    if model is None:
        from heatwave_ic.model import load_model
        model = load_model(cfg["model_name"])
    build_ic_zarr(model, cfg)
    eval_era5 = load_ic_on_model_grid(model, cfg["paths"]["ic_zarr"])

    ens = run_ensemble(model, eval_era5, cfg, n_members=n_members,
                       progress=progress)
    ens.to_csv(ens_path)
    summary.update(status="ok", **ensemble_metrics(ens, opt_peak))

    from heatwave_ic import plots  # deferred: pulls in matplotlib/common
    plots.plot_ensemble_storyline(
        ens, storyline["unperturbed_C"], storyline["optimized_C"],
        event["start"], event["end"],
        title=f"{name} storyline vs {ens.shape[1]}-member ensemble",
        save=Path(cfg["paths"]["plots_dir"]) / f"{name}_ensemble.pdf")
    import matplotlib.pyplot as plt
    plt.close("all")
    return summary
