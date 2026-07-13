"""Trajectory evaluation: unrolls, target-point series, skill, lead-time sweep."""

import gc

import jax
import numpy as np
import pandas as pd
import xarray as xr


def unroll_to_xarray(model, state, all_forcings, steps: int,
                     init_time=None, inner_hours: int = 1,
                     variables: list[str] | None = None) -> xr.Dataset:
    """Unroll `steps` and return an xarray Dataset; if init_time is given the
    time coordinate is real datetimes at `inner_hours` cadence."""
    timedelta = np.timedelta64(1, "h") * inner_hours
    _, preds = model.unroll(state, all_forcings, steps=steps, timedelta=timedelta)
    ds = model.data_to_xarray(preds, times=np.arange(steps) * inner_hours)
    if variables:
        ds = ds[variables]
    if init_time is not None:
        times = np.datetime64(init_time) + np.arange(steps) * timedelta
        ds = ds.as_numpy().assign_coords(time=times)
    return ds


def box_t1000_trajectory(model, state, all_forcings, steps: int,
                         lat_i: int, lon_i: int, init_date,
                         box_halfwidth: int = 2) -> pd.Series:
    """Box-mean surface-level (index -1) temperature (°C), hourly — the same
    box the loss objective averages. Returns a pandas Series on real dates."""
    hw = box_halfwidth
    _, preds = model.unroll(state, all_forcings, steps=steps)
    T = np.asarray(preds["temperature"])[:, -1,
                                         lon_i - hw:lon_i + hw,
                                         lat_i - hw:lat_i + hw].mean((1, 2)) - 273.15
    t = np.datetime64(init_date) + np.arange(T.shape[0]).astype("timedelta64[h]")
    return pd.Series(T, index=pd.to_datetime(t))


def daily_max_t1000_at(ds: xr.Dataset, target_lat: float,
                       target_lon_east: float) -> pd.Series:
    """Nearest-grid-point 1000 hPa temperature (°C), daily MAX — the Tmax
    proxy (the 2.8° model has no true 2 m daily Tmax)."""
    pt = ds.temperature.sel(level=1000).sel(
        latitude=target_lat, longitude=target_lon_east, method="nearest"
    ) - 273.15
    return pt.resample(time="1D").max().to_pandas()


def event_skill(fc: pd.Series, truth: pd.Series, event_start, event_end) -> dict:
    """Peak error and window RMSE of a forecast series vs a truth series over
    the event window (both daily series, °C)."""
    win = slice(pd.Timestamp(event_start), pd.Timestamp(event_end))
    fc_w, obs_w = fc.loc[win], truth.loc[win]
    common = fc_w.index.intersection(obs_w.index)
    fc_w, obs_w = fc_w.loc[common], obs_w.loc[common]
    return {
        "fc_peak_C": round(float(fc_w.max()), 2),
        "truth_peak_C": round(float(obs_w.max()), 2),
        "peak_err_C": round(float(fc_w.max() - obs_w.max()), 2),
        "window_rmse_C": round(float(np.sqrt(np.mean((fc_w - obs_w) ** 2))), 2),
    }


def run_leadtime_sweep(model, eval_era5: xr.Dataset, cfg: dict,
                       lead_times_days: list[int],
                       inner_hours: int = 6) -> tuple[pd.DataFrame, dict]:
    """Deterministic forecast per lead time (days before event peak), verified
    against ERA5-on-model-grid over the event window.

    eval_era5 must be a regridded dataset (see data.regrid_window) spanning
    max(lead) days before the peak through past event end, at `inner_hours`
    cadence. Returns (skill DataFrame, {lead: daily-max series})."""
    event, run = cfg["event"], cfg["run"]
    lat, lon = event["target_lat"], event["target_lon_east"]
    window_end = eval_era5.time.values[-1]
    era5_truth = daily_max_t1000_at(eval_era5, lat, lon)

    records, traj = [], {}
    for lead in lead_times_days:
        init_time = event["peak"] - np.timedelta64(lead, "D")
        print(f"Forecast: lead {lead:2d} d  (init {init_time}) ...")
        fc = eval_era5.sel(time=slice(init_time, window_end))
        n_steps = fc.sizes["time"]                       # start_with_input=True

        inputs = model.inputs_from_xarray(fc.isel(time=0))
        in_forcings = model.forcings_from_xarray(fc.isel(time=0))
        state = model.encode(inputs, in_forcings, jax.random.key(int(run["rng_seed"])))

        all_forcings = model.forcings_from_xarray(fc)    # perfect (ERA5) forcings
        timedelta = np.timedelta64(1, "h") * inner_hours
        _, preds = model.unroll(
            state, all_forcings, steps=n_steps,
            timedelta=timedelta, start_with_input=True,
        )
        preds_ds = model.data_to_xarray(
            preds, times=np.arange(n_steps) * inner_hours
        ).as_numpy().assign_coords(time=fc.time.values)
        fc_series = daily_max_t1000_at(preds_ds, lat, lon)
        traj[lead] = fc_series

        skill = event_skill(fc_series, era5_truth, event["start"], event["end"])
        records.append({
            "lead_days": lead,
            "init_date": pd.Timestamp(init_time).date(),
            **skill,
        })
        print(f"  peak err {skill['peak_err_C']:+.2f} °C   "
              f"window RMSE {skill['window_rmse_C']:.2f} °C")
        del preds, preds_ds
        gc.collect()

    return pd.DataFrame(records).sort_values("lead_days"), traj
