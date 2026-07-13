"""
Initialization-lead-time sensitivity sweep for the August 2025 St. John's heat wave.

Question (per Elsa's email): can NeuralGCM reproduce a real, observed heat wave,
and from how far ahead? We initialise the deterministic 2.8° model from ERA5 at a
range of lead times before the event peak and integrate each forecast forward to
cover the event window, then measure how well each forecast captures the heat wave
at St. John's. The expectation is that the longest leads (~20 d) "miss" the event
(it is past the deterministic predictability limit) and that skill improves as the
lead shortens. The crossover lead time is the deliverable — and it defines the
window in which the later worst-case initial-condition optimisation is meaningful.

EVENT — St. John's Intl A (ECCC Climate ID 8403505) observed daily Tmax (°C):
    Aug 9: 30.5 | 10: 27.7 | 11: 28.1 | 12: 30.9 | 13: 30.3 | 14: 27.8 | 15: 29.2
  Station peak 2025-08-12 (30.9 °C); the regional-mean peak was 2025-08-13
  (ClimateData.ca). Hot stretch Aug 9–15, then a sharp drop to 13.0 °C on Aug 16
  — the Avalon Peninsula's longest-ever heat warning (7 days). These are the
  observational ground truth; the model-grid verification below uses ERA5.

PIPELINE — mirrors notebooks/climate_sim_neuralgcm_stjohns-2020.ipynb:
  ERA5 from ARCO-ERA5 zarr (GCS) → conservative regrid 0.25°→2.8° (month-chunked)
  → model.encode(IC at init date) → model.unroll(forward to event end).

RUNTIME: needs a GPU and GCS access — run on the cluster, not the local CPU venv.
This is a DRAFT to review/run there; it has not been executed locally.

Outputs:
  data/heatwave_leadtime_stjohns_aug2025.nc   per-lead St. John's daily-max T1000
  data/heatwave_leadtime_skill_aug2025.csv    skill vs lead time
  plots/heatwave_leadtime_trajectories.pdf     spaghetti of forecasts vs ERA5
  plots/heatwave_leadtime_skill.pdf            skill metric vs lead time
"""

import gc
import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Project plotting convention (CLAUDE.md).
sys.path.insert(0, str(Path(__file__).parent))
from common import mpl_apply  # noqa: E402

mpl_apply()

# ──────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ──────────────────────────────────────────────────────────────────────────

MODEL_NAME = "v1/deterministic_2_8_deg.pkl"   # same checkpoint as the 2020 run
ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# Event definition — anchored on the St. John's station peak (Aug 12, 30.9 °C);
# window spans the observed hot stretch (Aug 9–15) before the Aug 16 crash.
EVENT_PEAK = np.datetime64("2025-08-12")
EVENT_START = np.datetime64("2025-08-09")
EVENT_END = np.datetime64("2025-08-15")

# Lead times (days before EVENT_PEAK) at which to initialise. Elsa's set plus a
# couple of shorter leads to bracket the expected crossover.
LEAD_TIMES_DAYS = [20, 18, 16, 14, 12, 10, 9, 7, 5, 3]

# Output cadence of each forecast, in hours. 6 h lets us take a daily MAX of the
# 1000 hPa temperature as a Tmax proxy (the 2.8° model has no true 2 m daily Tmax;
# T1000 is the near-surface proxy used in the 2020 run). Set to 24 for daily-only.
DATA_INNER_HOURS = 6

# Regrid-block size, in days. NeuralGCM needs the GLOBAL ERA5 field, so peak RAM
# is set by one block at 0.25° (~0.77 GB per snapshot × snapshots-per-block, all
# 37 levels). 2 days at 6-hourly ≈ 6 GB peak. Lower this if you hit OOM.
CHUNK_DAYS = 2

# St. John's, NL grid point (0–360 longitude convention, as in the 2020 run).
STJOHNS_LAT = 47.5615
STJOHNS_LON = 360.0 - 52.7126   # ≈ 307.29 °E

RNG_SEED = 42

DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")

# ──────────────────────────────────────────────────────────────────────────
# Load model + open ERA5 + build regridder (identical to the 2020 notebook)
# ──────────────────────────────────────────────────────────────────────────

import gcsfs   # noqa: E402
import pickle  # noqa: E402
import neuralgcm  # noqa: E402
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils  # noqa: E402

print(f"Loading model {MODEL_NAME} …")
gcs = gcsfs.GCSFileSystem(token="anon")
with gcs.open(f"gs://neuralgcm/models/{MODEL_NAME}", "rb") as f:
    ckpt = pickle.load(f)
model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

print("Opening ARCO-ERA5 (lazy) …")
full_era5 = xr.open_zarr(ERA5_PATH, chunks=None, storage_options=dict(token="anon"))

era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes["latitude"],
    longitude_nodes=full_era5.sizes["longitude"],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)

# Time-shift forcings (SST / sea-ice) by 24 h, as the model expects.
shifted_full = full_era5[model.input_variables + model.forcing_variables].pipe(
    xarray_utils.selective_temporal_shift,
    variables=model.forcing_variables,
    time_shift="24 hours",
)

# ──────────────────────────────────────────────────────────────────────────
# Load + regrid a [t0, t1] window to the model grid at DATA_INNER_HOURS cadence
# ──────────────────────────────────────────────────────────────────────────


def load_regridded(t0: np.datetime64, t1: np.datetime64,
                   inner_hours: int, chunk_days: int) -> xr.Dataset:
    """Subsample the window [t0, t1] to `inner_hours` cadence, then materialise
    (0.25°) and conservatively regrid to the 2.8° model grid in blocks of
    `chunk_days` days. Peak RAM is one block at 0.25°; the regridded blocks
    (2.8°) are tiny, so only the concat result persists."""
    sub = (shifted_full.sel(time=slice(t0, t1))
           .isel(time=slice(None, None, inner_hours)))          # lazy
    steps = max(1, chunk_days * (24 // inner_hours))
    n = sub.sizes["time"]
    pieces = []
    for i in range(0, n, steps):
        chunk = sub.isel(time=slice(i, i + steps)).compute()
        chunk_g = xarray_utils.regrid(chunk, regridder)
        chunk_g = xarray_utils.fill_nan_with_nearest(chunk_g)
        pieces.append(chunk_g)
        print(f"  block {i // steps + 1}: {chunk.sizes['time']:3d} snapshots regridded")
        del chunk
        gc.collect()
    ds = xr.concat(pieces, dim="time")
    del pieces
    gc.collect()
    return ds


# Load one window spanning the earliest init through the event end, plus a small
# tail buffer — every forecast is a sub-slice of this single regridded dataset.
buffer_days = 2
window_start = EVENT_PEAK - np.timedelta64(max(LEAD_TIMES_DAYS), "D")
window_end = EVENT_END + np.timedelta64(buffer_days, "D")
print(f"\nLoading + regridding ERA5 {window_start} → {window_end} "
      f"at {DATA_INNER_HOURS} h cadence …")
eval_era5 = load_regridded(window_start, window_end, DATA_INNER_HOURS, CHUNK_DAYS)
print(f"  → {eval_era5.sizes['time']} snapshots on the model grid.")


def daily_max_t1000_at_stjohns(ds: xr.Dataset) -> pd.Series:
    """St. John's nearest-grid-point 1000 hPa temperature (°C), daily MAX."""
    pt = ds.temperature.sel(level=1000).sel(
        latitude=STJOHNS_LAT, longitude=STJOHNS_LON, method="nearest"
    ) - 273.15
    return pt.resample(time="1D").max().to_pandas()


# ERA5 "truth" on the model grid (the fair forecast-skill reference).
era5_truth = daily_max_t1000_at_stjohns(eval_era5)

# ──────────────────────────────────────────────────────────────────────────
# Forecast sweep: one deterministic unroll per lead time
# ──────────────────────────────────────────────────────────────────────────


def run_forecast(init_time: np.datetime64) -> pd.Series:
    """Encode the IC at init_time and unroll forward to window_end.
    Returns the St. John's daily-max T1000 (°C) trajectory."""
    fc = eval_era5.sel(time=slice(init_time, window_end))
    n_steps = fc.sizes["time"]                       # start_with_input=True

    inputs = model.inputs_from_xarray(fc.isel(time=0))
    in_forcings = model.forcings_from_xarray(fc.isel(time=0))
    state = model.encode(inputs, in_forcings, jax.random.key(RNG_SEED))

    all_forcings = model.forcings_from_xarray(fc)    # perfect (ERA5) forcings
    timedelta = np.timedelta64(1, "h") * DATA_INNER_HOURS
    _, preds = model.unroll(
        state, all_forcings, steps=n_steps,
        timedelta=timedelta, start_with_input=True,
    )
    times = fc.time.values
    preds_ds = model.data_to_xarray(
        preds, times=np.arange(n_steps) * DATA_INNER_HOURS
    ).as_numpy()
    preds_ds = preds_ds.assign_coords(time=times)
    return daily_max_t1000_at_stjohns(preds_ds)


records, traj = [], {}
for lead in LEAD_TIMES_DAYS:
    init_time = EVENT_PEAK - np.timedelta64(lead, "D")
    print(f"\nForecast: lead {lead:2d} d  (init {init_time}) …")
    fc_series = run_forecast(init_time)
    traj[lead] = fc_series

    # Skill over the event window.
    win = slice(pd.Timestamp(EVENT_START), pd.Timestamp(EVENT_END))
    fc_w, obs_w = fc_series.loc[win], era5_truth.loc[win]
    common = fc_w.index.intersection(obs_w.index)
    fc_w, obs_w = fc_w.loc[common], obs_w.loc[common]

    peak_err = float(fc_w.max() - obs_w.max())       # forecast peak − ERA5 peak
    rmse = float(np.sqrt(np.mean((fc_w - obs_w) ** 2)))
    records.append({
        "lead_days": lead,
        "init_date": pd.Timestamp(init_time).date(),
        "fc_peak_C": round(float(fc_w.max()), 2),
        "era5_peak_C": round(float(obs_w.max()), 2),
        "peak_err_C": round(peak_err, 2),
        "window_rmse_C": round(rmse, 2),
    })
    print(f"  peak err {peak_err:+.2f} °C   window RMSE {rmse:.2f} °C")
    gc.collect()

skill = pd.DataFrame(records).sort_values("lead_days")

# ──────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────

DATA_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

skill.to_csv(DATA_DIR / "heatwave_leadtime_skill_aug2025.csv", index=False)
print(f"\nSkill table → {DATA_DIR/'heatwave_leadtime_skill_aug2025.csv'}")

traj_da = xr.concat(
    [xr.DataArray(s.values, coords={"time": s.index}, dims="time")
     for s in traj.values()],
    dim=pd.Index(list(traj.keys()), name="lead_days"),
)
xr.Dataset({"tmax_proxy_C": traj_da,
            "era5_tmax_proxy_C": xr.DataArray(
                era5_truth.values, coords={"time": era5_truth.index}, dims="time")
            }).to_netcdf(DATA_DIR / "heatwave_leadtime_stjohns_aug2025.nc")

# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────

# (1) Trajectory spaghetti: every forecast vs ERA5 truth.
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(era5_truth.index, era5_truth.values, color="black", lw=2.5,
        label="ERA5 (truth)", zorder=10)
cmap = plt.cm.viridis(np.linspace(0, 1, len(LEAD_TIMES_DAYS)))
for c, lead in zip(cmap, LEAD_TIMES_DAYS):
    s = traj[lead]
    ax.plot(s.index, s.values, color=c, lw=1.3, alpha=0.9,
            label=f"init −{lead} d")
ax.axvspan(pd.Timestamp(EVENT_START), pd.Timestamp(EVENT_END),
           color="red", alpha=0.08, label="event window")
ax.set_ylabel("St. John's daily-max T$_{1000}$ (°C)")
ax.set_title("NeuralGCM forecasts of the Aug 2025 St. John's heat wave by lead time")
ax.legend(ncol=2, fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "heatwave_leadtime_trajectories.pdf", bbox_inches="tight")
plt.close()

# (2) Skill vs lead time.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.axhline(0, color="grey", lw=1, ls="--")
ax1.plot(skill["lead_days"], skill["peak_err_C"], marker="o")
ax1.set(xlabel="lead time (days before peak)",
        ylabel="peak Tmax error (°C)\nforecast − ERA5",
        title="Peak error vs lead time")
ax1.invert_xaxis()        # short lead (skilful) on the right
ax1.grid(alpha=0.3)
ax2.plot(skill["lead_days"], skill["window_rmse_C"], marker="s", color="C3")
ax2.set(xlabel="lead time (days before peak)",
        ylabel="event-window RMSE (°C)",
        title="Event-window RMSE vs lead time")
ax2.invert_xaxis()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "heatwave_leadtime_skill.pdf", bbox_inches="tight")
plt.close()

print("\n── Skill summary ───────────────────────────")
print(skill.to_string(index=False))
print(f"\nPlots → {PLOTS_DIR}/heatwave_leadtime_{{trajectories,skill}}.pdf")
print("\nNOTE: 'Tmax' here is the daily MAX of 1000 hPa T (near-surface proxy); "
      "the 2.8° model has no true 2 m daily Tmax. Verification is on the model "
      "grid. For local realism also compare against native-0.25° ERA5 / the ECCC "
      "station Tmax, and consider a small ensemble (perturbed ICs or the "
      "stochastic model) to confirm the crossover is robust, not single-run luck.")
print("Done.")
