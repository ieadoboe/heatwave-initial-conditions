"""Heat-wave intensity and length as Whittaker & Di Luca (2026) define them.

Eq. 4 of the paper: a heat-wave event is a run of consecutive days on which
the region-mean temperature exceeds the 99th percentile of the 1981-2020 JJA
climatology of 6-hourly values over the same region; its intensity is the
mean exceedance over those days. The optimized run is compared with each
ensemble member on this quantity over the targeted 5-day window, and the
paper's +3.7 C is the optimized intensity minus the hottest member's.

No jax here: this runs on a laptop against the CSVs a run leaves behind.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ERA5 regridded conservatively to 5.625 deg (WeatherBench2). Conservative
# regridding preserves area means, so the box-mean series is close to the
# 0.25 deg one the paper used. Chunks hold 100 timesteps, so 40 JJA seasons
# of 6-hourly data are about 1.6 GB of download.
WB2_ERA5_5P6 = ("gs://weatherbench2/datasets/era5/"
                "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")

# The 4x4-cell loss box on the NeuralGCM 2.8 deg grid around Lytton, BC, as
# heatwave_ic.optimize slices it (lon_i-2:lon_i+2, lat_i-2:lat_i+2), given by
# its outer cell edges.
PNW_BOX = dict(lat=(44.6, 55.8), lon=(232.0, 243.3))


# Resolved against the checkout, not the working directory, so a notebook
# run from notebooks/ still finds the same cache as a script run from the root.
_DATA = Path(__file__).resolve().parents[1] / "data"


def box_mean_t1000_climatology(box=PNW_BOX, years=(1981, 2020),
                               cache=_DATA / "pnw_box_t1000_jja_1981_2020.csv",
                               store=WB2_ERA5_5P6) -> pd.Series:
    """6-hourly JJA box-mean T1000 (C) over the climatology years, cached
    to `cache` after the first download."""
    cache = Path(cache)
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    ds = xr.open_zarr(store, chunks=None, storage_options=dict(token="anon"))
    t = ds["temperature"].sel(level=1000)
    t = t.sel(latitude=slice(*box["lat"]), longitude=slice(*box["lon"]))
    pieces = []
    for y in range(years[0], years[1] + 1):
        sel = t.sel(time=slice(f"{y}-06-01", f"{y}-08-31T23"))
        w = np.cos(np.deg2rad(sel.latitude))
        pieces.append(sel.weighted(w).mean(("latitude", "longitude")).load())
        print(f"  {y}: {sel.sizes['time']} snapshots")
    series = xr.concat(pieces, "time").to_series() - 273.15
    series.name = "box_T1000_C"
    cache.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(cache)
    return series


def heatwave_events(daily: pd.Series, thresh: float) -> list[pd.Series]:
    """Split a daily series into runs of consecutive days above thresh."""
    above = daily > thresh
    events, current = [], []
    for day, hot in above.items():
        if hot:
            current.append(day)
        elif current:
            events.append(daily.loc[current])
            current = []
    if current:
        events.append(daily.loc[current])
    return events


def intensity(daily: pd.Series, thresh: float) -> dict:
    """Eq. 4 on one daily series: the mean exceedance over the longest event
    (the paper reports a single event per run), its length in days, and the
    peak day. Zero length when no day exceeds the threshold."""
    events = heatwave_events(daily, thresh)
    if not events:
        return {"intensity_C": 0.0, "length_d": 0, "peak_C": float(daily.max())}
    ev = max(events, key=len)
    return {"intensity_C": float((ev - thresh).mean()),
            "length_d": int(len(ev)),
            "peak_C": float(ev.max())}


def daily_mean(hourly: pd.Series) -> pd.Series:
    """Daily mean of an hourly (or 6-hourly) series, on calendar days."""
    return hourly.resample("1D").mean()


if __name__ == "__main__":
    # One runnable check: a planted 3-day event with known exceedances.
    idx = pd.date_range("2021-06-27", periods=5, freq="1D")
    d = pd.Series([20.0, 31.0, 33.0, 32.0, 20.0], index=idx)
    r = intensity(d, 30.0)
    assert r["length_d"] == 3 and abs(r["intensity_C"] - 2.0) < 1e-9, r
    assert intensity(d, 40.0)["length_d"] == 0
    # Two separate runs: the longer one is reported.
    d2 = pd.Series([31.0, 20.0, 31.0, 31.0, 20.0], index=idx)
    assert intensity(d2, 30.0)["length_d"] == 2
    print("OK: Eq. 4 intensity and event length")
