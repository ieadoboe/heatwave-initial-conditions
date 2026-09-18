"""Gridded heat-wave detection: CTX90pct plus space-time event labelling.

The index is Perkins & Alexander (2013) CTX90pct, the same definition
scripts/identify_heatwaves.py applies at a single point: a cell is in a heat
wave when its daily maximum temperature exceeds the calendar-day 90th
percentile of a baseline period for at least three consecutive days, where the
percentile is pooled over a 15-day window centred on that calendar day so the
threshold varies smoothly through the season.

Applying that to a grid gives a boolean (time, lat, lon) array of cell-days,
which is not yet a catalogue of events. `detect` turns it into one by
connected-component labelling: every group of hot cells that touches in space
or persists into the next day gets one id, so a spell that drifts from Iberia
into France stays a single event while an unrelated Nordic spell gets its own.

One subtlety decides whether that works. A 90th-percentile threshold puts
about a tenth of all cell-days above it by construction, so scattered
exceedances are everywhere, and labelling straight away lets them bridge
unrelated blobs together. Each day is therefore filtered for spatial extent
first, which is what `min_area_km2` controls.

Everything here is plain numpy and scipy on in-memory arrays. Europe at 0.25
degrees for one season is about 150 x 280 x 153, which is small.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

EARTH_R_KM = 6371.0088

_TMAX_NAMES = ("t2m", "2m_temperature", "mx2t", "tmax")
_TIME_NAMES = ("valid_time", "time", "date")


def open_daily_tmax(paths) -> xr.DataArray:
    """Open the downloaded daily-max files as one DataArray in degrees C.

    CDS names things inconsistently between catalogue entries and versions, so
    the variable and the time coordinate are found rather than assumed, and
    kelvin is detected from the values instead of trusted from an attribute."""
    paths = sorted(str(p) for p in paths)
    if not paths:
        raise FileNotFoundError("no daily-max files found — run "
                                "scripts/download_era5_europe_daily.py first")
    ds = xr.open_mfdataset(paths, combine="by_coords")

    var = next((v for v in _TMAX_NAMES if v in ds.data_vars), None)
    if var is None:
        raise KeyError(f"no temperature variable in {list(ds.data_vars)}; "
                       f"expected one of {_TMAX_NAMES}")
    da = ds[var]

    renames = {}
    time_dim = next((t for t in _TIME_NAMES if t in da.dims), None)
    if time_dim is None:
        raise KeyError(f"no time dimension in {da.dims}")
    if time_dim != "time":
        renames[time_dim] = "time"
    for short, long in (("lat", "latitude"), ("lon", "longitude")):
        if short in da.dims:
            renames[short] = long
    if renames:
        da = da.rename(renames)

    # CDS carries a scalar `number` (ensemble member) coordinate on reanalysis
    # output. Nothing here uses it and it can trip up concatenation across
    # files, so keep only the three coordinates that matter.
    da = da.drop_vars([c for c in da.coords
                       if c not in ("time", "latitude", "longitude")],
                      errors="ignore")

    da = da.sortby("time").load()
    if float(da.max()) > 100.0:          # kelvin, not celsius
        da = da - 273.15
    return da.astype("float32").rename("tmax_C")


def land_mask_for(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Land mask on an arbitrary lat/lon grid, from the Koppen map already in
    data/ (class 0 is ocean). Saves downloading a separate land-sea mask, and
    keeps ocean cells out of a land heat-wave catalogue."""
    from heatwave_ic.zones import koppen_grid

    classes, klat, klon, _ = koppen_grid()
    i = np.abs(klat[:, None] - lat[None, :]).argmin(axis=0)
    j = np.abs(klon[:, None] - (((lon + 180.0) % 360.0) - 180.0)[None, :]
               ).argmin(axis=0)
    return classes[np.ix_(i, j)] > 0


def cell_area_km2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Area of each grid cell, shape (lat, lon).

    Cells shrink towards the poles, so a raw cell count would make a Finnish
    event look larger than an equal-area Spanish one."""
    dlat = float(np.abs(np.diff(lat)).mean())
    dlon = float(np.abs(np.diff(lon)).mean())
    rad = np.pi / 180.0
    height = EARTH_R_KM * dlat * rad
    width = EARTH_R_KM * dlon * rad * np.cos(lat * rad)
    return np.repeat((height * width)[:, None], lon.size, axis=1)


def calendar_day_percentile(baseline: xr.DataArray, q: float = 0.90,
                            window: int = 7) -> xr.DataArray:
    """Percentile threshold per calendar day, pooled over +/- `window` days.

    Returns (dayofyear, lat, lon). With 30 baseline years and a 15-day window
    each threshold is estimated from 450 samples, which is what makes a 90th
    percentile stable enough to threshold against."""
    doy = baseline["time"].dt.dayofyear
    baseline = baseline.assign_coords(dayofyear=doy)
    target_doys = np.unique(doy.values)

    out = np.empty((target_doys.size, baseline.sizes["latitude"],
                    baseline.sizes["longitude"]), dtype="float32")
    values = baseline.values
    doy_values = doy.values
    for i, d in enumerate(target_doys):
        # Circular distance in calendar days, so the window works across the
        # year boundary even though this domain only uses May-September.
        delta = np.abs(doy_values - d)
        delta = np.minimum(delta, 366 - delta)
        sel = values[delta <= window]
        out[i] = np.nanquantile(sel, q, axis=0)

    return xr.DataArray(
        out,
        coords={"dayofyear": target_doys,
                "latitude": baseline["latitude"],
                "longitude": baseline["longitude"]},
        dims=("dayofyear", "latitude", "longitude"),
        name=f"tmax_p{int(q * 100)}",
    )


def exceedance(target: xr.DataArray, threshold: xr.DataArray) -> xr.DataArray:
    """Boolean (time, lat, lon): target above its calendar-day threshold."""
    thresh = threshold.sel(dayofyear=target["time"].dt.dayofyear)
    thresh = thresh.drop_vars("dayofyear", errors="ignore")
    return (target > thresh).rename("exceedance")


def sustained(mask: np.ndarray, min_days: int = 3) -> np.ndarray:
    """Keep only cell-days belonging to a run of `min_days` consecutive days.

    A binary opening along the time axis alone: erosion removes any day whose
    run is shorter than the structuring element, dilation restores the days of
    the runs that survived. This is the "3+ consecutive days" half of
    CTX90pct."""
    structure = np.ones((min_days, 1, 1), dtype=bool)
    return ndimage.binary_opening(mask, structure=structure)


def spatially_coherent(mask: np.ndarray, area: np.ndarray,
                       min_area_km2: float) -> np.ndarray:
    """Day by day, drop connected patches smaller than `min_area_km2`.

    This has to happen BEFORE the space-time labelling, not after. At the 90th
    percentile roughly a tenth of all cell-days exceed by construction, so
    there is a permanent background of scattered exceedances. Labelling first
    lets that background chain onto a genuine blob through single-cell
    bridges, which silently inflates the event's duration and extent: in
    testing, a planted 5-day event came back as 12 days. Requiring each day to
    be spatially extensive first removes the bridges."""
    out = np.zeros_like(mask)
    structure = ndimage.generate_binary_structure(2, 2)   # 8-connectivity
    for t in range(mask.shape[0]):
        labels, n = ndimage.label(mask[t], structure=structure)
        if n == 0:
            continue
        sums = ndimage.sum_labels(area, labels, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = sums >= min_area_km2
        out[t] = keep[labels]
    return out


def _connectivity() -> np.ndarray:
    """Neighbourhood for labelling: all 8 spatial neighbours within a day,
    and the same cell on the previous and next day.

    Full 26-neighbour connectivity would also join cells that are diagonal in
    space AND time, which merges events that only brush past one another."""
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True
    return structure


@dataclass
class Detection:
    """Everything the event table is built from."""
    labels: np.ndarray            # (time, lat, lon) int, 0 = no event
    n_events: int
    mask: np.ndarray              # sustained exceedance
    anomaly: np.ndarray           # target - threshold, degrees C
    time: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    area: np.ndarray              # (lat, lon) km^2


def detect(target: xr.DataArray, threshold: xr.DataArray,
           land_mask: np.ndarray | None = None,
           min_days: int = 3, min_area_km2: float = 50_000.0) -> Detection:
    """Threshold, require spatial coherence, require persistence, then label.

    The order is deliberate. Filtering scattered exceedances out of each day
    first stops them bridging separate events together during the space-time
    labelling; see `spatially_coherent`."""
    exc = exceedance(target, threshold)
    thresh = threshold.sel(dayofyear=target["time"].dt.dayofyear).values
    anomaly = (target.values - thresh).astype("float32")
    area = cell_area_km2(target["latitude"].values, target["longitude"].values)

    mask = exc.values
    if land_mask is not None:
        mask &= land_mask[None, :, :]
    if min_area_km2:
        mask = spatially_coherent(mask, area, min_area_km2)
    mask = sustained(mask, min_days=min_days)

    labels, n = ndimage.label(mask, structure=_connectivity())
    return Detection(
        labels=labels, n_events=int(n), mask=mask, anomaly=anomaly,
        time=pd.DatetimeIndex(target["time"].values),
        lat=target["latitude"].values, lon=target["longitude"].values,
        area=area,
    )


def event_table(det: Detection, target: xr.DataArray,
                min_area_km2: float = 50_000.0) -> pd.DataFrame:
    """One row per event, ranked by severity.

    severity_C_km2_day is the area-weighted sum of the exceedance anomaly over
    every cell-day in the event. It is the gridded analogue of Russo's heat
    wave magnitude: an event scores highly by being hot, large and long, so a
    brief national record and a mild continental month are not conflated.

    Events smaller than min_area_km2 at their largest are dropped; without
    that the table fills with single-cell specks on coastlines and mountains.
    """
    tmax = target.values
    rows = []
    # objects gives each label's bounding box, so per-event work touches only
    # its own slice instead of the whole domain.
    for eid, sl in enumerate(ndimage.find_objects(det.labels), start=1):
        if sl is None:
            continue
        sub_labels = det.labels[sl] == eid
        sub_anom = det.anomaly[sl]
        sub_tmax = tmax[sl]
        sub_area = det.area[sl[1], sl[2]]

        area_per_day = (sub_labels * sub_area[None, :, :]).sum(axis=(1, 2))
        peak_area = float(area_per_day.max())
        if peak_area < min_area_km2:
            continue

        times = det.time[sl[0]]
        anom_in = np.where(sub_labels, sub_anom, np.nan)
        tmax_in = np.where(sub_labels, sub_tmax, np.nan)
        severity = float(np.nansum(np.where(sub_labels, sub_anom, 0.0)
                                   * sub_area[None, :, :]))

        # Peak day: the largest area-weighted mean anomaly, which is the day
        # the event was most intense rather than merely most widespread.
        daily_mean_anom = np.nanmean(anom_in.reshape(anom_in.shape[0], -1), axis=1)
        peak_i = int(np.nanargmax(daily_mean_anom))

        peak_day_mask = sub_labels[peak_i]
        w = peak_day_mask * sub_area
        lat_sub = det.lat[sl[1]]
        lon_sub = det.lon[sl[2]]
        total_w = w.sum()
        centroid_lat = float((w.sum(axis=1) * lat_sub).sum() / total_w)
        centroid_lon = float((w.sum(axis=0) * lon_sub).sum() / total_w)

        hottest = np.unravel_index(np.nanargmax(tmax_in), tmax_in.shape)

        rows.append({
            "event_id": eid,
            "start": times[0].date(),
            "peak": times[peak_i].date(),
            "end": times[-1].date(),
            "duration_days": int(sub_labels.shape[0]),
            "peak_area_km2": round(peak_area),
            "mean_area_km2": round(float(area_per_day[area_per_day > 0].mean())),
            "centroid_lat": round(centroid_lat, 2),
            "centroid_lon": round(centroid_lon, 2),
            "peak_tmax_C": round(float(np.nanmax(tmax_in)), 2),
            "peak_tmax_lat": round(float(lat_sub[hottest[1]]), 2),
            "peak_tmax_lon": round(float(lon_sub[hottest[2]]), 2),
            "max_anom_C": round(float(np.nanmax(anom_in)), 2),
            "mean_anom_C": round(float(np.nanmean(anom_in)), 2),
            "severity_C_km2_day": round(severity),
            "n_cell_days": int(sub_labels.sum()),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("severity_C_km2_day", ascending=False, ignore_index=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
