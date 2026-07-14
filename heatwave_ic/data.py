"""ARCO-ERA5 access, IC building and regridding onto the model grid."""

import gc
from pathlib import Path

import numpy as np
import xarray as xr

from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

from heatwave_ic.config import ARCO_ERA5_PATH


def open_arco_era5(path: str = ARCO_ERA5_PATH) -> xr.Dataset:
    """Open the ARCO-ERA5 zarr lazily (anonymous GCS access)."""
    return xr.open_zarr(path, chunks=None, storage_options=dict(token="anon"))


def era5_grid_of(ds: xr.Dataset) -> spherical_harmonic.Grid:
    """The dinosaur Grid matching an ERA5(-like) lat/lon dataset."""
    return spherical_harmonic.Grid(
        latitude_nodes=ds.sizes["latitude"],
        longitude_nodes=ds.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude),
    )


def make_regridder(source_ds: xr.Dataset, model):
    """Conservative regridder source grid → model grid."""
    return horizontal_interpolation.ConservativeRegridder(
        era5_grid_of(source_ds), model.data_coords.horizontal, skipna=True
    )


def shift_forcings(ds: xr.Dataset, model) -> xr.Dataset:
    """Select the model's input+forcing variables and time-shift the forcings
    (SST / sea-ice) by 24 h, as the model expects."""
    return ds[model.input_variables + model.forcing_variables].pipe(
        xarray_utils.selective_temporal_shift,
        variables=model.forcing_variables,
        time_shift="24 hours",
    )


def build_ic_zarr(model, cfg: dict, window_days: int = 2, overwrite: bool = False) -> Path:
    """Slice the model's input+forcing variables at the config's init_date from
    ARCO-ERA5 and write the IC zarr that `load_ic_on_model_grid` regrids.
    No-op if the zarr already exists (unless overwrite=True)."""
    out = Path(cfg["paths"]["ic_zarr"])
    if out.exists() and not overwrite:
        print(f"IC already built: {out}")
        return out
    t0 = np.datetime64(cfg["run"]["init_date"])
    print(f"Opening ARCO-ERA5 and slicing the IC window at {t0} ...")
    full = open_arco_era5(cfg["paths"]["era5_arco"])
    window = full[model.input_variables + model.forcing_variables].sel(
        time=slice(t0, t0 + np.timedelta64(window_days, "D"))
    )
    window = window.pipe(
        xarray_utils.selective_temporal_shift,
        variables=model.forcing_variables,
        time_shift="24 hours",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Materialising {window.sizes['time']} snapshots -> {out} ...")
    data = window.compute()
    # Drop the encoding inherited from the ARCO store (zarr-v2 numcodecs
    # Blosc), which zarr-python 3 cannot write out in its default v3 format.
    if hasattr(data, "drop_encoding"):
        data = data.drop_encoding()
    else:  # older xarray
        for v in data.variables.values():
            v.encoding = {}
    data.to_zarr(str(out), mode="w")
    print("IC built.")
    return out


def load_ic_on_model_grid(model, ic_zarr: str | Path) -> xr.Dataset:
    """Open a previously-built IC zarr and conservatively regrid it onto the
    model grid (NaNs filled with nearest)."""
    sliced = xr.open_zarr(str(ic_zarr), chunks=None)
    regridder = make_regridder(sliced, model)
    return xarray_utils.fill_nan_with_nearest(xarray_utils.regrid(sliced, regridder))


def regrid_window(
    shifted_era5: xr.Dataset,
    regridder,
    t0: np.datetime64,
    t1: np.datetime64,
    inner_hours: int = 6,
    chunk_days: int = 2,
) -> xr.Dataset:
    """Subsample [t0, t1] to `inner_hours` cadence, then materialise (0.25°)
    and regrid to the model grid in blocks of `chunk_days` days. Peak RAM is
    one block at 0.25°; lower chunk_days if you hit OOM. (Used by the
    lead-time sweep, where the window spans weeks.)"""
    sub = (shifted_era5.sel(time=slice(t0, t1))
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
