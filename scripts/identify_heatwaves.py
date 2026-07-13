"""
Heatwave Detection for St. John's, NL
Based on Perkins & Alexander (2013) definitions:
  - CTX90pct: 3+ consecutive days above calendar-day 90th percentile of Tmax
  - CTN90pct: 3+ consecutive nights above calendar-day 90th percentile of Tmin
  - EHF:      3+ consecutive days of positive Excess Heat Factor

Inputs (in data/):
  Tmax: era5_tmax_stjohns.nc      single combined 1985–2024 file. Carries a
        spurious per-year `time` axis on top of valid_time (block-diagonal,
        one finite value per day); load_available collapses it to a daily
        series on load.
  Tmin: era5_tmin_stjohns.nc      single combined file from
        scripts/download_era5_tmax_tmin.py, extended by extend_era5_daily.py
        (var: t2m, time coord: valid_time).
  If the Tmin file is absent, only CTX90pct is computed and CTN90pct/EHF are
  skipped with a warning — re-run once the Tmin archive is present.

Outputs:
  - heatwave_events.csv      catalogue of all detected events
  - heatwave_summary.csv     yearly aspect summary (HWN, HWD, HWF, HWA, HWM)
  - heatwave_timeseries.pdf  time series of yearly heatwave frequency
"""

import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Project plotting convention (CLAUDE.md).
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from common import mpl_apply  # noqa: E402

mpl_apply()

# ─────────────────────────────────────────────
# USER SETTINGS — edit these
# ─────────────────────────────────────────────

TMAX_GLOB = "data/era5_tmax_stjohns.nc"   # single combined archive (1985–)
TMIN_GLOB = "data/era5_tmin_stjohns.nc"   # single combined archive (1985–)

T2M_VAR = "t2m"          # ERA5-Land 2m_temperature variable
TIME_COORD = "valid_time"

# Bounding box around St. John's, NL
LAT_MIN, LAT_MAX = 47.2, 47.8
LON_MIN, LON_MAX = -53.5, -52.5

# Percentile window half-width (±days around each calendar day)
WINDOW = 7   # 15-day window total, as in Perkins & Alexander

# Minimum consecutive days to qualify as a heatwave
MIN_DURATION = 3

OUTPUT_DIR = Path(".")

# ─────────────────────────────────────────────
# Load whatever years are available
# ─────────────────────────────────────────────


def load_available(glob_pattern: str, lat_min: float, lat_max: float,
                   lon_min: float, lon_max: float) -> pd.Series | None:
    """
    Open every file matching glob_pattern, take a spatial mean over the
    bounding box, convert K→°C, and return a daily pandas Series.

    Handles two on-disk layouts transparently:
      - per-year files (era5_tmin_<YYYY>_tmp.nc), combined on valid_time; and
      - a single combined file (era5_tmax_stjohns.nc) that carries a spurious
        per-year `time` axis on top of valid_time (block-diagonal: exactly one
        finite value per day). Any axis other than valid_time is collapsed with
        a skipna reduction, which recovers the true daily series.

    Returns None if no files match.
    """
    files = sorted(glob(glob_pattern))
    if not files:
        return None

    ds = xr.open_mfdataset(files, combine="by_coords")

    # ERA5 latitude is descending; slice high→low.
    lat_slice = (slice(lat_max, lat_min)
                 if ds.latitude[0] > ds.latitude[-1]
                 else slice(lat_min, lat_max))

    da = ds[T2M_VAR].sel(latitude=lat_slice,
                         longitude=slice(lon_min, lon_max))
    da = da.mean(dim=["latitude", "longitude"])

    # Collapse any axis other than valid_time (e.g. the combined file's
    # spurious per-year `time` axis). One value is finite per day, so a
    # skipna mean recovers it.
    extra = [d for d in da.dims if d != TIME_COORD]
    if extra:
        da = da.mean(dim=extra, skipna=True)

    # Convert K → °C if needed.
    vals = da.values
    if np.nanmean(vals) > 200:
        da = da - 273.15
        da.attrs["units"] = "°C"

    s = da.to_pandas()
    s.index.name = "time"
    s = s.sort_index()

    years = sorted(s.index.year.unique())
    print(f"  {glob_pattern}: {len(files)} file(s), "
          f"years {years[0]}–{years[-1]}")
    return s


print("Loading ERA5 Tmax / Tmin (whatever's available)…")
tmax = load_available(TMAX_GLOB, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
tmin = load_available(TMIN_GLOB, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

if tmax is None:
    raise SystemExit(f"No Tmax files matched {TMAX_GLOB!r}. Nothing to do.")

YEAR_START = int(tmax.index.year.min())
YEAR_END = int(tmax.index.year.max())
N_YEARS = YEAR_END - YEAR_START + 1
print(f"  Tmax: {len(tmax)} days "
      f"({tmax.index[0].date()} – {tmax.index[-1].date()}, {N_YEARS} yr)")

have_tmin = tmin is not None
if have_tmin:
    print(f"  Tmin: {len(tmin)} days "
          f"({tmin.index[0].date()} – {tmin.index[-1].date()})")
    # Align Tmax/Tmin to their common date range.
    common = tmax.index.intersection(tmin.index)
    tmax_eh, tmin_eh = tmax.loc[common], tmin.loc[common]
    tmean = (tmax_eh + tmin_eh) / 2.0
else:
    print("  Tmin: not on disk yet — skipping CTN90pct and EHF.")
    tmean = None

if N_YEARS < 30:
    print(f"\n  NOTE: only {N_YEARS} yr of Tmax — calendar-day"
          "90th-percentile  baseline is short; treat results"
          "as a preliminary sanity check, not the final climatology.")

# Identify Consecutive Runs


def find_heatwave_events(condition_series, min_duration=3):
    """
    Given a boolean Series, return a list of (start_date, end_date, indices)
    for all runs of True >= min_duration days.
    """
    events = []
    arr = condition_series.values
    dates = condition_series.index
    i = 0
    while i < len(arr):
        if arr[i]:
            j = i
            while j < len(arr) and arr[j]:
                j += 1
            duration = j - i
            if duration >= min_duration:
                events.append((dates[i], dates[j - 1], list(range(i, j))))
            i = j
        else:
            i += 1
    return events


# CTX90pct - Tmax above calendar-day 90th percentile


def compute_calendar_percentile(series, percentile=90, window=7):
    """
    For each calendar day (1–366), compute the given percentile
    using all values within ±window days across all years.
    Returns a dict keyed by day-of-year (1–366).
    """
    doy_pctile = {}
    doys = series.index.dayofyear
    values = series.values

    for doy in range(1, 367):
        window_doys = set()
        for d in range(doy - window, doy + window + 1):
            wd = ((d - 1) % 365) + 1
            window_doys.add(wd)

        mask = np.isin(doys, list(window_doys))
        pool = values[mask]
        if len(pool) > 0:
            doy_pctile[doy] = np.percentile(pool, percentile)
        else:
            doy_pctile[doy] = np.nan

    return doy_pctile


print("\nComputing CTX90pct thresholds…")
ctx_thresholds = compute_calendar_percentile(
    tmax, percentile=90, window=WINDOW)
ctx_exceed = pd.Series(
    [tmax.iloc[i] > ctx_thresholds.get(tmax.index[i].dayofyear, np.nan)
     for i in range(len(tmax))],
    index=tmax.index,
    dtype=bool,
)
ctx_events = find_heatwave_events(ctx_exceed, MIN_DURATION)
print(f"  CTX90pct: {len(ctx_events)} heatwave events detected")

# CTN90pct — Tmin above calendar-day 90th percentile (needs Tmin)

if have_tmin:
    print("Computing CTN90pct thresholds…")
    ctn_thresholds = compute_calendar_percentile(
        tmin, percentile=90, window=WINDOW)
    ctn_exceed = pd.Series(
        [tmin.iloc[i] > ctn_thresholds.get(tmin.index[i].dayofyear, np.nan)
         for i in range(len(tmin))],
        index=tmin.index,
        dtype=bool,
    )
    ctn_events = find_heatwave_events(ctn_exceed, MIN_DURATION)
    print(f"  CTN90pct: {len(ctn_events)} heatwave events detected")
else:
    ctn_events = []

# EHF — Excess Heat Factor (needs both Tmax and Tmin)

#   EHI(accl.) = mean(T_i, T_{i-1}, T_{i-2}) - mean(T_{i-3}…T_{i-32})
#   EHI(sig.)  = mean(T_i, T_{i-1}, T_{i-2}) - T95  (climatological
#                                                   95th pctile)
#   EHF        = max(1, EHI_accl) × EHI_sig
#   Heatwave day: EHF > 0 for >= 3 consecutive days

if have_tmin:
    print("Computing EHF…")
    T95 = np.percentile(tmean.values, 95)
    print(f"  T95 (climatological 95th pctile of Tmean) = {T95:.2f}°C")

    tmean_arr = tmean.values
    n = len(tmean_arr)
    ehf = np.full(n, np.nan)

    for i in range(32, n):
        t3 = tmean_arr[i - 2:i + 1].mean()
        t30 = tmean_arr[i - 32:i - 2].mean()
        ehi_accl = t3 - t30
        ehi_sig = t3 - T95
        ehf[i] = max(1.0, ehi_accl) * ehi_sig

    ehf_series = pd.Series(ehf, index=tmean.index)
    ehf_positive = ehf_series > 0
    ehf_events = find_heatwave_events(ehf_positive, MIN_DURATION)
    print(f"  EHF: {len(ehf_events)} heatwave events detected")
else:
    ehf_series = None
    ehf_events = []

# Build Event Catalogue


def build_catalogue(events, magnitude_series, index_name):
    """Build a DataFrame of events with duration and magnitude."""
    rows = []
    for start, end, idx in events:
        vals = magnitude_series.iloc[idx].values
        rows.append({
            "index":    index_name,
            "start":    start.date(),
            "end":      end.date(),
            "duration": len(idx),
            "peak":     round(float(np.max(vals)), 2),
            "mean_mag": round(float(np.mean(vals)), 2),
            "year":     start.year,
        })
    return pd.DataFrame(rows)


cats = [build_catalogue(ctx_events, tmax, "CTX90pct")]
if have_tmin:
    cats.append(build_catalogue(ctn_events, tmin, "CTN90pct"))
    cats.append(build_catalogue(ehf_events, ehf_series, "EHF"))

catalogue = pd.concat(cats, ignore_index=True)
catalogue.to_csv(OUTPUT_DIR / "heatwave_events.csv", index=False)
print(f"\nEvent catalogue saved: heatwave_events.csv "
      f"({len(catalogue)} total events)")

# Yearly Aspects (HWN, HWD, HWF, HWA, HWM)


def yearly_aspects(catalogue_df, index_name, year_start, year_end):
    """
    Compute Perkins & Alexander yearly aspects for one index:
      HWN = yearly number of heatwaves
      HWD = length of longest yearly event (days)
      HWF = sum of participating heatwave days per year
      HWA = peak magnitude of hottest yearly event
      HWM = average magnitude across all yearly events
    """
    years = range(year_start, year_end + 1)
    sub = catalogue_df[catalogue_df["index"] == index_name]
    rows = []
    for yr in years:
        yr_data = sub[sub["year"] == yr]
        hwn = len(yr_data)
        hwd = int(yr_data["duration"].max()) if hwn > 0 else 0
        hwf = int(yr_data["duration"].sum()) if hwn > 0 else 0
        hwa = round(float(yr_data["peak"].max()), 2) if hwn > 0 else np.nan
        hwm = (round(float(yr_data["mean_mag"].mean()), 2)
               if hwn > 0 else np.nan)
        rows.append({"year": yr, "index": index_name,
                     "HWN": hwn, "HWD": hwd, "HWF": hwf,
                     "HWA": hwa, "HWM": hwm})
    return pd.DataFrame(rows)


summary_parts = [yearly_aspects(catalogue, "CTX90pct", YEAR_START, YEAR_END)]
if have_tmin:
    summary_parts.append(yearly_aspects(catalogue, "CTN90pct",
                                        YEAR_START, YEAR_END))
    summary_parts.append(yearly_aspects(catalogue, "EHF",
                                        YEAR_START, YEAR_END))

summary = pd.concat(summary_parts, ignore_index=True)
summary.to_csv(OUTPUT_DIR / "heatwave_summary.csv", index=False)
print("Yearly aspect summary saved: heatwave_summary.csv")

# Plot - Yearly HWF (heatwave days per year)
indices_to_plot = ["CTX90pct"]
if have_tmin:
    indices_to_plot += ["CTN90pct", "EHF"]

colors = {"CTX90pct": "#d62728", "CTN90pct": "#1f77b4", "EHF": "#ff7f0e"}
labels = {
    "CTX90pct": "CTX90pct (Tmax days above 90th pctile)",
    "CTN90pct": "CTN90pct (Tmin nights above 90th pctile)",
    "EHF":      "EHF (Excess Heat Factor)",
}

fig, axes = plt.subplots(len(indices_to_plot), 1,
                         figsize=(12, 3 * len(indices_to_plot)),
                         sharex=True, squeeze=False)
axes = axes[:, 0]

for ax, idx_name in zip(axes, indices_to_plot):
    sub = summary[summary["index"] == idx_name]
    ax.bar(sub["year"], sub["HWF"],
           color=colors[idx_name], alpha=0.75, label="HWF")
    # 5-yr rolling mean only when there's enough data.
    if N_YEARS >= 5:
        ax.plot(sub["year"], sub["HWF"].rolling(5, center=True).mean(),
                color="black", linewidth=1.5, label="5-yr rolling mean")
    ax.set_ylabel("Heatwave days/yr")
    ax.set_title(labels[idx_name])
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

axes[-1].set_xlabel("Year")
suptitle = f"Heatwave Frequency — St. John's NL ({YEAR_START}–{YEAR_END})"
if N_YEARS < 30:
    suptitle += f"  [PRELIMINARY: {N_YEARS}-yr baseline]"
fig.suptitle(suptitle, fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatwave_timeseries.pdf", bbox_inches="tight")
plt.close()
print("Plot saved: heatwave_timeseries.pdf")


print("\n── Summary ─────────────────────────────────────")
for idx_name in indices_to_plot:
    sub = catalogue[catalogue["index"] == idx_name]
    print(f"\n{idx_name}:")
    print(f"  Total events    : {len(sub)}")
    if len(sub) > 0:
        print(f"  Avg duration    : {sub['duration'].mean():.1f} days")
        top = sub.nlargest(3, "peak")[["start", "end", "duration", "peak"]]
        print("  Top 3 events by peak magnitude:")
        print(top.to_string(index=False))

if not have_tmin:
    print("\nCTN90pct and EHF skipped — re-run once "
          "era5_tmin_*_tmp.nc files appear at the repo root.")
print("\nDone.")
