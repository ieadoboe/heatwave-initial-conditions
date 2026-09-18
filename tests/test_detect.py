"""Plant known heat waves in a synthetic field and check detection finds them.

    python tests/test_detect.py

Needs numpy, pandas, xarray and scipy only.

Covers the four things that can quietly go wrong: the 3-day persistence rule,
the area filter, whether two separate blobs stay separate, and whether one
drifting blob stays a single event.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Import heatwave_ic.detect without executing the package __init__, which
# pulls in jax and neuralgcm. Detection needs neither.
REPO = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("heatwave_ic")
pkg.__path__ = [str(REPO / "heatwave_ic")]
sys.modules["heatwave_ic"] = pkg
_spec = importlib.util.spec_from_file_location(
    "heatwave_ic.detect", REPO / "heatwave_ic" / "detect.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["heatwave_ic.detect"] = D
_spec.loader.exec_module(D)

rng = np.random.default_rng(0)
LAT = np.arange(60, 40, -0.25)      # 80 cells, north to south like ERA5
LON = np.arange(-10, 20, 0.25)      # 120 cells
NDAY = 153                          # May-Sep

def season(days):
    """A smooth seasonal cycle peaking in late July."""
    return 20 + 8 * np.sin(np.pi * (days - 0) / NDAY)

def make_year(year, extra=None):
    days = pd.date_range(f"{year}-05-01", periods=NDAY, freq="D")
    base = season(np.arange(NDAY))[:, None, None]
    field = base + rng.normal(0, 2.0, (NDAY, LAT.size, LON.size))
    if extra is not None:
        field = field + extra
    return xr.DataArray(field.astype("float32"),
                        coords={"time": days, "latitude": LAT, "longitude": LON},
                        dims=("time", "latitude", "longitude"))

print("building 30-year baseline ...")
baseline = xr.concat([make_year(y) for y in range(1991, 2021)], dim="time")
print(f"  {baseline.sizes['time']} days")

thr = D.calendar_day_percentile(baseline, q=0.90)
print(f"threshold: {thr.sizes['dayofyear']} calendar days, "
      f"mean {float(thr.mean()):.1f} C")

# --- plant events -----------------------------------------------------------
extra = np.zeros((NDAY, LAT.size, LON.size), dtype="float32")
# A: big Iberian-scale blob, days 60-64 (5 days), lat rows 60:78, lon 0:40
extra[60:65, 60:78, 0:40] += 12.0
# B: separate northern blob, days 100-103 (4 days), far away in space
extra[100:104, 2:20, 80:118] += 12.0
# C: 2-day blip — must NOT survive the 3-day rule
extra[30:32, 30:50, 40:80] += 12.0
# D: 5 days but a single cell — must be dropped by the area filter
extra[80:85, 40:41, 60:61] += 12.0

target = make_year(2026, extra=extra)
det = D.detect(target, thr, land_mask=None, min_days=3,
               min_area_km2=50_000.0)
print(f"\ndetected {det.n_events} blobs before filtering")

ev = D.event_table(det, target, min_area_km2=50_000.0)
print(ev[["rank", "start", "peak", "end", "duration_days", "peak_area_km2",
          "centroid_lat", "centroid_lon", "max_anom_C"]].to_string(index=False))

# --- assertions -------------------------------------------------------------
assert len(ev) == 2, f"expected 2 events after the area filter, got {len(ev)}"

a = ev[ev.centroid_lat < 50].iloc[0]      # southern blob
b = ev[ev.centroid_lat > 50].iloc[0]      # northern blob
assert a.duration_days == 5, a.duration_days
assert b.duration_days == 4, b.duration_days
assert str(a.start) == "2026-06-30", a.start          # day 60 from May 1
assert str(b.start) == "2026-08-09", b.start          # day 100 from May 1
# centroids land inside the planted boxes
assert 42 < a.centroid_lat < 46, a.centroid_lat
assert 55 < b.centroid_lat < 60, b.centroid_lat
print("\nOK: 3-day rule dropped the 2-day blip, area filter dropped the "
      "single cell,\n    the two separated blobs stayed separate, dates and "
      "centroids correct.")

# --- a drifting blob must stay ONE event ------------------------------------
drift = np.zeros((NDAY, LAT.size, LON.size), dtype="float32")
for k, day in enumerate(range(50, 56)):
    drift[day, 40:60, 10 + 4 * k: 50 + 4 * k] += 12.0   # slides east daily
det2 = D.detect(make_year(2026, extra=drift), thr, min_days=3,
                min_area_km2=50_000.0)
ev2 = D.event_table(det2, make_year(2026, extra=drift), min_area_km2=50_000.0)
assert len(ev2) == 1, f"a drifting blob split into {len(ev2)} events"
assert ev2.iloc[0].duration_days == 6, ev2.iloc[0].duration_days
print("OK: a blob drifting 40 cells east over 6 days stayed one event.")

# --- area weighting ---------------------------------------------------------
area = D.cell_area_km2(np.array([0.0, 60.0]), np.arange(0, 1, 0.25))
ratio = area[1, 0] / area[0, 0]
assert 0.49 < ratio < 0.51, ratio
print(f"OK: a cell at 60N is {ratio:.2f} of the area of one at the equator.")
print("\nALL CHECKS PASSED")
