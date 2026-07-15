"""Köppen-Geiger climate-zone classification for events and grid points.

Data: data/koppen_geiger_0p5_1991_2020.npz — the 0.5°, 1991-2020 historical
map from Beck et al. (2023), repackaged from koppen_geiger_tif.zip
(figshare article 21789074) into a small npz (uint8 class grid + lat/lon
axes + code legend; 0 = ocean/no data).

Cite in any publication:
    Beck, H. E., T. R. McVicar, N. Vergopolan, A. Berg, N. J. Lutsko,
    A. Dufour, Z. Zeng, X. Jiang, A. I. J. M. van Dijk, and D. G. Miralles
    (2023). High-resolution (1 km) Köppen-Geiger maps for 1901-2099 based on
    constrained CMIP6 projections. Scientific Data 10, 724.

CAVEAT for coarse-model work: a 0.5° (let alone 2.8°) cell class can differ
from the station's classic classification — e.g. Lytton, BC reads Dsc here
because the cell averages in high terrain around the hot valley. Report both
the point class and the modal class over the model's target box
(`classify_event`), and say which one the zone assignment uses.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

KOPPEN_NPZ = (Path(__file__).resolve().parents[1]
              / "data" / "koppen_geiger_0p5_1991_2020.npz")

#: Köppen main groups.
GROUP_NAMES = {
    "A": "tropical",
    "B": "arid",
    "C": "temperate",
    "D": "cold (continental)",
    "E": "polar",
}


@lru_cache(maxsize=1)
def _load():
    with np.load(KOPPEN_NPZ) as z:
        return z["classes"], z["lat"], z["lon"], z["codes"]


def _wrap_lon(lon: float) -> float:
    """Signed degrees East in [-180, 180)."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def koppen_class(lat: float, lon: float) -> str:
    """Köppen-Geiger code ('Dfb', ...) of the 0.5° cell nearest (lat, lon);
    'ocean' if the cell has no land. lon in signed degrees East (any wrap)."""
    classes, lats, lons, codes = _load()
    i = int(np.abs(lats - float(lat)).argmin())
    j = int(np.abs(lons - _wrap_lon(lon)).argmin())
    return str(codes[classes[i, j]])


def koppen_class_modal(lat: float, lon: float, box_degrees: float = 2.8) -> str:
    """Most common LAND class within a box_degrees square centred on
    (lat, lon) — closer to what a coarse model grid cell 'is' than the single
    0.5° point value. 'ocean' if the box contains no land."""
    classes, lats, lons, codes = _load()
    half = box_degrees / 2.0
    lon = _wrap_lon(lon)
    ii = np.where(np.abs(lats - float(lat)) <= half)[0]
    dlon = np.abs(lons - lon)
    jj = np.where(np.minimum(dlon, 360.0 - dlon) <= half)[0]
    vals = classes[np.ix_(ii, jj)].ravel()
    vals = vals[vals > 0]
    if vals.size == 0:
        return "ocean"
    return str(codes[np.bincount(vals).argmax()])


def koppen_group(code: str) -> str:
    """Main group letter ('A'..'E') of a Köppen code; 'ocean' passes through."""
    return code if code == "ocean" else code[0]


def classify_event(cfg: dict, box_degrees: float = 2.8) -> dict:
    """Köppen classification of an event config's target location: the point
    class, the modal class over one model grid cell (box_degrees), and the
    modal class over the 4x4-cell loss box (4 * box_degrees)."""
    ev = cfg["event"]
    lat, lon = ev["target_lat"], ev["target_lon"]
    point = koppen_class(lat, lon)
    cell = koppen_class_modal(lat, lon, box_degrees)
    loss_box = koppen_class_modal(lat, lon, 4 * box_degrees)
    return {
        "event": ev["name"],
        "koppen_point": point,
        "koppen_model_cell": cell,
        "koppen_loss_box": loss_box,
        "group": koppen_group(cell if cell != "ocean" else point),
        "group_name": GROUP_NAMES.get(
            koppen_group(cell if cell != "ocean" else point), "ocean"),
    }
