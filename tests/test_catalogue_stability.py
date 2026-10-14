"""Check the leave-one-out baseline, the threshold cache names, and the
all-summers summary helpers.

    python tests/test_catalogue_stability.py
"""
import importlib.util
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DH = load("detect_heatwaves_europe")
AY = load("catalogue_stability_allyears")


def test_leave_one_out():
    inside = DH.baseline_years_for(2003, 1991, 2020, leave_one_out=True)
    assert 2003 not in inside and len(inside) == 29, inside
    outside = DH.baseline_years_for(2026, 1991, 2020, leave_one_out=True)
    assert len(outside) == 30, outside
    shared = DH.baseline_years_for(2003, 1991, 2020, leave_one_out=False)
    assert 2003 in shared and len(shared) == 30


def test_threshold_name():
    full = list(range(1991, 2021))
    # The default keeps its original name, so old caches still match.
    assert DH.threshold_name(1991, 2020, 0.90, "m04-10", 15, full) == \
        "heatwave_threshold_europe_1991_2020_p90_m04-10.nc"
    loo = DH.baseline_years_for(2003, 1991, 2020, True)
    name = DH.threshold_name(1991, 2020, 0.95, "m04-10", 31, loo)
    assert name == "heatwave_threshold_europe_1991_2020_p95_w31_loo2003_m04-10.nc"
    # Two summers must never share a leave-one-out cache.
    other = DH.baseline_years_for(2004, 1991, 2020, True)
    assert name != DH.threshold_name(1991, 2020, 0.95, "m04-10", 31, other)


def test_names_never_collide():
    full = list(range(1991, 2021))
    # Truncating 0.57 * 100 gave 56, colliding with 0.56.
    a = DH.threshold_name(1991, 2020, 0.57, "m04-10", 15, full)
    b = DH.threshold_name(1991, 2020, 0.56, "m04-10", 15, full)
    assert a != b and "_p57_" in a, (a, b)
    assert "_p97.5_" in DH.threshold_name(1991, 2020, 0.975, "m04-10", 15, full)
    # A standalone run must not overwrite the reference catalogue.
    loo = DH.baseline_years_for(2003, 1991, 2020, True)
    assert DH.default_tag(15, 1991, 2020, full) == ""
    assert DH.default_tag(31, 1991, 2020, loo) == "_w31_loo"


def test_changed_parameter():
    assert AY.changed_parameter(25_000, 0.90, 15) == "area"
    assert AY.changed_parameter(50_000, 0.95, 15) == "percentile"
    assert AY.changed_parameter(50_000, 0.90, 7) == "window"
    assert AY.changed_parameter(25_000, 0.95, 15) is None


def test_crowding():
    top = pd.DataFrame({
        "start": pd.to_datetime(["2026-06-12", "2026-06-30", "2026-08-19"]),
        "end": pd.to_datetime(["2026-07-03", "2026-07-24", "2026-09-12"]),
        "centroid_lat": [48.9, 46.8, 41.3],
        "centroid_lon": [13.6, 4.8, 17.1]})
    # June and July overlap by four days, about 6 degrees apart: one pair.
    # August starts 26 days after July ends: not close in time.
    assert AY.crowding(top) == 1
    assert AY.crowding(top, max_dist_deg=5.0) == 0


def fp(cells_by_event):
    """Footprint table from {event_id: [flat cell indices]}."""
    rows = [(c, e) for e, cells in cells_by_event.items() for c in cells]
    return pd.DataFrame(rows, columns=["cell", "event_id"])


def events(ids):
    return pd.DataFrame({"event_id": ids, "rank": range(1, len(ids) + 1)})


def test_match_by_overlap():
    # Reference: big event 1 on cells 0-99, small neighbour 2 on 100-109.
    ref, ref_fp = events([1, 2]), fp({1: range(100), 2: range(100, 110)})

    # Stricter setting: event 1 shrinks, event 2 vanishes. Dates and centres
    # would have matched 2 to 1's counterpart and called it a merge.
    m = AY.match_by_overlap(ref, ref_fp, events([7]), fp({7: range(20, 60)}))
    assert list(m.run_index) == [0, -1] and not m.merged.any(), m

    # Looser setting: one run event covers both, a real merge.
    m = AY.match_by_overlap(ref, ref_fp, events([3]), fp({3: range(0, 110)}))
    assert list(m.run_index) == [0, 0] and m.merged.all(), m

    # Event 1 splits in two; the larger piece is its match.
    m = AY.match_by_overlap(ref, ref_fp, events([4, 5]),
                            fp({4: range(0, 30), 5: range(30, 100)}))
    assert m.run_index[0] == 1 and m.n_pieces[0] == 2, m

    # A setting that found nothing.
    empty = pd.DataFrame(columns=["rank", "start", "peak", "end"])
    m = AY.match_by_overlap(ref, ref_fp, empty, fp({}))
    assert (m.run_index == -1).all(), m


def test_empty_setting():
    """compare_year end to end, with one setting that found nothing."""
    import tempfile
    import numpy as np
    ref = pd.DataFrame({
        "rank": [1, 2], "event_id": [1, 2],
        "start": ["2003-07-29", "2003-04-29"], "peak": ["2003-08-10", "2003-05-08"],
        "end": ["2003-08-15", "2003-05-16"],
        "centroid_lat": [46.0, 45.0], "centroid_lon": [4.0, 19.0]})
    shape = np.array([10, 5, 5])

    def save(path, ids, cells):
        t, i, j = np.unravel_index(np.asarray(cells, dtype=int), tuple(shape))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=t, i=i, j=j,
                            event_id=np.asarray(ids, dtype=np.int32), shape=shape)

    with tempfile.TemporaryDirectory() as d:
        AY.OUT = Path(d)
        for area, pct, window in AY.product(AY.AREAS_KM2, AY.PERCENTILES,
                                            AY.WINDOWS):
            csv = AY.OUT / f"heatwave_events_europe_2003{AY.tag_for(area, pct, window)}.csv"
            if (area, pct, window) == (100_000, 0.95, 7):
                pd.DataFrame(columns=["rank", "start", "peak", "end"]).to_csv(
                    csv, index=False)
                save(AY.labels_for(csv), [], [])
            else:
                ref.to_csv(csv, index=False)
                save(AY.labels_for(csv), [1] * 5 + [2] * 3, list(range(8)))
        table, summary = AY.compare_year(2003, top=6)
    empty = table[(table.min_area_km2 == 100_000) & (table.percentile == 0.95)
                  & (table.window_days == 7)]
    assert empty.moved.all() and empty["rank"].isna().all(), empty
    assert summary["n_ref"] == 2 and summary["n_moved_any"] == 2, summary
    assert summary["n_moved_area"] == 0, summary   # only the "several" bucket
    assert summary["n_top_merged"] == 0, summary


if __name__ == "__main__":
    test_leave_one_out()
    test_threshold_name()
    test_names_never_collide()
    test_match_by_overlap()
    test_empty_setting()
    test_changed_parameter()
    test_crowding()
    print("ok")
