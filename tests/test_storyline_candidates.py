"""Check the default storyline dates, the robustness label, and that a
generated config loads back with the same target window.

    python tests/test_storyline_candidates.py
"""
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC = load("storyline_candidates")
EX = load("export_storyline_events")


def test_storyline_setup():
    s = SC.storyline_setup("2010-07-29")      # Moscow's peak
    assert str(s["init_date"]) == "2010-07-23", s       # 6 days before
    assert str(s["target_window_start"]) == "2010-07-27", s
    assert str(s["target_window_end"]) == "2010-07-31", s
    # The run's last 5 days must be exactly the window.
    start, last = EX.target_window(np.datetime64(str(s["init_date"])),
                                   s["evol_days"], 5)
    assert (str(start), str(last)) == ("2010-07-27", "2010-07-31"), (start, last)


def test_robustness():
    cols = ["event_id", "rank", "start", "end"]
    ref = pd.DataFrame([(1, 1, "2026-06-12", "2026-07-03"),
                        (2, 2, "2026-06-30", "2026-07-24"),
                        (3, 3, "2026-08-19", "2026-09-12")], columns=cols)
    ref[["start", "end"]] = ref[["start", "end"]].apply(pd.to_datetime)
    fp = lambda d: pd.DataFrame([(c, e) for e, cs in d.items() for c in cs],
                                columns=["cell", "event_id"])
    ref_fp = fp({1: range(0, 50), 2: range(50, 90), 3: range(200, 240)})
    # Setting A: the same three events. Setting B: 1 and 2 joined (a real
    # merge, they share cells), 3 shifted by two days.
    a = (ref.copy(), ref_fp)
    b_run = pd.DataFrame([(9, 1, "2026-06-12", "2026-07-25"),
                          (8, 2, "2026-08-21", "2026-09-10")], columns=cols)
    b_run[["start", "end"]] = b_run[["start", "end"]].apply(pd.to_datetime)
    b = (b_run, fp({9: range(0, 90), 8: range(205, 240)}))
    rob = SC.robustness(ref, ref_fp, [a, b])
    assert list(rob.robustness) == ["depends on settings",
                                    "depends on settings", "holds"], rob
    assert list(rob.n_settings_merged) == [1, 1, 0], rob
    assert rob.max_date_shift_days[2] == 2, rob   # Aug 19 -> 21


def test_config_round_trip():
    row = pd.Series({
        "event_id": "2026_r01", "year": 2026, "rank": 1, "koppen": "Dfb",
        "start": "2026-06-12", "peak": "2026-06-27", "end": "2026-07-03",
        "target_lat": 44.25, "target_lon": -1.0, "robustness": "holds",
        **SC.storyline_setup("2026-06-27")})
    C = EX._config_module()
    with tempfile.TemporaryDirectory() as d:
        cfg = C.load_config(SC.make_config(row, Path(d)))
    exported = EX.row_for(cfg)
    assert exported["target_window_start"] == "2026-06-25", exported
    assert exported["target_window_end"] == "2026-06-29", exported
    assert exported["init_date"] == "2026-06-21", exported
    assert exported["window_covers_peak"], exported


def test_empty_workdir_says_where_it_looked():
    """No catalogues in the workdir: a message naming it, not a pandas error."""
    import sys
    with tempfile.TemporaryDirectory() as d:
        argv = sys.argv
        sys.argv = ["storyline_candidates.py", "--years", "2003", "--workdir", d]
        try:
            SC.main()
        except SystemExit as e:
            assert d in str(e.code), e.code
        else:
            raise AssertionError("expected SystemExit")
        finally:
            sys.argv = argv


if __name__ == "__main__":
    test_storyline_setup()
    test_robustness()
    test_config_round_trip()
    test_empty_workdir_says_where_it_looked()
    print("ok")
