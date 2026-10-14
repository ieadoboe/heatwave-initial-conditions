"""Check the cross-run event matcher on hand-made tables.

    python tests/test_catalogue_sensitivity.py
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "catalogue_sensitivity", REPO / "scripts" / "catalogue_sensitivity.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def frame(rows):
    df = pd.DataFrame(rows, columns=["start", "end", "centroid_lat",
                                     "centroid_lon"])
    df[["start", "end"]] = df[["start", "end"]].apply(pd.to_datetime)
    return df


def test_match_events():
    ref = frame([("2026-06-12", "2026-07-03", 48.9, 13.6),   # central Europe
                 ("2026-05-14", "2026-05-22", 57.0, 37.6),   # western Russia
                 ("2026-08-19", "2026-09-05", 41.3, 17.1)])  # Italy
    run = frame([("2026-05-15", "2026-05-20", 56.5, 36.0),   # Russia, shorter
                 ("2026-06-10", "2026-06-20", 49.5, 12.0),   # first half of #1
                 ("2026-06-21", "2026-07-05", 48.0, 15.0),   # second half of #1
                 ("2026-08-20", "2026-09-01", 34.0, 5.0)])   # same dates, Algeria
    hit = S.match_events(ref, run).tolist()
    assert hit[0] == 2, hit        # longer overlap wins over nearer centroid
    assert hit[1] == 0, hit
    assert hit[2] == -1, hit       # right dates but 1,200 km away: no match
    assert S.tag_for(25_000, 0.95) == "_a25k_p95"
    print("ok")


if __name__ == "__main__":
    test_match_events()
