"""Check the storyline target window against the case written out by hand.

    python tests/test_export_storyline_events.py
"""
import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_storyline_events", REPO / "scripts" / "export_storyline_events.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def test_target_window():
    # configs/pnw_jun2021.yaml: the 264 hourly steps run 21 Jun 00:00 to
    # 1 Jul 23:00, so the 5-day target window is 27 Jun to 1 Jul.
    start, last = E.target_window(np.datetime64("2021-06-21"), 11, 5)
    assert str(start) == "2021-06-27" and str(last) == "2021-07-01", (start, last)


def test_every_config_exports():
    C = E._config_module()
    for path in sorted((REPO / "configs").glob("*.yaml")):
        row = E.row_for(C.load_config(path))
        assert row["target_window_start"] <= row["target_window_end"], row
        assert row["target_window_end"] >= row["event_start"], row


if __name__ == "__main__":
    test_target_window()
    test_every_config_exports()
    print("ok")
