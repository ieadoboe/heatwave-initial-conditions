"""One shareable table of every storyline event's dates and model set-up.

    python scripts/export_storyline_events.py

Reads each configs/*.yaml through heatwave_ic.config.load_config, the same
resolution the optimizer uses, so the table can never disagree with what was
run. The configs stay the only place dates are written down.

The model steps hourly from 00:00 on init_date for evol_days, and the loss is
the box mean of the LAST target_window_days of that run (heatwave_ic/optimize.py),
so the target window is the days from init_date + evol_days - target_window_days
up to the run's last day, both inclusive.

Output: data/storyline_events.csv
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HEATWAVE_ROOT", REPO))
DATA = ROOT / "data"
_DAY = np.timedelta64(1, "D")
BOX_CELLS = "4 x 4 cells of the 2.8 deg model grid"  # optimize.py box_halfwidth=2


def _config_module():
    """heatwave_ic.config without the package __init__, which imports jax."""
    if "heatwave_ic" not in sys.modules:
        pkg = types.ModuleType("heatwave_ic")
        pkg.__path__ = [str(REPO / "heatwave_ic")]
        sys.modules["heatwave_ic"] = pkg
    spec = importlib.util.spec_from_file_location(
        "heatwave_ic.config", REPO / "heatwave_ic" / "config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["heatwave_ic.config"] = mod
    spec.loader.exec_module(mod)
    return mod


def target_window(init_date, evol_days: float, window_days: float):
    """First and last calendar day of the window the loss is computed over."""
    run_end = init_date + np.timedelta64(int(round(evol_days * 24)), "h")
    start = run_end - np.timedelta64(int(round(window_days * 24)), "h")
    last = run_end - np.timedelta64(1, "h")
    return start.astype("datetime64[D]"), last.astype("datetime64[D]")


def row_for(cfg: dict) -> dict:
    event, run, loss, opt = cfg["event"], cfg["run"], cfg["loss"], cfg["optimizer"]
    start, last = target_window(run["init_date"], run["evol_days"],
                                loss["target_window_days"])
    day = lambda d: str(np.datetime64(d, "D"))
    return {
        "event": event["name"],
        "zone": event.get("zone", ""),
        "target_lat": event["target_lat"],
        "target_lon": event["target_lon"],
        "target_box": BOX_CELLS,
        "event_start": day(event["start"]),
        "event_peak": day(event["peak"]),
        "event_end": day(event["end"]),
        "init_date": day(run["init_date"]),
        "lead_days": run["lead_days"],
        "evol_days": run["evol_days"],
        "target_window_days": loss["target_window_days"],
        "target_window_start": day(start),
        "target_window_end": day(last),
        "window_covers_peak": bool(start <= event["peak"] <= last),
        "model": cfg["model_name"],
        "iterations": opt["iteration_number"],
        "learning_rate": float(opt["learning_rate"]),
        "lambda": loss["lambda"],
        "beta": loss["beta"],
    }


def main():
    C = _config_module()
    rows = [row_for(C.load_config(p))
            for p in sorted((REPO / "configs").glob("*.yaml"))]
    table = pd.DataFrame(rows)

    # Run status only where the summary row was made with the same start
    # date, so a stale result from an older set-up is never shown as current.
    summary = DATA / "atlas_summary.csv"
    if summary.exists():
        done = pd.read_csv(summary)[["event", "init_date", "status"]]
        table = table.merge(done, on=["event", "init_date"], how="left")
        table["status"] = table["status"].fillna("not run")
    else:
        table["status"] = "not run"

    out = DATA / "storyline_events.csv"
    table.to_csv(out, index=False)
    print(f"-> {out}\n")
    cols = ["event", "event_start", "event_peak", "event_end", "init_date",
            "lead_days", "evol_days", "target_window_start",
            "target_window_end", "window_covers_peak", "status"]
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
