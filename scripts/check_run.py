"""Post-run sanity checks on optimization run directories.

    python scripts/check_run.py                     # every run under data/opt_runs
    python scripts/check_run.py data/opt_runs/pnw_* # named runs
    python scripts/check_run.py --expect-gain 3.7 --tol 0.5 data/opt_runs/pnw_*
    python scripts/check_run.py --summary data/atlas_summary_pnw_jun2021.csv

Checks, per run directory:

  1. Iteration 1 of reg.npy. reg[0] is 0 by construction (nothing has moved
     yet), so reg[1] is what the FIRST Adam step did. Adam's first update
     moves every parameter by exactly the learning rate whatever the gradient
     says, so too large a learning rate shows up here as a penalty that
     dwarfs the heat term, and the remaining iterations are spent undoing it.
     See the regularization reference scale section of CLAUDE.md.
  2. Whether the run ever got below its own starting loss. If it did not, its
     "gain" is not a minimum.
  3. Optionally, the gain against an expected value (--expect-gain), which is
     how the PNW run is validated against W&DL's +3.7 C. W&DL measure that
     against the hottest member of the 75-member stochastic ensemble, NOT
     against the single unperturbed forecast, so this needs ensemble.csv in
     the run directory (stage 3, scripts/run_ensembles.py) and refuses to
     judge without it. See heatwave_ic/ensemble.py.

Exits non-zero if any run fails a check, so a batch script stops on it.
Needs numpy only: no jax, so it runs on a login node.
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np


def rooted(rel: str) -> str:
    """HEATWAVE_ROOT + rel, as heatwave_ic.config.rooted does. Repeated here
    rather than imported: importing the package pulls in jax, and this script
    has to run on a login node."""
    root = os.environ.get("HEATWAVE_ROOT")
    if not root or os.path.isabs(rel) or "://" in rel:
        return rel
    return os.path.join(root, rel)


def dirs_from_summary(path: str):
    """The run_dir column of an atlas summary CSV. Falls back to the local
    data/opt_runs when the recorded absolute path is from another machine,
    which is what happens after pull.sh brings the runs home."""
    with open(rooted(path)) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        d = Path(r.get("run_dir") or "")
        if not d.name:
            continue
        if not d.exists():
            d = Path(rooted(os.path.join("data/opt_runs", d.name)))
        out.append(d)
    return out


def reg_term_in_loss_units(losses, box_T_K, reg):
    """The lambda * reg contribution to the loss, per iteration.

    lambda is not saved with the run, but the loss is
    beta*T_ref/sqrt(box_T) + lambda*reg and reg[0] is 0, so the heat term's
    constant is losses[0]*sqrt(box_T[0]) and the rest follows."""
    heat_const = losses[0] * np.sqrt(box_T_K[0])
    return losses - heat_const / np.sqrt(box_T_K)


def _peaks(run_dir: Path):
    """(optimized peak, unperturbed peak) in C from storyline.csv, or None."""
    path = run_dir / "storyline.csv"
    if not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    return (max(float(r["optimized_C"]) for r in rows),
            max(float(r["unperturbed_C"]) for r in rows))


def storyline_gain(run_dir: Path):
    """max(optimized) - max(unperturbed): the gain over the single
    unperturbed forecast. NOT the quantity W&DL report."""
    peaks = _peaks(run_dir)
    return None if peaks is None else peaks[0] - peaks[1]


def ensemble_gain(run_dir: Path):
    """max(optimized) - the hottest peak of any ensemble member, which is
    W&DL's quantity (heatwave_ic/ensemble.py: gain_vs_ens_max_C)."""
    peaks = _peaks(run_dir)
    path = run_dir / "ensemble.csv"
    if peaks is None or not path.exists():
        return None
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    members = [c for c in (rows[0] if rows else {}) if c.startswith("member_")]
    if not members:
        return None
    # A member column can hold blanks: an ensemble that was interrupted
    # mid-write leaves a ragged final row.
    peaks_by_member = [
        max(float(r[m]) for r in rows if r.get(m) not in (None, ""))
        for m in members
        if any(r.get(m) not in (None, "") for r in rows)
    ]
    if not peaks_by_member:
        return None
    return peaks[0] - max(peaks_by_member)


def check_run(run_dir: Path, max_reg_ratio: float,
              expect_gain: float | None, tol: float):
    """(list of failure strings, printable one-line summary)."""
    losses = np.load(run_dir / "losses.npy")
    box_T_K = np.load(run_dir / "box_T_K.npy")
    reg = np.load(run_dir / "reg.npy")
    fails = []

    if len(losses) < 2:
        return [f"{run_dir.name}: only {len(losses)} iteration(s) recorded"], ""
    if reg[0] != 0.0:
        fails.append(f"{run_dir.name}: reg[0] is {reg[0]:g}, expected 0 — the "
                     "run did not start from the unperturbed state")

    reg_term = reg_term_in_loss_units(losses, box_T_K, reg)
    ratio = reg_term[1] / losses[0]
    if ratio > max_reg_ratio:
        fails.append(
            f"{run_dir.name}: the first Adam step put {ratio:.3g}x the heat "
            f"term into the penalty (reg[1]={reg[1]:.4g}, limit {max_reg_ratio:g}). "
            "Lower the learning rate.")
    if losses.min() >= losses[0]:
        fails.append(
            f"{run_dir.name}: loss never beat its starting value "
            f"({losses[0]:.4g}); ended at {losses[-1]:.4g}. Not a minimum.")

    gain = storyline_gain(run_dir)
    ens_gain = ensemble_gain(run_dir)
    if expect_gain is not None:
        if ens_gain is None:
            fails.append(
                f"{run_dir.name}: no ensemble.csv, so the gain W&DL report "
                "cannot be computed. Run stage 3 (scripts/run_ensembles.py) "
                "for this run first.")
        elif abs(ens_gain - expect_gain) > tol:
            fails.append(
                f"{run_dir.name}: gain over the hottest ensemble member "
                f"{ens_gain:+.2f} C is not {expect_gain:+.2f} +/- {tol:.2f} C")

    line = (f"{run_dir.name:44} iters={len(losses):3d} "
            f"reg1/heat={ratio:9.3g} loss {losses[0]:.4g} -> {losses[-1]:.4g} "
            f"(min {losses.min():.4g})"
            + (f" vs control {gain:+.2f} C" if gain is not None else "")
            + (f" vs ens max {ens_gain:+.2f} C" if ens_gain is not None
               else " vs ens max n/a"))
    return fails, line


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="*",
                        help="Run directories (default: every one under "
                             "$HEATWAVE_ROOT/data/opt_runs)")
    parser.add_argument("--summary", default=None,
                        help="Take the run directories from the run_dir "
                             "column of an atlas summary CSV instead")
    parser.add_argument("--max-reg-ratio", type=float, default=1.0,
                        help="Fail when the penalty after one step exceeds "
                             "this multiple of the heat term (default 1.0)")
    parser.add_argument("--expect-gain", type=float, default=None,
                        help="Expected gain over the HOTTEST ENSEMBLE MEMBER "
                             "in C, e.g. 3.7 for the PNW reproduction. Needs "
                             "ensemble.csv in the run directory")
    parser.add_argument("--tol", type=float, default=0.5,
                        help="Tolerance on --expect-gain, in C (default 0.5)")
    args = parser.parse_args()

    dirs = [Path(d) for d in args.run_dirs]
    if args.summary:
        dirs += dirs_from_summary(args.summary)
    if not dirs:
        root = os.environ.get("HEATWAVE_ROOT", "")
        dirs = [Path(d) for d in sorted(glob.glob(os.path.join(root, "data/opt_runs/*")))]
    missing = [d for d in dirs if not (d / "reg.npy").exists()]
    dirs = [d for d in dirs if (d / "reg.npy").exists()]
    if not dirs:
        print("check_run: no run directories with reg.npy found"
              + (f" (looked in {missing[0]})" if missing else ""), file=sys.stderr)
        return 1

    all_fails = []
    for d in dirs:
        fails, line = check_run(d, args.max_reg_ratio, args.expect_gain, args.tol)
        if line:
            print(line)
        all_fails += fails

    if all_fails:
        sys.stdout.flush()      # keep the table above the failures in a log
        print("\nFAILED:", file=sys.stderr)
        for f in all_fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"\nOK: {len(dirs)} run(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
