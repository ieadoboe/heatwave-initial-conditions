"""Plant a healthy run and an overshooting one, check check_run calls them.

    python tests/test_check_run.py

Needs numpy only. The point is the iteration-1 test: the penalty after the
first Adam step is derived from losses/box_T_K without knowing lambda, so the
derivation is what has to stay right.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_run", REPO / "scripts" / "check_run.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

BETA_T_REF = 10 * 293.15          # the heat term's constant
LAM = 100


def plant(tmp, name, reg, box_T_K, gain=None):
    """Write a run dir whose losses are consistent with reg and box_T."""
    d = Path(tmp) / name
    d.mkdir()
    losses = BETA_T_REF / np.sqrt(box_T_K) + LAM * reg
    np.save(d / "losses.npy", losses)
    np.save(d / "box_T_K.npy", box_T_K)
    np.save(d / "reg.npy", reg)
    if gain is not None:
        rows = [",unperturbed_C,optimized_C"]
        rows += [f"{i},20.0,{20.0 + gain}" for i in range(3)]
        (d / "storyline.csv").write_text("\n".join(rows) + "\n")
    return d


with tempfile.TemporaryDirectory() as tmp:
    n = 20
    # --- healthy: tiny penalty, loss falls ----------------------------------
    box = np.linspace(295.0, 299.0, n)                  # box warms
    reg = np.concatenate([[0.0], np.full(n - 1, 1e-4)])
    good = plant(tmp, "good", reg, box, gain=3.7)
    fails, line = C.check_run(good, 1.0, None, 0.5)
    assert not fails, fails
    print("OK: a run with a small penalty and a falling loss passes.")

    # the derived penalty must equal the real lambda*reg, which is the whole
    # basis of the check
    losses = np.load(good / "losses.npy")
    derived = C.reg_term_in_loss_units(losses, box, reg)
    assert np.allclose(derived, LAM * reg, atol=1e-9), derived[:3]
    print("OK: lambda*reg recovered from losses and box_T without knowing lambda.")

    # --- overshoot: reg explodes on step 1, loss never recovers -------------
    box_bad = np.full(n, 295.0)
    reg_bad = np.concatenate([[0.0], np.full(n - 1, 1e3)])
    bad = plant(tmp, "bad", reg_bad, box_bad)
    fails, _ = C.check_run(bad, 1.0, None, 0.5)
    assert any("first Adam step" in f for f in fails), fails
    assert any("never beat its starting value" in f for f in fails), fails
    print("OK: an iteration-1 blow-up is caught, and so is a loss that never improves.")

    # --- the gain check -----------------------------------------------------
    fails, _ = C.check_run(good, 1.0, 3.7, 0.5)
    assert not fails, fails
    fails, _ = C.check_run(good, 1.0, 9.5, 0.5)
    assert any("storyline gain" in f for f in fails), fails
    print("OK: the storyline gain is compared against the expected value.")

print("\nALL CHECKS PASSED")
