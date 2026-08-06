"""Translation-invariant shape statistic D_pp' (theory doc, section 9.2).

    D_pp' = min_{delta, A} int W(u) [ln P_p(u) - A - ln P_p'(u - delta)]^2 du,
    u = ln k

If D_pp' is at the numerical error floor, p is NOT observable through the
scalar two-point function: the integer ladder is pure translation and only the
reheating completion is testable. A residual above the floor is the p-shape
signal the 1/p expansion predicts at O(1/p).

The free shift delta absorbs the p-N_* degeneracy, so tables generated at
different N_* are directly comparable; a same-p pair from runs at different
N_* provides the null (error-floor) calibration.

Window: from 2 e-folds below each aligned notch to 7 above it -- the feature
through its recovery to the smooth region. The far-infrared stochastic rise is
excluded on the model's own interpretation of that regime.

Also reported: the rms after additionally marginalizing a linear tilt B*u.
That is a stricter null (a tilt difference is degenerate with n_s, which is
measured separately), so the pair (rms_A, rms_AB) brackets the honest answer.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WINDOW_BELOW = 2.0   # e-folds below the notch minimum
WINDOW_ABOVE = 7.0   # e-folds above it


def load(run):
    directory = ROOT / "results" / run
    table = np.genfromtxt(directory / "primordial_power.csv",
                          delimiter=",", names=True)
    meta = json.loads((directory / "summary.json").read_text())
    ln_k = np.log(table["k_Mpc"])
    ln_p = np.log(table["P_R"])
    ln_notch = np.log(meta["notch"]["notch_k_Mpc"])
    return CubicSpline(ln_k, ln_p), ln_k, ln_notch, meta


def residual_rms(spline_a, spline_b, window, delta, with_tilt):
    """rms of ln P_a(u) - ln P_b(u - delta) after removing A (and B u)."""
    u = window
    difference = spline_a(u) - spline_b(u - delta)
    design = [np.ones_like(u)] + ([u - u.mean()] if with_tilt else [])
    basis = np.vstack(design).T
    coefficients, *_ = np.linalg.lstsq(basis, difference, rcond=None)
    return float(np.sqrt(np.mean((difference - basis @ coefficients) ** 2)))


def compare(run_a, run_b, samples=2000):
    spline_a, ln_k_a, notch_a, _ = load(run_a)
    spline_b, ln_k_b, notch_b, _ = load(run_b)
    delta_guess = notch_a - notch_b
    window = np.linspace(notch_a - WINDOW_BELOW, notch_a + WINDOW_ABOVE, samples)

    # Both splines must cover the window (b's, after shifting).
    lo, hi = ln_k_b.min(), ln_k_b.max()

    def objective(delta, with_tilt=False):
        if window[0] - delta < lo or window[-1] - delta > hi:
            return 1.0e6
        return residual_rms(spline_a, spline_b, window, delta, with_tilt)

    best = minimize_scalar(objective, bracket=(delta_guess - 0.3, delta_guess,
                                               delta_guess + 0.3),
                           method="brent", options={"xtol": 1e-10})
    rms_shift_amp = float(best.fun)
    rms_with_tilt = float(
        minimize_scalar(lambda d: objective(d, True),
                        bracket=(best.x - 0.1, best.x, best.x + 0.1),
                        method="brent", options={"xtol": 1e-10}).fun
    )
    return best.x, rms_shift_amp, rms_with_tilt


def main():
    groups = {
        "same p, different N_* (NULL / error floor)": [
            ("reheat_p63", "resolved_p63"),
            ("reheat_p64", "resolved_p64"),
        ],
        "adjacent p, coordinate wall": [
            ("resolved_p62", "resolved_p63"),
            ("resolved_p63", "resolved_p64"),
            ("resolved_p64", "resolved_p65"),
        ],
        "adjacent p, invariant wall": [
            ("invwall_p65", "invwall_p66"),
            ("invwall_p66", "invwall_p67"),
        ],
        "delta p = 2": [
            ("resolved_p62", "resolved_p64"),
            ("resolved_p63", "resolved_p65"),
        ],
    }
    print(f"window: notch-{WINDOW_BELOW} to notch+{WINDOW_ABOVE} e-folds\n")
    print(f"{'pair':>28} {'delta*':>9} {'rms(A)':>10} {'rms(A+tilt)':>12}")
    for label, pairs in groups.items():
        print(f"--- {label} ---")
        for run_a, run_b in pairs:
            try:
                delta, rms_a, rms_ab = compare(run_a, run_b)
            except FileNotFoundError:
                print(f"{run_a+' vs '+run_b:>28}   (missing -- skipped)")
                continue
            print(f"{run_a+' vs '+run_b:>28} {delta:>9.4f} {rms_a:>10.2e} "
                  f"{rms_ab:>12.2e}")
    print("\nread: rms is the fractional ln-P shape difference after removing "
          "translation and amplitude (and tilt). Compare adjacent-p rows "
          "against the null rows; only the excess is p-shape information.")


if __name__ == "__main__":
    main()
