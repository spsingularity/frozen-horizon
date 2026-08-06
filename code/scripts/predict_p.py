"""Given N_*, determine which integer p the model selects -- and what it predicts.

Once reheating fixes N_*, nothing is free: the feature's physical wavenumber is
determined for each p, and therefore so is the quadrupole ratio. This script
turns an N_* into a prediction and reports how sharply p is separated.

    python3 scripts/predict_p.py --n-star 54.0

Uses the dense p = 66/67/68 runs. N_* enters as a pure translation of the
transfer curve in ln k (a mode's crossing e-fold is measured from the end of
inflation, so moving the pivot slides the whole curve), which is exact for the
feature location. The residual effect on the smooth reference is second order
and is reported as a caveat, not modelled.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frozen_horizon import config, projection  # noqa: E402

REFERENCE_N_STAR = 55.0


def load_dense(p):
    directory = ROOT / "results" / f"dense_p{p}"
    summary = json.loads((directory / "summary.json").read_text())
    table = np.genfromtxt(
        directory / "primordial_transfer.csv", delimiter=",", names=True
    )
    return summary, table


def feature_efold(summary, n_star_used=REFERENCE_N_STAR):
    """E-folds from the feature's horizon crossing to the end of inflation."""
    return n_star_used - np.log(summary["notch"]["notch_k_over_kpivot"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-star", type=float, required=True)
    parser.add_argument("--sigma", type=float, default=0.5,
                        help="uncertainty on N_* in e-folds")
    parser.add_argument("--p-values", type=int, nargs="+", default=[66, 67, 68])
    args = parser.parse_args()

    print(f"N_* = {args.n_star:.4f} +/- {args.sigma:.4f}  "
          f"(pivot k_* = {config.K_PIVOT} Mpc^-1)\n")

    print(f"{'p':>4} {'N_feature':>10} {'k_notch [Mpc^-1]':>18} {'depth':>8} "
          f"{'C_2/C_2^sm':>12} {'|dN_*| to fit':>14}")

    rows = []
    for p in args.p_values:
        summary, table = load_dense(p)
        n_feature = feature_efold(summary)

        # Slide the curve to the requested N_*.
        shift = args.n_star - REFERENCE_N_STAR
        k_ratio = table["k_over_kpivot"] * np.exp(shift)
        transfer = table["scalar_transfer"]

        k_notch = summary["notch"]["notch_k_over_kpivot"] * np.exp(shift)
        depth = summary["notch"]["notch_depth"]
        obs = summary["observables"]
        c2 = projection.projected_ratio(
            2, k_ratio, transfer, obs["n_s"], obs["alpha_s"]
        )

        # Everything depends on p and N_* only through the combination
        # u = N_* - 0.9933 (p - 67); the integer p just samples that axis.
        u = args.n_star - 0.9933 * (p - 67)
        rows.append((p, n_feature, k_notch * config.K_PIVOT, depth, c2, u))
        print(f"{p:>4} {n_feature:>10.4f} {k_notch*config.K_PIVOT:>18.4e} "
              f"{depth:>8.4f} {c2:>12.4f} {u:>14.4f}")

    # There is no "selection" without data. What the model does is predict a
    # definite C_2 per p; the comparison with the observed quadrupole is what
    # picks one, and it can pick none.
    strongest = min(rows, key=lambda r: r[4])
    print(f"\nstrongest suppression: p = {strongest[0]} -> "
          f"C_2/C_2^smooth = {strongest[4]:.4f} at {strongest[2]:.3e} Mpc^-1")
    print("each row is a prediction, not a fit; compare all of them against the "
          "observed quadrupole (cosmic variance at l=2 is ~+/-50%)")

    separation = 0.9933 / args.sigma
    print(f"separation between adjacent p: {separation:.2f} sigma "
          f"(0.9933 e-folds / {args.sigma} e-folds)")
    if separation < 1.0:
        print("  -> p is NOT resolved; report the family, not a selection")
    elif separation < 2.0:
        print("  -> p is marginally resolved; state it as a preference, not a determination")
    else:
        print("  -> p is resolved; the quadrupole ratio above is a genuine prediction")

    print("\ncaveat: N_* enters here as a pure translation in ln k. The induced "
          "second-order change in the smooth reference (n_s, alpha_s at the "
          "shifted pivot) is not modelled; re-run run_pipeline.py at the final "
          "N_* to confirm the selected model.")


if __name__ == "__main__":
    main()
