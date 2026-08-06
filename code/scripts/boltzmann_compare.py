"""Propagate each p through CAMB and report what the low-l CMB can actually say.

    ./.venv/bin/python scripts/boltzmann_compare.py --p 62 63 64

For every p this computes the lensed C_l ratio against an identical smooth model
and the cosmic-variance-limited Delta chi^2 -- the *best case* significance, for
a full-sky noiseless experiment. Any real likelihood does worse. Reporting this
bound is what stops a sub-variance feature being presented as an explanation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frozen_horizon import boltzmann  # noqa: E402


def delta_chi2(ratio, lmax, f_sky=0.7, lmin=2):
    """Expected -2 dlnL between featured and smooth models, cosmic variance only.

    For a Gaussian field, <-2 dlnL> = f_sky sum (2l+1) [ln(1/x) + x - 1] with
    x = C_featured / C_smooth. Vanishes quadratically as x -> 1, so it is a
    genuine information measure rather than a raw fractional difference.
    """
    ell = np.arange(lmin, lmax + 1)
    x = np.asarray(ratio)[lmin:lmax + 1]
    good = np.isfinite(x) & (x > 0)
    return float(f_sky * np.sum((2 * ell + 1)[good] * (np.log(1.0 / x[good]) + x[good] - 1.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, nargs="+", default=[62, 63, 64])
    parser.add_argument("--prefix", default="reheat_p")
    parser.add_argument("--lmax", type=int, default=2500)
    parser.add_argument("--f-sky", type=float, default=0.7)
    args = parser.parse_args()

    summary = {}
    for p in args.p:
        directory = ROOT / "results" / f"{args.prefix}{p}"
        power = np.genfromtxt(
            directory / "primordial_power.csv", delimiter=",", names=True
        )
        meta = json.loads((directory / "summary.json").read_text())
        obs = meta["observables"]

        print(f"\n=== p = {p}  (N_* = {meta['inputs']['n_star']:.4f}, "
              f"n_s = {obs['n_s']:.5f}) ===", flush=True)
        ratio, _, _ = boltzmann.feature_ratio(
            power["k_Mpc"], power["P_R"], obs["n_s"], obs["alpha_s"],
            lmax=args.lmax,
        )
        np.savez(directory / "camb_ratio.npz", **ratio)

        print(f"{'l':>4} {'TT':>8} {'EE':>8} {'TE':>8}")
        for ell in (2, 3, 4, 5, 10, 20, 30):
            print(f"{ell:>4} {ratio['TT'][ell]:>8.4f} {ratio['EE'][ell]:>8.4f} "
                  f"{ratio['TE'][ell]:>8.4f}")

        chi_tt = delta_chi2(ratio["TT"], 30, args.f_sky)
        chi_ee = delta_chi2(ratio["EE"], 30, args.f_sky)
        print(f"  CV-limited (l<=30, f_sky={args.f_sky}): "
              f"TT {np.sqrt(max(chi_tt,0)):.2f} sigma, "
              f"TT+EE {np.sqrt(max(chi_tt+chi_ee,0)):.2f} sigma")

        summary[p] = {
            "n_star": meta["inputs"]["n_star"],
            "n_s": obs["n_s"],
            "r": obs["r"],
            "alpha_s": obs["alpha_s"],
            "notch_k_Mpc": meta["notch"]["notch_k_Mpc"],
            "TT": {str(l): float(ratio["TT"][l]) for l in (2, 3, 4, 5, 10, 20, 30)},
            "EE": {str(l): float(ratio["EE"][l]) for l in (2, 3, 4, 5, 10, 20, 30)},
            "dchi2_TT_lmax30": chi_tt,
            "dchi2_TTEE_lmax30": chi_tt + chi_ee,
            "sigma_TT_lmax30": float(np.sqrt(max(chi_tt, 0.0))),
            "sigma_TTEE_lmax30": float(np.sqrt(max(chi_tt + chi_ee, 0.0))),
        }

    out = ROOT / "results" / "boltzmann_comparison.json"
    out.write_text(json.dumps(summary, indent=2))

    print("\n\n=== summary ===")
    print(f"{'p':>4} {'n_s':>9} {'C2^TT':>8} {'C2^EE':>8} {'sig(TT)':>8} {'sig(TT+EE)':>11}")
    for p, row in summary.items():
        print(f"{p:>4} {row['n_s']:>9.5f} {row['TT']['2']:>8.4f} "
              f"{row['EE']['2']:>8.4f} {row['sigma_TT_lmax30']:>8.2f} "
              f"{row['sigma_TTEE_lmax30']:>11.2f}")
    print(f"\nwrote {out}")
    print("these are BEST-CASE significances; a real likelihood with masking, "
          "noise and foregrounds does worse")


if __name__ == "__main__":
    main()
