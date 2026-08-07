"""Why the exact SimAll likelihood is used for EE rather than diagonal errors.

Paper I states that a simpler diagnostic built from per-multipole errors
overestimates the polarization penalty of the broad pattern relative to the
exact non-Gaussian SimAll likelihood. That comparison is made here so the
quoted numbers regenerate rather than being asserted.

SimAll supplies, for each multipole 2 <= l <= 29, a tabulated probability
P(C_l) on a grid -- not a Gaussian. The tail of that distribution is strongly
skewed at low l, which is exactly the regime a suppressed model probes. The
diagonal approximation replaces each tabulated P(C_l) by the Gaussian with the
same mean and variance and adds the resulting chi^2 in quadrature across
multipoles:

    chi2_diag = sum_l (D_l^model - <D_l>)^2 / var_l ,

against the exact

    chi2_exact = -2 sum_l ln P_l(D_l^model) .

Both are evaluated on the featured and smooth spectra of a run and differenced,
so the comparison is like for like.

Usage:  ./.venv/bin/python scripts/ee_diagonal_diagnostic.py [run ...]
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = os.path.join(REPO, "cobaya_packages")

from cobaya.likelihoods.planck_2018_lowl.EE import EE  # noqa: E402

EE_LIKE = EE(packages_path=PACKAGES)
LMIN, LMAX = 2, 29


def gaussian_moments():
    """Mean and variance of the SimAll distribution at each multipole.

    probEE has shape (n_grid, n_ell) and holds LOG probabilities: the exact
    likelihood indexes it by int(D_l / step) and sums, so entry [i, l] is
    ln P_l at D_l = i * step. Both facts are taken from the likelihood object
    rather than assumed, since getting either wrong silently rescales the
    moments and would make the comparison meaningless.
    """
    log_table = np.asarray(EE_LIKE.probEE, dtype=float)
    n_grid, n_ell = log_table.shape
    step = EE_LIKE._stepEE
    grid = (np.arange(n_grid) * step)[:, None]            # (n_grid, 1)

    weight = np.exp(log_table - log_table.max(axis=0, keepdims=True))
    weight /= weight.sum(axis=0, keepdims=True)
    mean = (grid * weight).sum(axis=0)
    var = ((grid - mean) ** 2 * weight).sum(axis=0)
    return mean, var, n_ell


def dl_from_raw(cl_raw):
    ells = np.arange(cl_raw.shape[0])
    return cl_raw * ells * (ells + 1) / (2 * np.pi)


def chi2_diagonal(dl_ee, mean, var, n_ell):
    model = dl_ee[LMIN:LMIN + n_ell]
    return float(np.sum((model - mean) ** 2 / var))


def evaluate(run, mean, var, n_ell):
    from frozen_horizon import boltzmann

    table = np.genfromtxt(
        os.path.join(REPO, "results", run, "primordial_power.csv"),
        delimiter=",", names=True)
    meta = json.loads(open(
        os.path.join(REPO, "results", run, "summary.json")).read())["observables"]
    _, featured, smooth = boltzmann.feature_ratio(
        table["k_Mpc"], table["P_R"], meta["n_s"], meta["alpha_s"], lmax=300)

    out = {}
    for tag, arr in (("featured", featured), ("smooth", smooth)):
        dl_ee = dl_from_raw(arr[:, 1])
        out[tag] = {
            "exact": -2.0 * EE_LIKE.log_likelihood(dl_ee),
            "diagonal": chi2_diagonal(dl_ee, mean, var, n_ell),
        }
    return {
        "run": run,
        "dEE_exact": out["featured"]["exact"] - out["smooth"]["exact"],
        "dEE_diagonal": out["featured"]["diagonal"] - out["smooth"]["diagonal"],
    }


def main():
    runs = sys.argv[1:] or ["invwall_p65", "invwall_p66"]
    mean, var, n_ell = gaussian_moments()
    print(f"SimAll table: {n_ell} multipoles, l = {LMIN}..{LMIN + n_ell - 1}\n")
    print(f"{'run':>14} {'dEE exact':>11} {'dEE diagonal':>13} {'ratio':>8}")
    rows = []
    for run in runs:
        row = evaluate(run, mean, var, n_ell)
        rows.append(row)
        ratio = (row["dEE_diagonal"] / row["dEE_exact"]
                 if row["dEE_exact"] else float("nan"))
        print(f"{run:>14} {row['dEE_exact']:>11.3f} "
              f"{row['dEE_diagonal']:>13.3f} {ratio:>8.1f}")
    destination = os.path.join(REPO, "results", "ee_diagonal_diagnostic.json")
    with open(destination, "w") as handle:
        json.dump(rows, handle, indent=1)
        handle.write("\n")
    print(f"\nwrote {destination}")
    print("The diagonal figure is the one Paper I Sec. 6 quotes as the "
          "overestimate; the exact figure is the one used everywhere else.")


if __name__ == "__main__":
    main()
