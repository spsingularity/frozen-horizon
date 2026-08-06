"""Evaluate Planck 2018 low-l likelihoods (native python: Commander-Gibbs TT,
SimAll EE prob-table) on featured vs smooth spectra from frozen_horizon runs.

Usage: .venv/bin/python scripts/lowl_likelihood_eval.py [run1 run2 ...]
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = os.path.join(REPO, "cobaya_packages")

from cobaya.likelihoods.planck_2018_lowl.TT import TT  # noqa: E402
from cobaya.likelihoods.planck_2018_lowl.EE import EE  # noqa: E402

tt_like = TT(packages_path=PACKAGES)
ee_like = EE(packages_path=PACKAGES)


def dl_from_raw(cl_raw):
    """raw C_l (zero-based, muK^2) -> D_l = l(l+1)C_l/2pi (zero-based, muK^2)."""
    ells = np.arange(cl_raw.shape[0])
    return cl_raw * ells * (ells + 1) / (2 * np.pi)


def chi2s(featured, smooth):
    """featured/smooth: (lmax+1, 4) raw C_l arrays, cols TT EE BB TE."""
    out = {}
    for tag, arr in (("featured", featured), ("smooth", smooth)):
        dl_tt = dl_from_raw(arr[:, 0])
        dl_ee = dl_from_raw(arr[:, 1])
        out[tag] = {
            "TT": -2.0 * tt_like.log_likelihood(dl_tt),
            "EE": -2.0 * ee_like.log_likelihood(dl_ee),
        }
    return out


def validate():
    """Sanity check on Planck 2018 best-fit LCDM."""
    import camb

    pars = camb.set_params(
        H0=67.36, ombh2=0.02237, omch2=0.1200, tau=0.0544,
        As=2.100e-9, ns=0.9649, lmax=300, lens_potential_accuracy=1,
    )
    res = camb.get_results(pars)
    cl = res.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["lensed_scalar"]
    dl_tt = dl_from_raw(cl[:, 0])
    dl_ee = dl_from_raw(cl[:, 1])
    c_tt = -2.0 * tt_like.log_likelihood(dl_tt)
    c_ee = -2.0 * ee_like.log_likelihood(dl_ee)
    print(f"VALIDATION (Planck 2018 best-fit LCDM, 28 multipoles each):")
    print(f"  -2lnL TT (Commander Gibbs native) = {c_tt:.3f}")
    print(f"  -2lnL EE (SimAll prob-table native) = {c_ee:.3f}")
    print()


def evaluate_run(run):
    from frozen_horizon import boltzmann

    d = np.genfromtxt(
        os.path.join(REPO, "results", run, "primordial_power.csv"),
        delimiter=",", names=True,
    )
    with open(os.path.join(REPO, "results", run, "summary.json")) as f:
        meta = json.load(f)
    obs = meta["observables"]
    ratio, featured, smooth = boltzmann.feature_ratio(
        d["k_Mpc"], d["P_R"], obs["n_s"], obs["alpha_s"], lmax=300
    )
    r = chi2s(featured, smooth)
    dtt = r["featured"]["TT"] - r["smooth"]["TT"]
    dee = r["featured"]["EE"] - r["smooth"]["EE"]
    return {
        "run": run,
        "n_s": obs["n_s"],
        "alpha_s": obs["alpha_s"],
        "chi2_TT_feat": r["featured"]["TT"],
        "chi2_TT_smooth": r["smooth"]["TT"],
        "dTT": dtt,
        "chi2_EE_feat": r["featured"]["EE"],
        "chi2_EE_smooth": r["smooth"]["EE"],
        "dEE": dee,
        "dTOT": dtt + dee,
    }


def main():
    validate()
    runs = sys.argv[1:] or ["reheat_p62", "reheat_p63", "reheat_p64"]
    rows = []
    for run in runs:
        print(f"evaluating {run} ...", flush=True)
        rows.append(evaluate_run(run))

    hdr = (f"{'run':<14} {'TT_feat':>9} {'TT_smooth':>9} {'dTT':>8} "
           f"{'EE_feat':>9} {'EE_smooth':>9} {'dEE':>8} {'dTT+EE':>8}")
    print()
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<14} {r['chi2_TT_feat']:>9.3f} {r['chi2_TT_smooth']:>9.3f} "
              f"{r['dTT']:>8.3f} {r['chi2_EE_feat']:>9.3f} {r['chi2_EE_smooth']:>9.3f} "
              f"{r['dEE']:>8.3f} {r['dTOT']:>8.3f}")
    print()
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
