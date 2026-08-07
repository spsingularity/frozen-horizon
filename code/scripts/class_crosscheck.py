"""Independent CLASS reproduction of the CAMB feature ratios.

Paper I states that CLASS, fed the same primordial tables through an entirely
separate input path, reproduces every CAMB ratio at l <= 30 to better than a
quoted tolerance. That claim was previously asserted in a caption string with
no code behind it. This script performs the comparison so the number is a
build artifact like everything else.

The two codes are driven as independently as their interfaces allow:

  CAMB  -- SplinedInitialPower on the (k, P) grid, via frozen_horizon.boltzmann
  CLASS -- 'P_k_ini type: external_Pk' reading the same grid from a file

Both are given the identical fiducial cosmology and the identical pair of
primordial tables (featured, and the smooth power law built from the same
n_s and alpha_s), and each forms its own featured/smooth ratio, so radiation
transfer, reionization and lensing cancel within each code separately. What
is compared is therefore the ratio, not the absolute C_l -- the same quantity
the manuscript tabulates.

Usage:  ./.venv/bin/python scripts/class_crosscheck.py [run ...]
"""

import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from frozen_horizon import boltzmann, config  # noqa: E402

LMAX_COMPARE = 30


def class_spectra(k_grid, p_grid, lmax=2500):
    """Lensed C_l from CLASS for an external primordial table (raw, muK^2)."""
    from classy import Class

    cosmology = config.FIDUCIAL_COSMOLOGY
    handle, path = tempfile.mkstemp(suffix=".dat", text=True)
    try:
        with os.fdopen(handle, "w") as stream:
            for k, power in zip(k_grid, p_grid):
                stream.write(f"{k:.16e} {power:.16e}\n")

        cosmo = Class()
        cosmo.set({
            "output": "tCl,pCl,lCl",
            "lensing": "yes",
            "l_max_scalars": lmax,
            "H0": cosmology["H0"],
            "omega_b": cosmology["ombh2"],
            "omega_cdm": cosmology["omch2"],
            "tau_reio": cosmology["tau"],
            "P_k_ini type": "external_Pk",
            "command": f"cat {path}",
            "N_ur": 2.0328,
            "N_ncdm": 1,
            "m_ncdm": 0.06,
        })
        cosmo.compute()
        # CLASS returns dimensionless C_l; scale to muK^2 to match CAMB.
        cls = cosmo.lensed_cl(lmax)
        t_cmb_muk = cosmo.T_cmb() * 1.0e6
        out = {name: cls[key] * t_cmb_muk**2
               for name, key in (("TT", "tt"), ("EE", "ee"), ("TE", "te"))}
        cosmo.struct_cleanup()
        cosmo.empty()
        return out
    finally:
        if os.path.exists(path):
            os.remove(path)


def smooth_grid(k_Mpc, power_R, n_s, alpha_s):
    """The same smooth reference boltzmann.feature_ratio builds internally."""
    k_grid, p_smooth = boltzmann.build_primordial(
        k_Mpc, power_R, n_s, alpha_s, far_ir="truncate")
    ln_k = np.log(k_grid)
    pivot = int(np.argmin(np.abs(k_grid - config.K_PIVOT)))
    amplitude = p_smooth[pivot] / np.exp(
        (n_s - 1.0) * (ln_k[pivot] - np.log(config.K_PIVOT)))
    p_smooth = amplitude * np.exp(
        (n_s - 1.0) * (ln_k - np.log(config.K_PIVOT))
        + 0.5 * alpha_s * (ln_k - np.log(config.K_PIVOT)) ** 2)
    return k_grid, p_smooth


def compare(run, lmax=2500):
    table = np.genfromtxt(os.path.join(ROOT, "results", run,
                                       "primordial_power.csv"),
                          delimiter=",", names=True)
    meta = json.loads(open(os.path.join(ROOT, "results", run,
                                        "summary.json")).read())["observables"]
    n_s, alpha_s = meta["n_s"], meta["alpha_s"]

    camb_ratio, _, _ = boltzmann.feature_ratio(
        table["k_Mpc"], table["P_R"], n_s, alpha_s, lmax=lmax)

    k_feat, p_feat = boltzmann.build_primordial(
        table["k_Mpc"], table["P_R"], n_s, alpha_s)
    k_smooth, p_smooth = smooth_grid(table["k_Mpc"], table["P_R"], n_s, alpha_s)

    class_feat = class_spectra(k_feat, p_feat, lmax=lmax)
    class_smooth = class_spectra(k_smooth, p_smooth, lmax=lmax)

    worst = {}
    for name in ("TT", "EE", "TE"):
        with np.errstate(divide="ignore", invalid="ignore"):
            class_r = np.where(class_smooth[name] != 0.0,
                               class_feat[name] / class_smooth[name], np.nan)
        ells = np.arange(2, LMAX_COMPARE + 1)
        deviation = np.abs(class_r[ells] - camb_ratio[name][ells])
        worst[name] = float(np.nanmax(deviation))
    return {"run": run, "max_abs_deviation": worst,
            "worst_overall": float(max(worst.values()))}


def main():
    runs = sys.argv[1:] or ["invwall_p65", "invwall_p66"]
    rows = []
    print(f"CAMB vs CLASS on the feature ratio, l = 2..{LMAX_COMPARE}\n")
    print(f"{'run':>14} {'TT':>10} {'EE':>10} {'TE':>10} {'worst':>10}")
    for run in runs:
        row = compare(run)
        rows.append(row)
        w = row["max_abs_deviation"]
        print(f"{run:>14} {w['TT']:>10.2e} {w['EE']:>10.2e} "
              f"{w['TE']:>10.2e} {row['worst_overall']:>10.2e}")
    destination = os.path.join(ROOT, "results", "class_crosscheck.json")
    with open(destination, "w") as handle:
        json.dump(rows, handle, indent=1)
        handle.write("\n")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main()
