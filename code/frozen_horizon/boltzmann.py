"""Boltzmann propagation of the exact primordial spectrum through CAMB.

The Sachs-Wolfe estimate in projection.py omits the late integrated Sachs-Wolfe
term, which is a substantial fraction of C_2 in LCDM and is sourced by the same
primordial modes. It therefore *dilutes* any primordial suppression, and the SW
ratio should be read as an upper bound on the observable effect. This module
computes the real thing.

Two modelling choices have to be declared explicitly rather than inherited from
wherever the mode table happened to stop:

* Above the table, the transfer has returned to unity, so the spectrum is
  continued as the smooth power law with running. Harmless: these scales are
  far from the feature.
* Below the table, the scalar transfer rises steeply (P/P_smooth ~ 10^3 and
  climbing). The model interprets that rise stochastically -- it is generated
  before the trajectory reaches the classical attractor -- so it is NOT an
  ordinary linear spectrum inside one universe. The SW kernel suppresses it to
  irrelevance, but CAMB would take it at face value in the ISW and matter
  sectors. FAR_IR_MODE controls what happens there and must be stated in the
  paper.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import config


def build_primordial(k_Mpc, power, n_s, alpha_s, k_pivot=None,
                     k_min=1.0e-6, k_max=10.0, samples=4096,
                     far_ir="freeze"):
    """Return (k, P_R) on a grid wide enough for CAMB, with the tails declared.

    far_ir:
      "freeze"   -- hold the transfer at its value at the lowest tabulated k,
                    i.e. keep the smooth power law and drop the stochastic rise
      "truncate" -- force the transfer to 1 below the table (pure power law)
      "keep"     -- extrapolate the rise as a power law in k (NOT recommended;
                    treats pre-classical modes as ordinary linear perturbations)
    """
    k_pivot = config.K_PIVOT if k_pivot is None else k_pivot

    ln_k = np.log(k_Mpc)
    smooth = lambda lk: np.exp(  # noqa: E731
        (n_s - 1.0) * (lk - np.log(k_pivot))
        + 0.5 * alpha_s * (lk - np.log(k_pivot)) ** 2
    )
    amplitude = power[np.argmin(np.abs(k_Mpc - k_pivot))] / smooth(
        np.log(k_Mpc[np.argmin(np.abs(k_Mpc - k_pivot))])
    )
    transfer = power / (amplitude * smooth(ln_k))

    interpolate = PchipInterpolator(ln_k, np.log(transfer))
    grid = np.linspace(np.log(k_min), np.log(k_max), samples)

    inside = (grid >= ln_k.min()) & (grid <= ln_k.max())
    log_transfer = np.zeros_like(grid)
    log_transfer[inside] = interpolate(grid[inside])
    log_transfer[grid > ln_k.max()] = 0.0  # transfer -> 1 at high k

    low = grid < ln_k.min()
    if far_ir == "truncate":
        log_transfer[low] = 0.0
    elif far_ir == "freeze":
        log_transfer[low] = np.log(transfer[0])
    elif far_ir == "keep":
        slope = (np.log(transfer[1]) - np.log(transfer[0])) / (ln_k[1] - ln_k[0])
        log_transfer[low] = np.log(transfer[0]) + slope * (grid[low] - ln_k.min())
    else:
        raise ValueError(f"unknown far_ir mode {far_ir!r}")

    return np.exp(grid), amplitude * smooth(grid) * np.exp(log_transfer)


def spectra(k_Mpc, power_R, n_s, alpha_s, cosmology=None, lmax=2500,
            lensed=True, **kwargs):
    """Run CAMB on a tabulated primordial spectrum and return the C_l."""
    k_grid, p_grid = build_primordial(k_Mpc, power_R, n_s, alpha_s, **kwargs)
    return _spectra_from_grid(
        k_grid, p_grid, cosmology=cosmology, lmax=lmax, lensed=lensed,
        effective_ns=n_s,
    )


def feature_ratio(k_Mpc, power_R, n_s, alpha_s, lmax=2500, lensed=True,
                  cosmology=None, **kwargs):
    """C_l with the feature, divided by C_l from an identical smooth model.

    The reference uses the same cosmology, amplitude, n_s and alpha_s with the
    transfer forced to unity everywhere, so the ratio isolates the primordial
    feature and the radiation transfer functions cancel. Returns
    (ratio_dict, featured, smooth) where the C_l arrays are raw (not l(l+1)/2pi).
    """
    common = dict(lmax=lmax, lensed=lensed, cosmology=cosmology)

    featured = spectra(k_Mpc, power_R, n_s, alpha_s, **common, **kwargs)

    # Same grid, transfer identically 1: a pure power law with running.
    smooth_kwargs = {k: v for k, v in kwargs.items() if k != "far_ir"}
    k_grid, p_smooth = build_primordial(
        k_Mpc, power_R, n_s, alpha_s, far_ir="truncate", **smooth_kwargs
    )
    ln_k = np.log(k_grid)
    pivot_index = int(np.argmin(np.abs(k_grid - config.K_PIVOT)))
    amplitude = p_smooth[pivot_index] / np.exp(
        (n_s - 1.0) * (ln_k[pivot_index] - np.log(config.K_PIVOT))
    )
    p_smooth = amplitude * np.exp(
        (n_s - 1.0) * (ln_k - np.log(config.K_PIVOT))
        + 0.5 * alpha_s * (ln_k - np.log(config.K_PIVOT)) ** 2
    )
    smooth = _spectra_from_grid(k_grid, p_smooth, effective_ns=n_s, **common)

    ells = np.arange(featured.shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = {
            name: np.where(smooth[:, col] != 0.0,
                           featured[:, col] / smooth[:, col], np.nan)
            for col, name in enumerate(("TT", "EE", "BB", "TE"))
        }
    ratio["ell"] = ells
    return ratio, featured, smooth


def _spectra_from_grid(k_grid, p_grid, cosmology=None, lmax=2500, lensed=True,
                       effective_ns=0.965):
    """Run CAMB on an already-built (k, P) grid."""
    import camb

    cosmology = {**config.FIDUCIAL_COSMOLOGY, **(cosmology or {})}
    params = camb.CAMBparams()
    params.set_cosmology(
        H0=cosmology["H0"],
        ombh2=cosmology["ombh2"],
        omch2=cosmology["omch2"],
        tau=cosmology["tau"],
    )
    params.set_for_lmax(lmax, lens_potential_accuracy=1)
    initial = camb.initialpower.SplinedInitialPower(ks=k_grid, PK=p_grid)
    # CAMB needs a spectral index for its nonlinear fitting formula; the
    # splined table carries no analytic tilt of its own.
    initial.effective_ns_for_nonlinear = effective_ns
    params.InitPower = initial
    results = camb.get_results(params)
    powers = results.get_cmb_power_spectra(params, CMB_unit="muK", raw_cl=True)
    return powers["lensed_scalar" if lensed else "unlensed_scalar"]
