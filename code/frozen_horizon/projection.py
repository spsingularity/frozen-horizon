"""Large-angle Sachs-Wolfe projection of a primordial transfer table.

C_l^TT ~ (4 pi / 25) int d ln k P_R(k) j_l^2(k chi_*).

This is an approximation retained as a cross-check on the Boltzmann result: it
omits the late integrated Sachs-Wolfe term, which is a large fraction of C_2 in
LCDM and will dilute any primordial suppression. Ratios computed here should be
read as upper bounds on the observable effect.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import spherical_jn

from . import config


def projected_ratio(ell, k_ratio, transfer, n_s, alpha_s,
                    k_pivot=None, chi_star=None, samples=120000):
    """C_l / C_l^smooth for one multipole, given a transfer ratio table."""
    k_pivot = config.K_PIVOT if k_pivot is None else k_pivot
    chi_star = config.CHI_STAR if chi_star is None else chi_star

    interpolation = PchipInterpolator(np.log(k_ratio), np.log(transfer))
    log_ratio = np.linspace(np.log(k_ratio.min()), np.log(k_ratio.max()), samples)
    ratio = np.exp(log_ratio)

    smooth_shape = np.exp((n_s - 1.0) * log_ratio + 0.5 * alpha_s * log_ratio**2)
    transfer_dense = np.exp(interpolation(log_ratio))
    kernel = spherical_jn(ell, ratio * k_pivot * chi_star) ** 2

    denominator = np.trapezoid(smooth_shape * kernel, log_ratio)
    numerator = np.trapezoid(smooth_shape * transfer_dense * kernel, log_ratio)
    return float(numerator / denominator)


def projection_table(k_ratio, transfer, n_s, alpha_s, ells=range(2, 51), **kwargs):
    """C_l / C_l^smooth over a range of multipoles."""
    return {
        int(ell): projected_ratio(ell, k_ratio, transfer, n_s, alpha_s, **kwargs)
        for ell in ells
    }
