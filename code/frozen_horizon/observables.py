"""Local pivot observables from the exact mode solution.

n_s and alpha_s come from a polynomial fit of ln P against ln k over a narrow
window around the pivot. The window width matters: alpha_s is a second
derivative of a numerically integrated quantity and is the least converged
number the pipeline produces, so the fit is exposed as a parameter rather than
hard-coded.
"""

import numpy as np

from . import config, modes


def local_fit(background, spacing=0.03, half_points=2, degree=2, **kwargs):
    """Fit ln P_scalar(ln k) near the pivot and return the derived observables."""
    offsets = np.arange(-half_points, half_points + 1)
    k_ratios = np.exp(offsets * spacing)
    scalar = np.array(
        [modes.solve_mode(background, k, True, **kwargs) for k in k_ratios]
    )
    fit = np.polyfit(np.log(k_ratios), np.log(scalar), degree)

    pivot_index = half_points
    scalar_pivot = float(scalar[pivot_index])
    tensor_pivot = float(modes.solve_mode(background, 1.0, False, **kwargs))

    # Refine M so the exact mode amplitude, not the slow-roll estimate, matches
    # the observed A_s. Downstream stages must read this value, never retype it.
    corrected_mass = background.mass_scale * np.sqrt(config.A_S_OBS / scalar_pivot)

    return {
        "n_s": float(1.0 + fit[degree - 1]),
        "alpha_s": float(2.0 * fit[degree - 2]) if degree >= 2 else 0.0,
        "A_s_raw": scalar_pivot,
        "r": tensor_pivot / scalar_pivot,
        "M_over_Mpl_corrected": float(corrected_mass),
        "fit_spacing": spacing,
        "fit_half_points": half_points,
        "fit_degree": degree,
    }


def window_scan(background, spacings=(0.01, 0.02, 0.03, 0.05, 0.08),
                half_points=(2, 3, 4), **kwargs):
    """Vary the fitting window to expose how stable n_s and alpha_s really are."""
    rows = []
    for spacing in spacings:
        for half in half_points:
            if 2 * half + 1 < 3:
                continue
            result = local_fit(
                background, spacing=spacing, half_points=half, **kwargs
            )
            rows.append(
                {
                    "spacing": spacing,
                    "half_points": half,
                    "ln_k_halfwidth": spacing * half,
                    "n_s": result["n_s"],
                    "alpha_s": result["alpha_s"],
                }
            )
    return rows
