"""Scalar and tensor mode integration on the exact background.

Modes satisfy v_k'' + (k^2 - z''/z) v_k = 0 with z = a phi_dot / H (scalar) and
u_k'' + (k^2 - a''/a) u_k = 0 (tensor), written here in e-fold time and
initialized in an adiabatic subhorizon state.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import config


def solve_mode(background, k_ratio, scalar=True, start_ratio=None,
               rtol=None, atol_rel=None, max_step=None):
    """Return the primordial power of one mode.

    k_ratio is k / (aH) at pivot crossing, so k_ratio = 1 is the pivot mode.
    """
    start_ratio = config.MODE_START_RATIO if start_ratio is None else start_ratio
    rtol = config.MODE_RTOL if rtol is None else rtol
    atol_rel = config.MODE_ATOL_REL if atol_rel is None else atol_rel
    max_step = config.MODE_MAX_STEP if max_step is None else max_step

    k = k_ratio * background.H_pivot

    def ratio_at(at_N):
        return background.k_over_aH(at_N, k)

    if ratio_at(background.start_N) < start_ratio:
        raise RuntimeError(
            f"Background starts too late for k/k_pivot={k_ratio:.3e}: "
            f"k/(aH) = {ratio_at(background.start_N):.2f} < {start_ratio}"
        )
    mode_start = brentq(
        lambda nn: ratio_at(nn) - start_ratio,
        background.start_N,
        background.end_N,
    )

    a0 = np.exp(mode_start - background.pivot_N)
    _, v0, _, _, dv0 = background.quantities(mode_start)
    ratio0 = ratio_at(mode_start)

    if scalar:
        amplitude = 1.0 / (a0 * abs(v0) * np.sqrt(2.0 * k))
        log_derivative = -(1.0 + dv0 / v0) - 1j * ratio0
    else:
        amplitude = 2.0 / (a0 * np.sqrt(2.0 * k))
        log_derivative = -1.0 - 1j * ratio0
    initial = np.array([amplitude + 0j, amplitude * log_derivative])

    def rhs(at_N, state):
        _, velocity, eps, _, dv = background.quantities(at_N)
        ratio = ratio_at(at_N)
        friction = 3.0 - eps
        if scalar:
            friction += 2.0 * dv / velocity
        return np.array([state[1], -friction * state[1] - ratio**2 * state[0]])

    sol = solve_ivp(
        rhs,
        (mode_start, background.end_N),
        initial,
        rtol=rtol,
        atol=atol_rel * amplitude,
        max_step=max_step,
    )
    final = sol.y[0, -1]
    norm = 2.0 * np.pi**2 if scalar else np.pi**2
    return k**3 * abs(final) ** 2 / norm


def transfer_table(background, k_ratios, **kwargs):
    """Scalar and tensor power on a grid of k/k_pivot."""
    k_ratios = np.asarray(k_ratios, dtype=float)
    scalar = np.array([solve_mode(background, k, True, **kwargs) for k in k_ratios])
    tensor = np.array([solve_mode(background, k, False, **kwargs) for k in k_ratios])
    return k_ratios, scalar, tensor


def smooth_reference(k_ratios, pivot_power, tilt):
    """Featureless power-law reference used to define the transfer ratio."""
    return pivot_power * k_ratios**tilt


def hubble_flow_tilts(background):
    """First-order n_s and r from the Hubble-flow parameters at pivot."""
    _, v_p, eps_p, _, dv_p = background.quantities(background.pivot_N)
    eta_h = 2.0 * dv_p / v_p + 2.0 * eps_p
    return {
        "ns_first_order": float(1.0 - 2.0 * eps_p - eta_h),
        "r_first_order": float(16.0 * eps_p),
        "epsilon_pivot": float(eps_p),
    }
