"""Scalaron reheating and the self-consistent determination of N_*.

Why this module exists
----------------------
The observable location of the primordial feature is

    ln(k_notch / k_*) = N_* - N_total(p) + const,

with dN_total/dp = +0.975 and dN_*/d(N_*) = 1, so p and N_* are exactly
degenerate as long as N_* is a free parameter. The degeneracy is broken by
noticing that N_* is *not* free: this f(R) reduces to R + R^2/(6M^2) exactly at
low curvature, so reheating is ordinary Starobinsky reheating, and M is already
pinned by the observed scalar amplitude.

Crucially the reheating inputs are p-blind. Measured across p = 66, 67, 68:

    Delta ln V_*   = 0.00175
    Delta ln rho_end = 0.00175
    Delta M        = 0.09 %

which propagate to Delta N_* ~ 0.001 e-folds, against 0.99 e-folds of feature
motion per unit p. So N_* may be computed once and used for every p.

N_* must be solved as a fixed point: N_* sets where the pivot sits, which sets
V_* and the A_s-normalized M, which set Gamma and T_reh, which set N_*.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import config


def scalaron_decay_rate(mass_scale, coefficient=None):
    """Total scalaron width Gamma / M_Pl, with M_Pl reduced.

    The scalaron couples to matter through the trace of the stress tensor,
    L ~ (phi / sqrt(6) M_Pl) T^mu_mu, so the width scales as M^3 / M_Pl^2 with a
    dimensionless coefficient set by the available non-conformal channels.
    """
    coefficient = (
        config.SCALARON_DECAY_COEFFICIENT if coefficient is None else coefficient
    )
    return coefficient * mass_scale**3


def reheating_temperature(gamma, g_star=None):
    """T_reh / M_Pl in the sudden-decay approximation, M_Pl reduced.

    Equating H = Gamma with the radiation Friedmann equation
    3 H^2 M_Pl^2 = (pi^2/30) g_* T^4 gives
    T_reh = (90 / (pi^2 g_*))^(1/4) sqrt(Gamma M_Pl).
    """
    g_star = config.G_STAR if g_star is None else g_star
    return (90.0 / (np.pi**2 * g_star)) ** 0.25 * np.sqrt(gamma)


def rho_from_temperature(temperature, g_star=None):
    """Radiation energy density (pi^2/30) g_* T^4 in reduced-Planck units."""
    g_star = config.G_STAR if g_star is None else g_star
    return np.pi**2 / 30.0 * g_star * temperature**4


def matching_n_star(v_star, rho_end, rho_reh, w=None, g_star_s=None, k_pivot=None):
    """Standard e-fold matching between pivot crossing and today.

    ALL DENSITIES MUST BE IN REDUCED-PLANCK UNITS (M_Pl = 1). The (1/4) term is
    really (1/4) ln[V_*^2 / (M_Pl^4 rho_end)] and is only unit-correct here
    because of that convention.

    N_* = D(k_*) + (1/4) ln(V_*^2 / rho_end)
          + (1 - 3w) / (12 (1 + w)) * ln(rho_reh / rho_end)
          - (1/12) ln(g_*s)

    Planck 2018 X (arXiv:1807.06211). D already absorbs -ln[k_*/(a_0 H_0)],
    which makes it exactly H_0-independent -- see config.N_STAR_MATCHING_D.
    """
    w = config.W_REHEATING if w is None else w
    g_star_s = config.G_STAR_S if g_star_s is None else g_star_s
    k_pivot = config.K_PIVOT if k_pivot is None else k_pivot

    try:
        d_constant = config.N_STAR_MATCHING_D[k_pivot]
    except KeyError:
        raise KeyError(
            f"No matching constant D tabulated for k_* = {k_pivot} Mpc^-1; "
            f"have {sorted(config.N_STAR_MATCHING_D)}"
        ) from None

    reheating_term = (1.0 - 3.0 * w) / (12.0 * (1.0 + w)) * np.log(rho_reh / rho_end)
    return (
        d_constant
        + 0.25 * np.log(v_star**2 / rho_end)
        + reheating_term
        - np.log(g_star_s) / 12.0
    )


def decay_rate_sm(mass_scale, xi_higgs=None, anomaly_bracket=None):
    """Gamma/M_Pl for the minimal Standard Model completion.

    Gamma = M^3/(192 pi M_Pl^2) [ 4(1 - 6 xi_H)^2 + anomaly ], where the first
    term is decay into the four real Higgs components and the second is the
    gauge trace-anomaly contribution, sum_i b_i^2 alpha_i^2(M/2) N_i^adj /(4pi^2).

    xi_H = 0 (minimal coupling) leaves the Higgs bracket at 4; xi_H = 1/6
    (conformal) switches the tree channel off entirely and leaves only the
    anomaly, which is ~500x smaller and costs about one e-fold of N_*.
    """
    xi_higgs = config.XI_HIGGS if xi_higgs is None else xi_higgs
    anomaly_bracket = (
        config.ANOMALY_BRACKET if anomaly_bracket is None else anomaly_bracket
    )
    bracket = 4.0 * (1.0 - 6.0 * xi_higgs) ** 2 + anomaly_bracket
    return mass_scale**3 * bracket / (192.0 * np.pi)


def integrate_reheating(rho_end, gamma, g_star=None, max_efolds=60.0):
    """Resolve the scalaron/radiation system instead of assuming sudden decay.

        drho_phi/dN = -3 rho_phi - (Gamma/H) rho_phi
        drho_r/dN   = -4 rho_r   + (Gamma/H) rho_phi

    Reheating is defined operationally by rho_r = rho_phi. This is a genuinely
    better treatment than equating H = Gamma: the two differ by roughly 0.15-0.2
    e-folds, which is ~20% of the spacing between adjacent p, and therefore can
    move which integer the theory selects.
    """
    g_star = config.G_STAR if g_star is None else g_star

    # Integrated in (ln rho_phi, ratio = rho_r/rho_phi). The densities fall by
    # ~e^-53 before equality, which underflows in linear variables; these stay
    # O(1). Ratio starts at 0 and the event is simply ratio = 1.
    #   dln rho_phi/dN = -3 - Gamma/H
    #   d ratio /dN    = -ratio + (Gamma/H)(1 + ratio)
    def rhs(efold, state):
        log_rho_phi, ratio = state
        hubble = np.sqrt(np.exp(log_rho_phi) * (1.0 + ratio) / 3.0)
        decay = gamma / hubble
        return [-3.0 - decay, -ratio + decay * (1.0 + ratio)]

    def equality(efold, state):
        return state[1] - 1.0

    equality.terminal = True
    equality.direction = 1

    solution = solve_ivp(
        rhs, (0.0, max_efolds), [np.log(rho_end), 0.0], events=equality,
        rtol=1e-10, atol=1e-12, dense_output=False,
    )
    if not solution.t_events[0].size:
        raise RuntimeError("radiation never reached equality with the scalaron")
    n_reheat = float(solution.t_events[0][0])
    # At equality rho_r = rho_phi, so the radiation density is exp(ln rho_phi).
    rho_radiation = float(np.exp(solution.y_events[0][0][0]))
    temperature = (30.0 * rho_radiation / (np.pi**2 * g_star)) ** 0.25
    return {
        "N_reheat": n_reheat,
        "rho_radiation": rho_radiation,
        "T_reh_over_Mpl": temperature,
        "T_reh_GeV": temperature * config.M_PL_GEV,
    }


def n_star_resolved(background, xi_higgs=None):
    """N_* from the resolved reheating history (no sudden-decay assumption).

    N_* = D0 + (1/2) ln V_* - N_re - ln T_re - (1/3) ln g_s,re

    with D0 = -ln k_* + ln T_0 + (1/3) ln g_s0 - (1/2) ln 3. This is the same
    matching chain as matching_n_star, written before T_re is eliminated in
    favour of rho_re, so an explicitly integrated N_re can be inserted.
    """
    energies = background.energy_densities()
    gamma = decay_rate_sm(energies["M_over_Mpl"], xi_higgs=xi_higgs)
    history = integrate_reheating(energies["rho_end"], gamma)

    n_star = (
        config.N_STAR_MATCHING_D0[config.K_PIVOT]
        + 0.5 * np.log(energies["V_star"])
        - history["N_reheat"]
        - np.log(history["T_reh_over_Mpl"])
        - np.log(config.G_STAR_S) / 3.0
    )
    return {
        **energies, **history,
        "Gamma_over_Mpl": float(gamma),
        "Gamma_GeV": float(gamma * config.M_PL_GEV),
        "xi_higgs": config.XI_HIGGS if xi_higgs is None else xi_higgs,
        "N_star_implied": float(n_star),
    }


def solve_n_star_resolved(background, xi_higgs=None, bracket=(35.0, 59.0)):
    """Fixed point of N_* using the resolved reheating history."""

    def residual(n_star):
        rebased = background.rebase(n_star)
        return n_star_resolved(rebased, xi_higgs=xi_higgs)["N_star_implied"] - n_star

    low, high = bracket
    if residual(low) * residual(high) > 0.0:
        raise RuntimeError(f"no fixed point in [{low}, {high}]")
    n_star = brentq(residual, low, high, xtol=1.0e-10)
    result = n_star_resolved(background.rebase(n_star), xi_higgs=xi_higgs)
    result["N_star"] = float(n_star)
    return result


def n_star_from_background(background, w=None):
    """Evaluate the matching formula for one pivot placement (no iteration)."""
    energies = background.energy_densities()
    gamma = scalaron_decay_rate(energies["M_over_Mpl"])
    temperature = reheating_temperature(gamma)
    rho_reh = rho_from_temperature(temperature)
    n_star = matching_n_star(
        energies["V_star"], energies["rho_end"], rho_reh, w=w
    )
    return {
        **energies,
        "Gamma_over_Mpl": float(gamma),
        "T_reh_over_Mpl": float(temperature),
        "T_reh_GeV": float(temperature * config.M_PL_GEV),
        "rho_reh": float(rho_reh),
        "N_star_implied": float(n_star),
    }


def solve_n_star(background, w=None, bracket=(35.0, 59.0), **kwargs):
    """Solve N_* = matching(N_*) as a fixed point.

    The upper bracket stops below the feature: once the pivot enters the wall
    region (N_* ~ 59 for p = 67) the slow-roll amplitude normalization that
    fixes M stops being meaningful, and the fixed point there would be spurious.
    """

    def residual(n_star):
        rebased = background.rebase(n_star)
        return n_star_from_background(rebased, w=w)["N_star_implied"] - n_star

    low, high = bracket
    if residual(low) * residual(high) > 0.0:
        raise RuntimeError(
            f"No N_* fixed point in [{low}, {high}]: "
            f"residuals {residual(low):.3f}, {residual(high):.3f}"
        )
    n_star = brentq(residual, low, high, xtol=1.0e-10, **kwargs)
    result = n_star_from_background(background.rebase(n_star), w=w)
    result["N_star"] = float(n_star)
    return result
