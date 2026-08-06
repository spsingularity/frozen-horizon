"""Causal-patch stochastic exit from the frozen horizon.

The coarse-grained unstable mode obeys dX = sX dN + (H/2pi) dW_N. Rescaling by
X = z H / (2 pi sqrt(s)) reduces the backward equation to

    (1/2) T'' + z T' = -1/s,    T(+/- 1/sqrt(s)) = 0,

so the first-passage moments in e-folds depend on s alone -- H cancels. The
Hubble rate re-enters only through the classicality boundary X_c, which sets
where the subsequent classical trajectory begins.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.special import hankel1
from scipy.stats import norm

from . import background as bg, config


def tachyonic_kick_factor(epsilon=1.0, mu2=3.0):
    """sqrt(Q_nu(eps)): the tachyonic enhancement of the stochastic kick.

    The frequently used massless kick H/(2 pi) is wrong at this horizon:
    m^2 = -mu^2 H^2 gives Bunch-Davies index nu = sqrt(9/4 + mu^2), and the
    exact de Sitter mode function at the coarse-graining boundary
    k = eps aH carries Q_nu(eps) = (pi/2) eps^3 |H^(1)_nu(eps)|^2.
    At mu^2 = 3, nu = sqrt(21)/2 and sqrt(Q(1)) = 2.805106.
    """
    nu = np.sqrt(2.25 + mu2)
    return np.sqrt(np.pi / 2.0 * epsilon**3 * np.abs(hankel1(nu, epsilon)) ** 2)


def seeded_moments(s, sigma_z, grid_size=12001):
    """First-passage moments averaged over the Born-seeded initial amplitude.

    Solves (s^2/2) T'' + s z T' = -1 and (s^2/2) M2'' + s z M2' = -2T on
    [-1, 1] with absorbing boundaries (z = X/X_c), then averages over the
    initial Gaussian z0 ~ N(0, sigma_z^2), assigning zero waiting time to the
    seed mass already outside the classical boundary.
    """
    z = np.linspace(-1.0, 1.0, grid_size)
    step = z[1] - z[0]
    interior = z[1:-1]

    diffusion = 0.5 * s**2 / step**2
    drift = s * interior / (2.0 * step)
    operator = diags(
        [(diffusion - drift)[1:], np.full(grid_size - 2, -2.0 * diffusion),
         (diffusion + drift)[:-1]],
        [-1, 0, 1], format="csc",
    )

    mean = np.zeros(grid_size)
    mean[1:-1] = spsolve(operator, np.full(grid_size - 2, -1.0))
    second = np.zeros(grid_size)
    second[1:-1] = spsolve(operator, -2.0 * mean[1:-1])

    pdf = norm.pdf(z, scale=sigma_z)
    n_mean = float(np.trapezoid(mean * pdf, z))
    n_second = float(np.trapezoid(second * pdf, z))
    outside = float(2.0 * norm.sf(1.0, scale=sigma_z))
    return {
        "N_stochastic_mean": n_mean,
        "N_stochastic_std": float(np.sqrt(max(n_second - n_mean**2, 0.0))),
        "P_seed_outside": outside,
    }


def first_passage_moments(s, grid_size=12001):
    """Mean and standard deviation of the exit time in e-folds."""
    boundary = 1.0 / np.sqrt(s)
    z = np.linspace(-boundary, boundary, grid_size)
    step = z[1] - z[0]
    interior = z[1:-1]

    lower = 0.5 / step**2 - interior / (2.0 * step)
    diagonal = np.full(grid_size - 2, -1.0 / step**2)
    upper = 0.5 / step**2 + interior / (2.0 * step)
    operator = diags([lower[1:], diagonal, upper[:-1]], [-1, 0, 1], format="csc")

    mean = np.zeros(grid_size)
    mean[1:-1] = spsolve(operator, np.full(grid_size - 2, -1.0 / s))

    second = np.zeros(grid_size)
    second[1:-1] = spsolve(operator, -2.0 * mean[1:-1] / s)

    center = grid_size // 2
    variance = second[center] - mean[center] ** 2
    return float(mean[center]), float(np.sqrt(variance))


def horizon_hubble(model, mass_scale):
    """H_H / M_Pl at the frozen horizon."""
    return mass_scale * np.sqrt(model.potential_shape(model.q) / 3.0)


def classical_efolds(model, mass_scale, s=None, epsilon=1.0, tachyonic=True,
                     **kwargs):
    """E-folds from the classicality boundary X_c to the end of inflation.

    Unlike background.prepare, the starting point here is physical rather than
    an arbitrary offset: the trajectory begins where the classical drift first
    exceeds the quantum kick. The kick is the tachyonic sigma_eps =
    (H/2pi) sqrt(Q_nu(eps)), not the massless H/2pi; the classical count MUST
    use the same threshold as the stochastic phase or the total double-counts
    ln sqrt(Q)/s = 1.303 e-folds of trajectory.
    """
    s = model.exit_exponent if s is None else s
    hubble = horizon_hubble(model, mass_scale)
    kick = hubble / (2.0 * np.pi)
    if tachyonic:
        kick *= tachyonic_kick_factor(epsilon, mu2=model.mu_squared)
    classical_boundary = kick / s

    _, gp, gpp, _, _ = model.geometry(model.q)
    dphi_dx = np.sqrt(1.5) * gpp / gp
    x0 = model.q - classical_boundary / dphi_dx

    solution = solve_ivp(
        bg.background_rhs(model),
        (0.0, 200.0),
        (x0, -model.dlogu_dphi(x0)),
        events=bg.end_of_inflation,
        rtol=kwargs.pop("rtol", 1.0e-11),
        atol=kwargs.pop("atol", 1.0e-13),
        max_step=kwargs.pop("max_step", config.BG_MAX_STEP),
    )
    if not solution.t_events[0].size:
        raise RuntimeError(f"Inflation did not end for p={model.p}")
    return float(solution.t_events[0][0]), float(classical_boundary)


def duration(model, mass_scale, grid_size=12001, epsilon=1.0,
             born_seed_over_h=None):
    """Full causal-patch duration: Born-seeded tachyonic first passage plus
    the classical roll from the SAME threshold.

    born_seed_over_h is sigma_C / H_H, the state-derived width of the real
    growing coefficient from the local classicalizing thimble (theory doc
    6.3/6.5); it is a property of the mu^2 = 3 horizon, not of p.
    """
    s = model.exit_exponent
    hubble = horizon_hubble(model, mass_scale)
    seed = (config.BORN_SEED_SIGMA_OVER_H if born_seed_over_h is None
            else born_seed_over_h)

    kick = hubble / (2.0 * np.pi) * tachyonic_kick_factor(
        epsilon, mu2=model.mu_squared
    )
    threshold = kick / s                       # X_c in M_Pl units
    sigma_z = seed * hubble / threshold        # seed width in units of X_c

    seeded = seeded_moments(s, sigma_z, grid_size=grid_size)
    sharp_mean, sharp_std = first_passage_moments(s, grid_size=grid_size)
    n_classical, boundary = classical_efolds(
        model, mass_scale, s=s, epsilon=epsilon
    )
    return {
        "H_horizon_over_Mpl": float(hubble),
        "exit_exponent_s": float(s),
        "epsilon_coarse_graining": epsilon,
        "tachyonic_kick_factor": float(tachyonic_kick_factor(epsilon)),
        "classicality_boundary_over_Mpl": boundary,
        "seed_sigma_z": float(sigma_z),
        **seeded,
        "N_stochastic_sharp_mean": sharp_mean,
        "N_stochastic_sharp_std": sharp_std,
        "N_classical": n_classical,
        "N_total_mean": n_classical + seeded["N_stochastic_mean"],
        "N_total_std": seeded["N_stochastic_std"],
    }
