"""Exact Einstein-frame background trajectory, integrated without slow roll."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from . import config


class Background:
    """Interpolated background trajectory with the pivot normalization applied.

    Scale factor convention: a = exp(N - pivot_N), i.e. a = 1 at pivot crossing.
    """

    def __init__(self, model, N, x, velocity, end_N, n_star=None,
                 amplitude_correction=1.0):
        self.model = model
        self.N = N
        self._x_raw = x
        self._v_raw = velocity
        self.start_N = float(N[0])
        self.end_N = float(end_N)
        self.n_star = config.N_STAR if n_star is None else float(n_star)
        self.pivot_N = self.end_N - self.n_star
        self.amplitude_correction = float(amplitude_correction)

        self._x = CubicSpline(N, x)
        self._v = CubicSpline(N, velocity)

        # Normalize M by matching the observed scalar amplitude at the pivot in
        # the slow-roll approximation. observables.py refines this using the
        # exact mode amplitude; amplitude_correction carries that refinement
        # back in so nothing downstream has to hard-code the result.
        epsilon_pivot = 0.5 * float(self._v(self.pivot_N)) ** 2
        shape_pivot = model.potential_shape(float(self._x(self.pivot_N)))
        self.mass_scale = self.amplitude_correction * np.sqrt(
            config.A_S_OBS * 24.0 * np.pi**2 * epsilon_pivot / shape_pivot
        )
        self.H_pivot = self.quantities(self.pivot_N)[3]

    def rebase(self, n_star, amplitude_correction=None):
        """Return the same trajectory with the pivot placed at a different N_*.

        N_* shifts only where the pivot sits, never the trajectory itself, so
        this reuses the integration and re-derives the pivot-dependent
        normalization. Cheap enough to iterate on.
        """
        return Background(
            self.model,
            self.N,
            self._x_raw,
            self._v_raw,
            self.end_N,
            n_star=n_star,
            amplitude_correction=(
                self.amplitude_correction
                if amplitude_correction is None
                else amplitude_correction
            ),
        )

    def energy_densities(self):
        """V_* , rho_* and rho_end in reduced-Planck units (M_Pl = 1).

        rho = 3U/(3 - epsilon), so rho_end = (3/2) U_end at epsilon = 1.
        """
        x_pivot, _, eps_pivot, _, _ = self.quantities(self.pivot_N)
        m2 = self.mass_scale**2
        v_pivot = m2 * self.model.potential_shape(float(x_pivot))
        v_end = m2 * self.model.potential_shape(float(self._x_raw[-1]))
        return {
            "V_star": float(v_pivot),
            "rho_star": float(3.0 * v_pivot / (3.0 - eps_pivot)),
            "rho_end": float(1.5 * v_end),
            "epsilon_star": float(eps_pivot),
            "M_over_Mpl": float(self.mass_scale),
        }

    @property
    def total_efolds(self):
        """E-folds spanned by this integration.

        Only meaningful relative to the start point. For the mode-solver
        background the start is an arbitrary offset below the horizon, so this
        is NOT the physical duration; see stochastic.classical_efolds.
        """
        return self.end_N - self.start_N

    def quantities(self, at_N):
        """Return x, dphi/dN, epsilon, H/M_Pl, and d(dphi/dN)/dN at e-fold at_N."""
        xx = self._x(at_N)
        vv = self._v(at_N)
        eps = 0.5 * vv * vv
        shape = self.model.potential_shape(xx)
        hubble = self.mass_scale * np.sqrt(shape / (3.0 - eps))
        dv = -(3.0 - eps) * (vv + self.model.dlogu_dphi(xx))
        return xx, vv, eps, hubble, dv

    def k_over_aH(self, at_N, k):
        """k / (aH) with a = exp(N - pivot_N)."""
        return k / (np.exp(at_N - self.pivot_N) * self.quantities(at_N)[3])

    def stability(self):
        """Minima of f_R = g' and M^2 f_RR = g'' along the trajectory.

        Positivity of both is what makes the scalaron representation legitimate
        and the negative horizon mass an exit instability rather than a ghost.
        """
        x = self._x(self.N)
        _, gp, gpp, _, _ = self.model.geometry(x)
        return {"min_f_R": float(np.min(gp)), "min_M2_f_RR": float(np.min(gpp))}


def background_rhs(model):
    """Return the (N, [x, dphi/dN]) right-hand side for this model."""

    def rhs(efold, state):
        x, velocity = state
        _, gp, gpp, _, _ = model.geometry(x)
        epsilon = 0.5 * velocity * velocity
        dx_dN = velocity * gp / (np.sqrt(1.5) * gpp)
        dv_dN = -(3.0 - epsilon) * (velocity + model.dlogu_dphi(x))
        return (dx_dN, dv_dN)

    return rhs


def end_of_inflation(efold, state):
    """Event: epsilon = 1."""
    return 0.5 * state[1] ** 2 - 1.0


end_of_inflation.terminal = True
end_of_inflation.direction = 1


def integrate(model, x0, rtol=None, atol=None, max_step=None):
    """Integrate from x0 on the slow-roll attractor until epsilon = 1."""
    v0 = -model.dlogu_dphi(x0)
    sol = solve_ivp(
        background_rhs(model),
        (0.0, config.BG_N_MAX),
        (x0, v0),
        events=end_of_inflation,
        rtol=config.BG_RTOL if rtol is None else rtol,
        atol=config.BG_ATOL if atol is None else atol,
        max_step=config.BG_MAX_STEP if max_step is None else max_step,
        dense_output=False,
    )
    if not sol.t_events[0].size:
        raise RuntimeError(f"Inflation did not end for p={model.p}")
    end_N = float(sol.t_events[0][0])
    keep = sol.t <= end_N
    return sol.t[keep], sol.y[0, keep], sol.y[1, keep], end_N


def prepare(model, horizon_offset=None, n_star=None, amplitude_correction=1.0,
            **kwargs):
    """Build the mode-solver background, starting just below the horizon."""
    offset = config.HORIZON_OFFSET if horizon_offset is None else horizon_offset
    x0 = model.q * (1.0 - offset)
    N, x, velocity, end_N = integrate(model, x0, **kwargs)
    return Background(
        model, N, x, velocity, end_N,
        n_star=n_star, amplitude_correction=amplitude_correction,
    )
