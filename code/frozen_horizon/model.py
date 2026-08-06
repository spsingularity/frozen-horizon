"""Metric f(R) model f(R) = M^2 g(x), x = R / M^2, parameterized by the integer p.

The curvature function is

    g(x) = x + x^2/6 - [2q/(p-1)] y^(p+1) + [3q/(2p-1)] y^(2p+1),   y = x/q,

with q fixed by the unit-gap bootstrap and the coefficients 2 and 3 fixed by the
trace-flow sum rule. p is the only input.
"""

import numpy as np

from . import bootstrap


class FrozenHorizonModel:
    """Curvature function and its derivatives for a given integer p."""

    def __init__(self, p=67, alpha=None):
        """alpha is the wall coefficient in S(z) = (1-z)(1 + alpha z).

        It defaults to 3, the value fixed by the sum rule in the dz measure.
        That measure is a choice: d ln R (the RG-natural one) gives alpha = 2,
        and alpha = 0 is the single-operator linear wall. Exposing it makes the
        choice testable rather than postulated.
        """
        self.p = int(p)
        self.alpha = bootstrap.WALL_COEFFICIENT if alpha is None else float(alpha)
        self.beta = self.alpha - 1.0
        self.q = bootstrap.q_of_p(self.p, self.alpha)
        self.n_power = 2 * self.p + 1                 # 135 at p = 67
        self.m_power = self.p + 1                     # 68 at p = 67

    def __repr__(self):
        return (f"FrozenHorizonModel(p={self.p}, alpha={self.alpha:g}, "
                f"q={self.q:.12f})")

    def power_ratio(self, x, exponent):
        """(x/q)^exponent, evaluated through logs to survive the large powers."""
        return np.exp(exponent * np.log(x / self.q))

    def geometry(self, x):
        """Return g, g', g'', Q = x g' - g, and the trace slope B = 2g - x g'."""
        dn = self.power_ratio(x, self.n_power - 1.0)
        dm = self.power_ratio(x, self.m_power - 1.0)
        high = x * (
            (1.0 + self.beta) * dn / (self.n_power - 2.0)
            - self.beta * dm / (self.m_power - 2.0)
        )
        g = x + x * x / 6.0 + high
        gp = 1.0 + x / 3.0 + (
            (1.0 + self.beta) * self.n_power * dn / (self.n_power - 2.0)
            - self.beta * self.m_power * dm / (self.m_power - 2.0)
        )
        gpp = 1.0 / 3.0 + (
            (1.0 + self.beta) * self.n_power * (self.n_power - 1.0) * dn
            / ((self.n_power - 2.0) * x)
            - self.beta * self.m_power * (self.m_power - 1.0) * dm
            / ((self.m_power - 2.0) * x)
        )
        qpot = x * gp - g
        slope = 2.0 * g - x * gp
        return g, gp, gpp, qpot, slope

    # --- Einstein-frame quantities ----------------------------------------

    def phi_of_x(self, x):
        """Canonical scalaron phi / M_Pl = sqrt(3/2) ln f_R."""
        return np.sqrt(1.5) * np.log(self.geometry(x)[1])

    def dlogu_dphi(self, x):
        """d ln U / d(phi/M_Pl)."""
        _, _, _, qpot, slope = self.geometry(x)
        return np.sqrt(2.0 / 3.0) * slope / qpot

    def potential_shape(self, x):
        """U / (M_Pl^2 M^2)."""
        _, gp, _, qpot, _ = self.geometry(x)
        return 0.5 * qpot / (gp * gp)

    # --- Bootstrap diagnostics --------------------------------------------

    @property
    def horizon_slope(self):
        """F_H = g'(q)."""
        return self.geometry(self.q)[1]

    @property
    def mu_squared(self):
        """|m_H^2| / H_H^2 at the horizon; the unit gap demands d - 1 = 3.

        Computed geometrically from m^2/H^2 = 4 f_R/(R f_RR) - 4, so it stays
        correct for any alpha. (The handout's 16p/(F_H + 4p) is the alpha = 3
        special case.)
        """
        _, gp, gpp, _, _ = self.geometry(self.q)
        return 4.0 - 4.0 * gp / (self.q * gpp)

    @property
    def exit_exponent(self):
        """s in delta phi ~ exp(sN)."""
        return bootstrap.exit_exponent(self.mu_squared)

    def horizon_residual(self, x=None):
        """R f_R - 2f in units of M^2, i.e. -(2g - x g'). Zero at the horizon."""
        x = self.q if x is None else x
        return -self.geometry(x)[4]

    def trace_slope_check(self, x):
        """Compare (2g - x g')/x against the postulated wall S(z), z = (x/q)^p."""
        _, _, _, _, slope = self.geometry(x)
        z = self.power_ratio(x, self.p)
        return slope / x, bootstrap.trace_flow(z, self.alpha)
