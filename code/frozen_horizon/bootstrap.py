"""Trace-flow wall and Euclidean unit-gap algebra.

Everything here is *computed* from the integer p. The handout states several of
these relations as results; this module derives them so that the tests can fail
if they stop holding.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def solve_wall_coefficient():
    """Return alpha in S(z) = (1 - z)(1 + alpha z), fixed by the sum rule.

    The bootstrap sum rule is int_0^1 [S(z) - 1] dz = 0. Solving it numerically
    rather than writing 3 makes the axiom executable: S - 1 = (alpha - 1) z -
    alpha z^2 integrates to (alpha - 1)/2 - alpha/3, which vanishes at alpha = 3.
    """

    def sum_rule(alpha):
        return quad(lambda z: (1.0 - z) * (1.0 + alpha * z) - 1.0, 0.0, 1.0)[0]

    return brentq(sum_rule, 0.5, 10.0, xtol=1.0e-14, rtol=1.0e-15)


WALL_COEFFICIENT = solve_wall_coefficient()  # == 3


def wall_coefficient_in_measure(measure, p=None):
    """alpha fixed by int (S-1) d mu = 0 in a declared measure.

    With S - 1 = (alpha-1) z - alpha z^2 and z = (x/q)^p, so x = q z^{1/p},
    three of the four measures used in the paper integrate in closed form:

        dz      : (a-1)/2 - a/3 = 0                       -> a = 3
        d ln R  : weight dz/(pz);  (a-1) - a/2 = 0        -> a = 2, every p
        dR      : weight z^{1/p-1};
                  (a-1)p/(1+p) = a p/(1+2p)               -> a = 2 + 1/p

    Only the canonical field-space measure d phi ∝ d ln f_R has no closed
    form, because the weight g''/g' depends on alpha through the wall itself;
    it is a fixed point, solved by canonical_wall_coefficient().

    The dz rule is not coordinate invariant: z embeds p in its own definition.
    d ln R is the renormalization-group-natural invariant choice, and is the
    only one of the three whose alpha does not depend on p at all.
    """
    if measure == "dz":
        return 3.0
    if measure == "dlnR":
        return 2.0
    if measure == "dR":
        if p is None:
            raise ValueError("the dR measure needs p: alpha = 2 + 1/p")
        return 2.0 + 1.0 / float(p)
    raise ValueError(f"unknown measure {measure!r}; "
                     "use 'dz', 'dlnR', 'dR', or canonical_wall_coefficient")


def canonical_wall_coefficient(p, bracket=(2.0, 4.0), tol=1.0e-12):
    """alpha for the canonical scalaron measure d phi = sqrt(3/2) M_Pl d ln f_R.

    This is the paper's primary branch. Unlike the three measures above it has
    no closed form and is not even a direct root: the weight

        d ln f_R / dz = (g''/g') (dx/dz),     dx/dz = (q/p) z^{1/p - 1},

    is built from g, which is itself built from alpha. Solving therefore means
    finding a fixed point of the wall against its own field-space measure.

    Returns 2.809064 at p = 66, the value the pipeline is run at. alpha falls
    slowly with p, from 2.8103 at p = 62 to a limit near 2.7914.
    """
    from .model import FrozenHorizonModel      # local: model imports bootstrap

    def residual(alpha):
        model = FrozenHorizonModel(p, alpha=alpha)

        def integrand(z):
            x = model.q * z ** (1.0 / p)
            _, gp, gpp, _, _ = model.geometry(x)
            return (((alpha - 1.0) * z - alpha * z * z)
                    * (gpp / gp) * (model.q / p) * z ** (1.0 / p - 1.0))

        return quad(integrand, 1.0e-12, 1.0, limit=200)[0]

    return brentq(residual, *bracket, xtol=tol, rtol=1.0e-14)


def trace_flow(z, alpha=WALL_COEFFICIENT):
    """Normalized trace-flow slope S(z) = (2g - x g') / x."""
    return (1.0 - z) * (1.0 + alpha * z)


def c_of_p(p, alpha=WALL_COEFFICIENT):
    """Horizon value of g' - q/3, for wall S(z) = (1-z)(1 + alpha z).

    C(p) = 1 + alpha(2p+1)/(2p-1) - beta(p+1)/(p-1) with beta = alpha - 1.
    At alpha = 3 this is the handout's 1 - 2(p+1)/(p-1) + 3(2p+1)/(2p-1).
    """
    beta = alpha - 1.0
    return (1.0
            + alpha * (2.0 * p + 1.0) / (2.0 * p - 1.0)
            - beta * (p + 1.0) / (p - 1.0))


def q_of_p(p, alpha=WALL_COEFFICIENT):
    """Horizon curvature q = R_H/M^2 from the unit gap, in d = 4.

    The unit gap mu^2 = d - 1 = 3 forces f_R(R_H) = p(alpha+1)/(d-1). With
    f_R(R_H) = q/3 + C(p, alpha) this gives q = p(alpha+1) - 3 C(p, alpha),
    which reduces to q = 4p - 3C(p) at alpha = 3.
    """
    return p * (alpha + 1.0) - 3.0 * c_of_p(p, alpha)


def mu_squared(p, horizon_slope):
    """Euclidean scalaron mass ratio |m_H^2| / H_H^2 from F_H = g'(q)."""
    return 16.0 * p / (horizon_slope + 4.0 * p)


def exit_exponent(mu2=3.0):
    """Growth rate s of the unstable mode, delta phi ~ exp(sN).

    s solves s(s + 3) = mu^2, so s = (sqrt(9 + 4 mu^2) - 3) / 2, which is
    (sqrt(21) - 3)/2 = 0.7913 at the unit-gap value mu^2 = 3.
    """
    return (np.sqrt(9.0 + 4.0 * mu2) - 3.0) / 2.0


def euclidean_eigenvalues(max_L=5, mu2=3.0):
    """Scalar eigenvalues lambda_L / H_H^2 = L(L+3) - mu^2 on the four-sphere."""
    L = np.arange(0, max_L + 1)
    return L, L * (L + 3.0) - mu2
