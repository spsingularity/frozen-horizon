"""Is the "4" in the two axioms the spacetime dimension, or a d=4 coincidence?

Both axioms reduce, at d = 4, to a horizon condition equal to 4:

    trace sum rule   <=>  S'(1) = -4
    unit gap mu^2=3  <=>  R f_RR / f_R = 4

If 4 is really d, both must track d in the d-dimensional construction. This
builds that construction and tests it.

d-dimensional ingredients (all derived, then checked numerically here):

  de Sitter condition      R f_R - (d/2) f = 0
  trace-flow slope         S = [2/(d-2)] [(d/2)g - x g'] / x    (S(0)=1)
  scale-invariant power    f ~ R^(d/2)   (the homogeneous solution; it has
                           (d/2)g - xg' = 0 identically, so it never touches S)
  scalaron mass            m^2 = [(d-2) f_R/2 - R f_RR] / [(d-1) f_RR]
  de Sitter radius         R = d(d-1) H^2
  S^d scalar eigenvalues   lambda_L = [L(L+d-1) - mu^2] H^2
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class DimensionalModel:
    """g(x) in d dimensions for wall S(z) = (1-z)(1+alpha z), z = (x/q)^p.

    Solving x g' - (d/2) g = -[(d-2)/2] x S(z) term by term gives
        g = x + c_h x^(d/2) + B x^(p+1) + C x^(2p+1)
    with c_h the free homogeneous (scale-invariant) coefficient.
    """

    def __init__(self, p, d, q, alpha=3.0, c_homogeneous=1.0 / 6.0):
        self.p, self.d, self.q, self.alpha = p, d, q, alpha
        self.beta = alpha - 1.0
        self.c_h = c_homogeneous
        # Written with y = x/q so the large powers never leave [0, 1].
        self.b = -(d - 2.0) * self.beta / 2.0 / (p + 1.0 - d / 2.0)
        self.c = (d - 2.0) * alpha / 2.0 / (2 * p + 1.0 - d / 2.0)

    def _y(self, x, power):
        return np.exp(power * np.log(x / self.q))

    def g(self, x):
        p, q = self.p, self.q
        return (x + self.c_h * x ** (self.d / 2.0)
                + q * (self.b * self._y(x, p + 1.0)
                       + self.c * self._y(x, 2 * p + 1.0)))

    def gp(self, x):
        p = self.p
        return (1.0 + self.c_h * (self.d / 2.0) * x ** (self.d / 2.0 - 1.0)
                + self.b * (p + 1.0) * self._y(x, p)
                + self.c * (2 * p + 1.0) * self._y(x, 2 * p))

    def gpp(self, x):
        d, p, q = self.d, self.p, self.q
        return (self.c_h * (d / 2.0) * (d / 2.0 - 1.0) * x ** (d / 2.0 - 2.0)
                + self.b * (p + 1.0) * p * self._y(x, p - 1.0) / q
                + self.c * (2 * p + 1.0) * (2 * p) * self._y(x, 2 * p - 1.0) / q)

    def trace_slope(self, x):
        return 2.0 / (self.d - 2.0) * ((self.d / 2.0) * self.g(x) - x * self.gp(x)) / x

    def mu_squared(self, x):
        """|m^2|/H^2 at a constant-curvature point of curvature x."""
        d = self.d
        m2_over_H2 = d * (d - 2.0) * self.gp(x) / (2.0 * x * self.gpp(x)) - d
        return -m2_over_H2


def solve_q(p, d, alpha=3.0, c_h=1.0 / 6.0):
    """Fix q by the unit-gap condition lambda_1 = H^2, i.e. mu^2 = d - 1."""
    target = d - 1.0

    def residual(q):
        return DimensionalModel(p, d, q, alpha, c_h).mu_squared(q) - target

    return brentq(residual, 1.0001, 1.0e5, xtol=1e-12, rtol=1e-15)


def main():
    p, alpha = 63, 3.0
    print("unit gap lambda_1 = H^2 on S^d:  lambda_1 = [d - mu^2] H^2 = H^2")
    print("  =>  mu^2 = d - 1     (3 at d=4, matching the handout)\n")

    print(f"{'d':>3} {'q':>13} {'mu^2':>7} {'R f_RR/f_R':>12} {'d(d-2)/2':>10} "
          f"{'S_(1)':>9} {'-(alpha+1)':>11} {'f_R(q)':>11} {'4p/(d-1)':>10}")
    for d in (4, 5, 6, 8, 10):
        q = solve_q(p, d, alpha)
        model = DimensionalModel(p, d, q, alpha)
        gp, gpp = model.gp(q), model.gpp(q)

        ratio = q * gpp / gp
        step = q * 1e-7
        ds_dx = (model.trace_slope(q + step) - model.trace_slope(q - step)) / (2 * step)
        s_prime = ds_dx * q / p          # dz/dx = p/q at x = q

        print(f"{d:>3} {q:>13.6f} {model.mu_squared(q):>7.4f} {ratio:>12.6f} "
              f"{d*(d-2)/2:>10.4f} {s_prime:>9.5f} {-(alpha+1):>11.1f} "
              f"{gp:>11.5f} {4*p/(d-1):>10.5f}")

    print("\n--- verdict ---")
    print("unit gap      : R f_RR/f_R = d(d-2)/2   -> genuinely d-dependent")
    print("               equals d only when d(d-2)/2 = d, i.e. d = 4 UNIQUELY")
    print("sum rule      : S'(1) = -(alpha+1) = -4 in EVERY d")
    print("               the sum rule constrains S(z) alone and never sees d")
    print("combined      : f_R(R_H) = p(alpha+1)/(d-1) = 4p/(d-1)")
    print("\nSo the two 4s have different origins and coincide only at d = 4.")


if __name__ == "__main__":
    main()
