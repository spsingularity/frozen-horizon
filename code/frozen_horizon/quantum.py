"""Quantum state of the frozen horizon: fluctuation kernel, thimble, shells.

Everything here concerns the compact Euclidean saddle at the frozen horizon
and its continuation to Lorentzian histories. The de Sitter background is
exact (the horizon is a constant-curvature solution), so these quantities
depend on the model only through mu^2, which the unit gap fixes to d - 1 = 3.
They are therefore independent of p and of the wall coefficient alpha.

Conventions: u = H_H tau on the Euclidean four-sphere (equator at pi/2),
v = H_H t after continuation through the equator, and all rates are in units
of H_H.

A numerical note that matters: the regular Euclidean solution behaves as
f_n ~ u^n near the pole, which underflows the solver's absolute tolerance for
n >~ 6 if imposed literally. All initial conditions here are normalized to
f_n(u_0) = 1, which is equivalent by linearity and keeps amplitudes O(1).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import hankel1

MU2 = 3.0                       # unit gap: mu^2 = d - 1
EQUATOR = np.pi / 2.0


def euclidean_mode(n, mu2=MU2, u0=1.0e-6, rtol=1.0e-12, atol=1.0e-14):
    """Regular Euclidean scalar mode on the half four-sphere.

    Solves f'' + 3 cot(u) f' - [n(n+2)/sin^2 u - mu^2] f = 0 from the pole to
    the equator with f ~ u^n regularity, and returns (f'/f, solution) at the
    equator. The logarithmic derivative is the only scale-free content.
    """
    y0 = [1.0, -mu2 * u0 / 4.0] if n == 0 else [1.0, n / u0]

    def rhs(u, y):
        return [y[1], -3.0 / np.tan(u) * y[1]
                + (n * (n + 2) / np.sin(u) ** 2 - mu2) * y[0]]

    solution = solve_ivp(rhs, (u0, EQUATOR), y0, rtol=rtol, atol=atol)
    return float(solution.y[1, -1] / solution.y[0, -1])


def fluctuation_spectrum(max_n=8, mu2=MU2):
    """Boundary kernel K_n H^2 = 2 pi^2 f_n'/f_n at the equator.

    K_0 < 0 is the single unstable direction; K_n > 0 for n >= 1 means every
    inhomogeneous mode is Gaussian suppressed.
    """
    ratios = {n: euclidean_mode(n, mu2=mu2) for n in range(max_n + 1)}
    return ratios, {n: 2.0 * np.pi**2 * r for n, r in ratios.items()}


def eigenvalues(max_L=6, mu2=MU2):
    """lambda_L / H^2 = L(L+3) - mu^2 on the four-sphere."""
    L = np.arange(0, max_L + 1)
    return L, L * (L + 3.0) - mu2


def lorentzian_mode(n, v_end, ratio=None, mu2=MU2,
                    rtol=1.0e-12, atol=1.0e-14):
    """Continuation through the equator: F_n(0)=1, F_n'(0)= i f_n'/f_n.

    Obtained from the Euclidean equation by u = pi/2 + i v, giving
    F'' + 3 tanh(v) F' + [n(n+2)/cosh^2 v - mu^2] F = 0.
    """
    ratio = euclidean_mode(n, mu2=mu2) if ratio is None else ratio

    def rhs(v, y):
        return [y[1], -3.0 * np.tanh(v) * y[1]
                - (n * (n + 2) / np.cosh(v) ** 2 - mu2) * y[0]]

    solution = solve_ivp(rhs, (0.0, v_end), [1.0 + 0j, 1j * ratio],
                         rtol=rtol, atol=atol)
    return solution.y[0, -1]


def classicalizing_thimble(mu2=MU2, v_end=14.0):
    """Contour phase that makes the growing Lorentzian branch real.

    Writing the late growing coefficient as C_g = (c_1 + i r_E c_2) A for an
    equatorial amplitude A, reality of the growing history fixes
    A = e^{i theta} Y with theta = -arg(c_1 + i r_E c_2). Normalizability
    then requires cos(2 theta) < 0, which is satisfied.
    """
    s = (np.sqrt(9.0 + 4.0 * mu2) - 3.0) / 2.0
    ratio = euclidean_mode(0, mu2=mu2)

    def growing(initial):
        solution = solve_ivp(
            lambda v, y: [y[1], -3.0 * np.tanh(v) * y[1] + mu2 * y[0]],
            (0.0, v_end), initial, rtol=1e-12, atol=1e-14)
        return solution.y[0, -1] / np.exp(s * v_end)

    c1 = float(growing([1.0, 0.0]))
    c2 = float(growing([0.0, 1.0]))
    amplitude = c1 + 1j * ratio * c2
    theta = -np.angle(amplitude)

    kernel = 2.0 * np.pi**2 * ratio                 # K_0 H^2 < 0
    suppression = abs(kernel) * abs(np.cos(2.0 * theta))
    # Re ln Psi = -(suppression/2H^2) Y^2, so the Born density |Psi|^2 carries
    # twice that exponent and the standard deviation is 1/sqrt(2 suppression).
    # Dropping this factor of two would inflate the exit seed by sqrt(2).
    sigma_y = 1.0 / np.sqrt(2.0 * suppression)
    return {
        "r_E": ratio, "c1": c1, "c2": c2,
        "theta_rad": float(theta), "theta_deg": float(np.degrees(theta)),
        "cos_2theta": float(np.cos(2.0 * theta)),
        "K0_H2": float(kernel),
        "sigma_Y_over_H": float(sigma_y),
        "sigma_Cg_over_H": float(abs(amplitude) * sigma_y),
        "kappa_C": float(suppression / abs(amplitude) ** 2),
        "exit_exponent_s": float(s),
    }


#: Normalizability threshold of the classicalizing Born measure, theta = 45 deg.
#: From theta = pi s / 2 (see theta_exact) this is s = 1/2, i.e. mu^2 = 7/4.
#: The one-negative-mode condition 0 < mu^2 < 4 is therefore necessary but NOT
#: sufficient: 7/4 < mu^2 < 4 is required. The unit gap mu^2 = d - 1 = 3 lies
#: inside this range.
MU2_NORMALIZABLE_MIN = 7.0 / 4.0


def growth_exponent(mu2=MU2):
    """Late-time Lorentzian growth exponent s, the positive root of s(s+3)=mu^2."""
    return 0.5 * (np.sqrt(9.0 + 4.0 * mu2) - 3.0)


def theta_exact(mu2=MU2):
    """Classicalizing contour angle in closed form: theta = pi s / 2.

    Both mode equations are hypergeometric. Substituting s = sinh(v) in the
    Lorentzian equation F'' + 3 tanh(v) F' - mu^2 F = 0 and then w = -s^2 gives

        w(1-w) F_ww + [1/2 - (5/2) w] F_w + (mu^2/4) F = 0,

    i.e. a = (s+3)/2, b = -s/2, c = 1/2 with s(s+3) = mu^2. The large-|w|
    connection coefficients of the even and odd equatorial solutions give

        c2/c1 = (1/2) G((s+3)/2) G((s+1)/2) / [ G((s+4)/2) G((s+2)/2) ].

    The Euclidean equation is the same one at x = cos(u): the Gegenbauer
    equation (1-x^2) f'' - 4 x f' + mu^2 f = 0, whose solution regular at the
    pole is C^{3/2}_s. Killing the singular (1-x^2)^{-1} branch fixes

        r_E = 2 G((s+4)/2) G((1-s)/2) / [ G((s+3)/2) G(-s/2) ] .

    In the product |r_E| c2/c1 every Gamma cancels pairwise, and the two
    reflection formulae G((1-s)/2)G((1+s)/2) = pi/cos(pi s/2) and
    |G(-s/2)|G(1+s/2) = pi/sin(pi s/2) collapse the rest to tan(pi s / 2).
    Hence theta = arctan(|r_E| c2/c1) = pi s / 2 identically, and the
    normalizability condition cos(2 theta) < 0 is simply s > 1/2.

    This is exact; classicalizing_thimble() integrates the same quantity
    numerically and the two agree to ~1e-11 degrees (see the tests).
    """
    return 0.5 * np.pi * growth_exponent(mu2)


def normalizability_window(mu2=MU2):
    """Whether a normalizable classicalizing contour exists at this mu^2."""
    s = growth_exponent(mu2)
    return {
        "mu2": mu2,
        "one_negative_mode": 0.0 < mu2 < 4.0,
        "normalizable": MU2_NORMALIZABLE_MIN < mu2 < 4.0,
        "threshold": MU2_NORMALIZABLE_MIN,
        "growth_exponent": float(s),
        "theta_deg": float(np.degrees(theta_exact(mu2))),
        "cos_2theta": float(np.cos(np.pi * s)),
    }


def shell_crossings(max_n=8):
    """Exact S^3 shell crossing times and the noise they inject.

    In closed slicing a(v) = cosh(v)/H, so aH_phys = sinh v and harmonic n
    crosses when sinh v_n = sqrt(n(n+2)); since cosh(arcsinh(x)) = sqrt(1+x^2),
    this gives the exact staircase N_n = ln(n+1).
    """
    rows = []
    previous = 0.0
    for n in range(1, max_n + 1):
        v_n = np.arcsinh(np.sqrt(n * (n + 2.0)))
        efolds = np.log(np.cosh(v_n))               # == ln(n+1) exactly
        ratio = euclidean_mode(n)
        amplitude = abs(lorentzian_mode(n, v_n, ratio=ratio))
        variance = (n + 1.0) ** 2 * amplitude**2 / (4.0 * np.pi**2 * ratio)
        rows.append({
            "n": n, "N_n": float(efolds), "ln_n_plus_1": float(np.log(n + 1)),
            "degeneracy": (n + 1) ** 2, "r_n": float(ratio),
            "F_at_crossing": float(amplitude),
            "delta_V_n": float(variance),
            "Q_effective": float(4.0 * np.pi**2 * variance
                                 / (efolds - previous)),
        })
        previous = efolds
    return rows


def continuum_kick(epsilon=1.0, mu2=MU2):
    """Q_nu(eps) = (pi/2) eps^3 |H^(1)_nu(eps)|^2, the tachyonic enhancement.

    The coarse-grained kick is sigma = (H/2pi) sqrt(Q_nu); the massless value
    Q = 1 is wrong at a tachyonic horizon. Q_nu(1) = 7.8686 at mu^2 = 3, which
    the discrete shell sequence approaches from above.
    """
    nu = np.sqrt(2.25 + mu2)
    return float(np.pi / 2.0 * epsilon**3 * np.abs(hankel1(nu, epsilon)) ** 2)
