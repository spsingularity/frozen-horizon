"""Measure the scalaron equation of state instead of assuming w = 0.

w = 0 is the standard assumption for coherent oscillations in a quadratic
minimum, and it is what separates p = 63 from p = 64 (w = 0 -> 0.05 moves N_*
by +0.84 e-folds against a 0.99 spacing). So it needs measuring, not asserting.

The Einstein-frame potential of R + R^2/(6M^2) is

    U(phi) = (3/4) M^2 M_Pl^2 [1 - exp(-sqrt(2/3) phi/M_Pl)]^2

which is quadratic only near the minimum. Inflation ends at phi ~ 0.94 M_Pl,
where the anharmonic correction is O(1), so the first oscillations genuinely
deviate from w = 0. The amplitude then falls as a^{-3/2} and the deviation dies
with it. What matters for N_* is the *accumulated* difference in e-folds, which
converges -- this integrates it directly.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ROOT_23 = np.sqrt(2.0 / 3.0)


def potential(phi):
    """U / (M^2 M_Pl^2)."""
    return 0.75 * (1.0 - np.exp(-ROOT_23 * phi)) ** 2


def potential_gradient(phi):
    """dU/dphi / (M^2 M_Pl)."""
    exponential = np.exp(-ROOT_23 * phi)
    return 1.5 * (1.0 - exponential) * exponential * ROOT_23


def rhs(tau, state):
    """d/d(Mt) of [phi, dphi/d(Mt), ln a], in reduced-Planck units."""
    phi, velocity, _ = state
    density = 0.5 * velocity**2 + potential(phi)
    hubble = np.sqrt(density / 3.0)
    return [velocity, -3.0 * hubble * velocity - potential_gradient(phi), hubble]


def main(tau_max=4000.0):
    # Start at the exact end of inflation: epsilon = 1 gives 3H^2 = H^2 + U,
    # so H^2 = U/2 and |dphi/dt| = sqrt(2) H = sqrt(U).
    phi_end = 0.940178
    u_end = potential(phi_end)
    velocity_end = -np.sqrt(u_end)
    density_end = 0.5 * velocity_end**2 + u_end

    solution = solve_ivp(
        rhs, (0.0, tau_max), [phi_end, velocity_end, 0.0],
        rtol=1e-10, atol=1e-12, dense_output=True, max_step=0.5,
    )
    phi, velocity, ln_a = solution.y
    kinetic = 0.5 * velocity**2
    density = kinetic + potential(phi)
    pressure = kinetic - potential(phi)

    # Instantaneous w oscillates between -1 and +1; the meaningful quantity is
    # the running average that controls how fast rho falls with a.
    print(f"{'ln a':>8} {'rho/rho_end':>13} {'w_eff (running)':>16} {'phi_amp':>10}")
    marks = np.linspace(0.5, ln_a[-1], 10)
    for mark in marks:
        index = int(np.argmin(np.abs(ln_a - mark)))
        # rho ~ a^{-3(1+w)}  =>  w_eff = -1 - ln(rho/rho_end)/(3 ln a)
        w_eff = -1.0 - np.log(density[index] / density_end) / (3.0 * ln_a[index])
        window = slice(max(index - 400, 0), index + 1)
        print(f"{ln_a[index]:>8.3f} {density[index]/density_end:>13.4e} "
              f"{w_eff:>16.6f} {np.max(np.abs(phi[window])):>10.4f}")

    # Accumulated e-fold error against an exact w = 0 history over the same
    # density range. This is the only thing that propagates into N_*.
    final = -1
    ln_rho_drop = np.log(density_end / density[final])
    n_actual = ln_a[final]
    n_w0 = ln_rho_drop / 3.0
    print(f"\nover ln(rho_end/rho) = {ln_rho_drop:.4f}:")
    print(f"  actual e-folds      N_re = {n_actual:.6f}")
    print(f"  exact w = 0 history N_re = {n_w0:.6f}")
    print(f"  accumulated offset       = {n_actual - n_w0:+.6f} e-folds")
    print(f"  => shift in N_*          = {-(n_actual - n_w0):+.6f} e-folds")
    print(f"\n  (adjacent p are separated by 0.9933 e-folds)")

    # Is the offset still growing, or has it converged?
    half = len(ln_a) // 2
    offset_half = ln_a[half] - np.log(density_end / density[half]) / 3.0
    print(f"  offset at half the integration: {offset_half:+.6f} "
          f"-> {'CONVERGED' if abs(offset_half - (n_actual - n_w0)) < 0.01 else 'still growing'}")


if __name__ == "__main__":
    main()
