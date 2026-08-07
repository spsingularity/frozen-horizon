#!/usr/bin/env python
"""Independent primordial mode solver for pipeline validation.

Deliberately different formulation from frozen_horizon/ (which integrates the
field perturbation in e-fold time with a 3 - eps + 2 dv/v friction term,
initialized with a leading-order adiabatic state at k/aH = 50):

  * Background: Einstein-frame scalar phi with potential U(phi) built by
    numerically inverting the closed-form map x -> phi = sqrt(3/2) ln g'(x)
    (PCHIP initial guess + Newton refinement); EOM written directly in terms
    of (phi, dphi/dN) and U, U_phi.  Units: M = M_Pl = 1.
  * Modes: Mukhanov-Sasaki equation in CONFORMAL TIME,
        v_k'' + (k^2 - z''/z) v_k = 0,   z = a * dphi/dN,
    with z''/z obtained numerically from a spline of the background
    (ln z spline -> conformal-time derivatives via exact chain rule),
    integrated in tau measured from the end of inflation (tau_end = 0)
    to keep double precision at late times.
  * Initial data: FULL Bunch-Davies solution v = e^{-ik tau}/sqrt(2k),
    v' = -ik v, imposed at k/(aH) = 100.
  * Pivot: k_pivot = (aH) at N_* = 51.6103 e-folds before eps = 1.

Model (p = 63, alpha = 3):
    g(x)  = x + x^2/6 - (2q/(p-1)) (x/q)^(p+1) + (3q/(2p-1)) (x/q)^(2p+1)
    q     = 4p - 3C(p),  C(p) = 1 - 2(p+1)/(p-1) + 3(2p+1)/(2p-1)
    phi   = sqrt(3/2) ln g'(x)
    U     = (x g' - g) / (2 g'^2)          [in units M^2 M_Pl^2]
    U_phi = (2g - x g') / (sqrt(6) g'^2)   [closed form via chain rule]
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator

T0 = time.time()

# ----------------------------------------------------------------------
# Model definition
# ----------------------------------------------------------------------
P = 63
CP = 1.0 - 2.0 * (P + 1) / (P - 1) + 3.0 * (2 * P + 1) / (2 * P - 1)
Q = 4.0 * P - 3.0 * CP
S6 = math.sqrt(6.0)
S32 = math.sqrt(1.5)
REFERENCE_RUN = "resolved_p63"

# N_* is read from the run this solver validates against, not hardcoded. The
# two must agree exactly: a 0.011 e-fold offset translates the notch in ln k,
# and since the comparison marginalizes only amplitude and tilt (not a shift),
# the mismatch lands straight in the residual. That is precisely what happened
# when the pipeline moved to the converged reheating fixed point and this
# literal did not -- the quoted agreement degraded from 4e-4 to 5.3e-3 while
# the script still reported success against its own stale self-check values.
_RUN_DIR = Path(__file__).resolve().parent.parent / "results" / REFERENCE_RUN
N_STAR = json.loads((_RUN_DIR / "summary.json").read_text())["inputs"]["n_star"]
OFFSET = 1e-10

A1 = 2.0 * Q / (P - 1)
A2 = 3.0 * Q / (2 * P - 1)
B1 = 2.0 * (P + 1) / (P - 1)
B2 = 3.0 * (2 * P + 1) / (2 * P - 1)
C1 = 2.0 * P * (P + 1) / ((P - 1) * Q)
C2 = 6.0 * P * (2 * P + 1) / ((2 * P - 1) * Q)


def g_all(x):
    """g, g', g'' (works on scalars and arrays)."""
    t = x / Q
    tp1 = t ** (P - 1)
    tp = tp1 * t
    t2p1 = tp * tp1
    t2p = tp * tp
    g = x + x * x / 6.0 - A1 * tp * t + A2 * t2p * t
    gp = 1.0 + x / 3.0 - B1 * tp + B2 * t2p
    gpp = 1.0 / 3.0 - C1 * tp1 + C2 * t2p1
    return g, gp, gpp


def U_and_Uphi(x):
    g, gp, _ = g_all(x)
    U = (x * gp - g) / (2.0 * gp * gp)
    Up = (2.0 * g - x * gp) / (S6 * gp * gp)
    return U, Up


_, GP_W, _ = g_all(Q)          # g'(q); U'(q) = 0 exactly (the frozen wall)
PHI_W = S32 * math.log(GP_W)

# PCHIP inverse x(phi) used only as a Newton initial guess.
_u = np.logspace(-13.0, math.log10(1.0 - 0.3 / Q), 6000)
_xtab = (Q * (1.0 - _u))[::-1]                     # ascending x: ~0.3 .. ~q
_phitab = S32 * np.log(g_all(_xtab)[1])
X_GUESS = PchipInterpolator(_phitab, _xtab)


def x_from_G(G, xg):
    """Scalar Newton solve of g'(x) = G."""
    x = xg
    for _ in range(100):
        _, gp, gpp = g_all(x)
        dx = (gp - G) / gpp
        x = min(max(x - dx, 0.05), Q)
        if abs(dx) < 1e-14 * x:
            break
    return x


def x_from_G_vec(G, xg):
    x = np.array(xg, dtype=float)
    for _ in range(100):
        _, gp, gpp = g_all(x)
        dx = (gp - G) / gpp
        x = np.clip(x - dx, 0.05, Q)
        if np.max(np.abs(dx)) < 1e-13 * np.min(x):
            break
    return x


# ----------------------------------------------------------------------
# Background: phi-field EOM in e-folds N (psi = phi - phi_wall for precision)
# ----------------------------------------------------------------------
x_start = Q * (1.0 - OFFSET)
_, gp0, _ = g_all(x_start)
psi_start = S32 * math.log1p((gp0 - GP_W) / GP_W)   # < 0
U0, Up0 = U_and_Uphi(x_start)
pi_start = -Up0 / U0                                # slow-roll attractor

_xcache = [x_start]


def bg_rhs(N, y):
    psi, pi = y
    G = GP_W * math.exp(psi / S32)
    x = x_from_G(G, _xcache[0])
    _xcache[0] = x
    U, Up = U_and_Uphi(x)
    eps = 0.5 * pi * pi
    return (pi, (eps - 3.0) * (pi + Up / U))


def ev_end(N, y):
    return 0.5 * y[1] * y[1] - 1.0


ev_end.terminal = True
ev_end.direction = 1.0

bg = solve_ivp(bg_rhs, (0.0, 400.0), [psi_start, pi_start], method="DOP853",
               rtol=1e-10, atol=1e-22, dense_output=True, events=ev_end)
if not bg.t_events[0].size:
    raise RuntimeError("epsilon = 1 never reached")
N_END = float(bg.t_events[0][0])
N_PIV = N_END - N_STAR

# Dense background sampling
Ng = np.linspace(0.0, N_END, int(N_END / 0.002) + 2)
psi_g, pi_g = bg.sol(Ng)
G_g = GP_W * np.exp(psi_g / S32)
x_g = x_from_G_vec(G_g, X_GUESS(PHI_W + psi_g))
g_g, gp_g, _ = g_all(x_g)
U_g = (x_g * gp_g - g_g) / (2.0 * gp_g ** 2)
Up_g = (2.0 * g_g - x_g * gp_g) / (S6 * gp_g ** 2)
eps_g = 0.5 * pi_g ** 2
H_g = np.sqrt(U_g / (3.0 - eps_g))
aH_g = np.exp(Ng) * H_g                    # a = e^N (a(0) = 1)
pip_g = (eps_g - 3.0) * (pi_g + Up_g / U_g)
assert np.all(pi_g < 0.0), "dphi/dN changed sign -- z would vanish"

# z''/z and a''/a via ln z spline + exact conformal chain rule:
#   z''/z = (aH)^2 [ w_NN + w_N^2 + (1 - eps) w_N ],  w = ln z = N + ln|pi|
wN_g = 1.0 + pip_g / pi_g                          # analytic w_N
wNN_g = CubicSpline(Ng, wN_g).derivative()(Ng)     # numerical w_NN (spline)
S_g = aH_g ** 2 * (wNN_g + wN_g ** 2 + (1.0 - eps_g) * wN_g)   # z''/z
T_g = aH_g ** 2 * (2.0 - eps_g)                                # a''/a

# Conformal time measured from the end of inflation: tau(N_end) = 0, tau < 0.
# Backward cumulative sum of exponential-rule segment integrals of 1/(aH).
f = 1.0 / aH_g
dN = np.diff(Ng)
b = np.diff(np.log(f)) / dN
seg = np.where(np.abs(b) > 1e-12,
               np.diff(f) / np.where(np.abs(b) > 1e-12, b, 1.0),
               0.5 * (f[1:] + f[:-1]) * dN)
tau = np.empty_like(Ng)
tau[:-1] = -np.cumsum(seg[::-1])[::-1]
tau[-1] = 0.0

# Interpolants on the mode-relevant window (avoids the tau -> 0 singularity)
mask = (Ng >= 2.0) & (Ng <= N_PIV + 17.0)
Nm = Ng[mask]
lam = np.log(-tau[mask])                   # strictly decreasing in N
lnaH_m = np.log(aH_g[mask])
N_OF_LNAH = PchipInterpolator(lnaH_m, Nm)
LAM_OF_N = PchipInterpolator(Nm, lam)
order = np.argsort(lam)
SPL_S = CubicSpline(lam[order], (S_g[mask] * tau[mask] ** 2)[order])
SPL_T = CubicSpline(lam[order], (T_g[mask] * tau[mask] ** 2)[order])

K_PIVOT = math.exp(float(PchipInterpolator(Nm, lnaH_m)(N_PIV)))

# Background sanity prints
psi_p, pi_p = bg.sol(N_PIV)
x_p = x_from_G(GP_W * math.exp(psi_p / S32), X_GUESS(PHI_W + psi_p))
U_p, _ = U_and_Uphi(x_p)
eps_p = 0.5 * pi_p ** 2
print("=== background (independent U(phi) construction) ===")
print(f"  N_end (eps=1)            = {N_END:.6f}   [package: 80.853830]")
print(f"  N_pivot                  = {N_PIV:.6f}   [package: 29.243530]")
print(f"  x at pivot               = {x_p:.6f}   [package: 214.580052]")
print(f"  eps at pivot             = {eps_p:.6e} [package: 2.577105e-04]")
print(f"  H at pivot (M=1 units)   = {math.sqrt(U_p/(3-eps_p)):.8f}")
print(f"  k_pivot (a0=1, M=1)      = {K_PIVOT:.6e}")

# ----------------------------------------------------------------------
# Mode integration in conformal time (variable xi = k * tau)
# ----------------------------------------------------------------------
N_LATE = N_PIV + 16.0
N_MID = N_LATE - 3.0
LAM_LATE = float(LAM_OF_N(N_LATE))
LAM_MID = float(LAM_OF_N(N_MID))
psi_l, pi_l = bg.sol(N_LATE)
psi_m, pi_m = bg.sol(N_MID)
Z_LATE = math.exp(N_LATE) * pi_l
Z_MID = math.exp(N_MID) * pi_m


def solve_mode(k, tensor=False):
    """Return v/sqrt->(v at N_MID, v at N_LATE) for MS eq in conformal time."""
    spl = SPL_T if tensor else SPL_S
    lnk = math.log(k)
    N_start = float(N_OF_LNAH(lnk - math.log(100.0)))   # k/(aH) = 100
    xi0 = -k * math.exp(float(LAM_OF_N(N_start)))
    xi_mid = -k * math.exp(LAM_MID)
    xi1 = -k * math.exp(LAM_LATE)

    def rhs(xi, y):
        shat = spl(math.log(-xi) - lnk)                 # (z''/z) tau^2
        return (y[1], (shat / (xi * xi) - 1.0) * y[0])

    y0 = np.array([np.exp(-1j * xi0), -1j * np.exp(-1j * xi0)],
                  dtype=complex)                        # full Bunch-Davies
    sol = solve_ivp(rhs, (xi0, xi1), y0, method="DOP853",
                    rtol=1e-8, atol=1e-12, t_eval=[xi_mid, xi1])
    if not sol.success:
        raise RuntimeError(f"mode k={k} failed: {sol.message}")
    s = 1.0 / math.sqrt(2.0 * k)
    return sol.y[0, 0] * s, sol.y[0, 1] * s


def P_scalar(k):
    vm, ve = solve_mode(k, tensor=False)
    pe = k ** 3 / (2.0 * math.pi ** 2) * abs(ve / Z_LATE) ** 2
    pm = k ** 3 / (2.0 * math.pi ** 2) * abs(vm / Z_MID) ** 2
    return pe, abs(pm / pe - 1.0)


def P_tensor(k):
    vm, ve = solve_mode(k, tensor=True)
    pe = k ** 3 * abs(ve) ** 2 * math.exp(-2.0 * N_LATE)   # up to const norm
    pm = k ** 3 * abs(vm) ** 2 * math.exp(-2.0 * N_MID)
    return pe, abs(pm / pe - 1.0)


# ----------------------------------------------------------------------
# Spectra: pivot fit + 20-point transfer grid
# ----------------------------------------------------------------------
lnk_fit = np.linspace(-0.14, 0.14, 7)                # |ln(k/kp)| < 0.15
lnP_fit, drift_fit = [], []
for lk in lnk_fit:
    p, d = P_scalar(K_PIVOT * math.exp(lk))
    lnP_fit.append(math.log(p))
    drift_fit.append(d)
c1, c0 = np.polyfit(lnk_fit, lnP_fit, 1)
NS = 1.0 + c1
A_PIV = math.exp(c0)

lgk_main = np.linspace(-4.7, 1.0, 20)
k_main = K_PIVOT * 10.0 ** lgk_main
lnk_main = lgk_main * math.log(10.0)
P_main, drift_main = [], []
for k in k_main:
    p, d = P_scalar(k)
    P_main.append(p)
    drift_main.append(d)
P_main = np.array(P_main)
T_mine = P_main / (A_PIV * np.exp(c1 * lnk_main))

print()
print("=== scalar spectrum ===")
print(f"  n_s at pivot (deg-1 fit, |ln k/kp|<0.15) = {NS:.6f}")
print(f"  ln A at pivot (M=1 units)                = {c0:.6f}")
print(f"  max super-horizon R drift (fit set)      = {max(drift_fit):.3e}")
print(f"  max super-horizon R drift (main grid)    = {max(drift_main):.3e}")

i_min = int(np.argmin(T_mine))
print(f"  notch minimum: transfer = {T_mine[i_min]:.6f} "
      f"at k/k_pivot = {10.0 ** lgk_main[i_min]:.6e}")

# ----------------------------------------------------------------------
# Comparison with the pipeline reference
# ----------------------------------------------------------------------
ref = np.genfromtxt(_RUN_DIR / "primordial_transfer.csv",
                    delimiter=",", names=True)
REF_S = PchipInterpolator(np.log(ref["k_over_kpivot"]),
                          np.log(ref["scalar_transfer"]))
REF_T = PchipInterpolator(np.log(ref["k_over_kpivot"]),
                          np.log(ref["tensor_transfer"]))

lnr = np.log(T_mine) - REF_S(lnk_main)
align = np.polyfit(lnk_main, lnr, 1)      # amplitude + tilt rescaling
resid = np.expm1(lnr - np.polyval(align, lnk_main))

print()
print("=== scalar transfer vs reference (after amplitude+tilt alignment) ===")
print(f"  alignment: amplitude factor = {math.exp(align[1]):.6f}, "
      f"tilt b = {align[0]:+.6f}")
print(f"  max |fractional residual| = {np.max(np.abs(resid)):.3e}")
print(f"  rms fractional residual   = {np.sqrt(np.mean(resid ** 2)):.3e}")
print()
print(f"  {'k/k_pivot':>13s} {'T_mine':>13s} {'T_ref':>13s} {'residual':>11s}")
for i in range(len(k_main)):
    print(f"  {10.0 ** lgk_main[i]:13.6e} {T_mine[i]:13.6e} "
          f"{math.exp(REF_S(lnk_main[i])):13.6e} {resid[i]:+11.3e}")

# ----------------------------------------------------------------------
# Tensor check (u'' + (k^2 - a''/a) u = 0) at 3 k-values
# ----------------------------------------------------------------------
lnPT_fit = [math.log(P_tensor(K_PIVOT * math.exp(lk))[0]) for lk in lnk_fit]
d1, d0 = np.polyfit(lnk_fit, lnPT_fit, 1)
r_check = np.array([2.5e-5, 5.0e-3, 0.5])
lnk_t = np.log(r_check)
Tt_mine = np.array([P_tensor(K_PIVOT * r)[0] for r in r_check]) \
    / np.exp(d0 + d1 * lnk_t)
Tt_ref = np.exp(REF_T(lnk_t))

print()
print("=== tensor check (n_t fit = {:+.6f}) ===".format(d1))
print(f"  {'k/k_pivot':>11s} {'T_mine':>11s} {'T_ref':>11s} {'raw diff':>11s}")
for i in range(3):
    print(f"  {r_check[i]:11.4e} {Tt_mine[i]:11.6f} {Tt_ref[i]:11.6f} "
          f"{Tt_mine[i] / Tt_ref[i] - 1.0:+11.3e}")
alt = np.polyfit(lnk_t, np.log(Tt_mine) - np.log(Tt_ref), 1)
res_t = np.expm1(np.log(Tt_mine) - np.log(Tt_ref) - np.polyval(alt, lnk_t))
print(f"  after amp+tilt alignment (1 dof): residuals = "
      + ", ".join(f"{r:+.3e}" for r in res_t))

print()
print(f"total runtime: {time.time() - T0:.1f} s")
