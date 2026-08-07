"""Single source of truth for observational anchors and numerical controls.

Every constant the pipeline depends on lives here. No other module defines a
physical or numerical constant, and no module hard-codes a value that another
module computes.
"""

import numpy as np

# --- Observational anchors -------------------------------------------------

A_S_OBS = 2.1e-9      # scalar amplitude used to normalize the mass scale M
K_PIVOT = 0.05        # Mpc^-1, pivot wavenumber
CHI_STAR = 13870.0    # Mpc, approximate comoving distance to last scattering

# Seed for the N_* fixed-point iteration ONLY -- not a physical input.
#
# N_* IS derived: reheating.solve_n_star_resolved() computes it from the
# integrated scalaron/radiation history, giving 51.62 (xi_H = 0) and 50.60
# (xi_H = 1/6) on the primary branch. Pass --n-star auto to run_pipeline.py to
# solve that fixed point; a literal goes stale silently when the reheating
# model changes, which is how the first release computed its spectra at
# 50.5852 against a converged 50.5966.
#
# The value below is the superseded benchmark, retained only as a bracket seed
# because the iteration converges from it in one step. It is deliberately NOT
# a plausible answer, so that any run which ends up reporting it is visibly
# wrong rather than quietly off by an e-fold.
N_STAR = 55.0

# --- Background integration ------------------------------------------------

BG_RTOL = 2.0e-11
BG_ATOL = 2.0e-13
BG_MAX_STEP = 0.01
BG_N_MAX = 300.0

# Fractional offset below the horizon for the mode-solver background start,
# x0 = q * (1 - HORIZON_OFFSET). The e-fold count from this point diverges
# logarithmically as the offset -> 0, so it sets only "early enough for the
# modes" and is NOT a prediction of the inflationary duration. The physical
# duration comes from stochastic.py, which starts at the classicality
# boundary X_c instead.
HORIZON_OFFSET = 1.0e-10

# --- Mode integration ------------------------------------------------------

MODE_RTOL = 2.0e-8
MODE_ATOL_REL = 1.0e-10
MODE_MAX_STEP = 0.03
MODE_START_RATIO = 50.0   # begin each mode when k/(aH) equals this

# --- Reheating -------------------------------------------------------------
#
# Used to compute N_STAR rather than assume it. All densities and rates are in
# reduced-Planck units.

M_PL_GEV = 2.435e18   # reduced Planck mass; every formula here uses this one

G_STAR = 106.75       # SM relativistic dof at reheating
G_STAR_S = 106.75     # entropy dof at reheating

# Additive constant D in the e-fold matching relation, already net of
# -ln[k_*/(a_0 H_0)]. Grouping it this way makes D exactly H_0-independent:
# the constant carries +ln(T_0/H_0) and the pivot term carries -ln(H_0), so H_0
# cancels. Keeping them separate would let an h mismatch inject a systematic
# offset at unit weight -- fatal here, since adjacent models are 0.99 e-folds
# apart. Verified constant to 6 digits at h = 0.66, 0.674, 0.70, 0.73.
#
# Planck 2018 X (arXiv:1807.06211); reduced M_Pl. D collects -ln(3)/2,
# -ln(30/pi^2)/4, +ln(T_0/H_0) and +ln(g_s0)/3 with T_0 = 2.7255 K.
# D0 is D before T_reh is eliminated in favour of rho_reh: it omits the
# (1/4) ln(pi^2/30) that appears when the sudden-decay relation is used, so a
# resolved reheating history can supply N_re and T_re directly.
N_STAR_MATCHING_D0 = {
    0.05: 61.766402,
    0.01: 63.375840,
    0.002: 64.985278,
}

N_STAR_MATCHING_D = {
    0.05: 61.488476,    # Planck pivot
    0.01: 63.097914,
    0.002: 64.707352,   # WMAP pivot -- literature N_* values are often quoted here
}

# Gamma = C M^3 / M_Pl^2 for scalaron decay to the Higgs doublet through the
# trace coupling: C = N_h/(192 pi) = 1/(48 pi) with N_h = 4 real scalar dof.
# Gorbunov & Panin arXiv:1009.2448 Eq. (7); reduced M_Pl.
# Fermion channels are m_psi^2/M^2 suppressed; gauge bosons ~7e-4 of this;
# gravitons vanish at tree level (Koshelev, Starobinsky & Tokareva
# arXiv:2211.02070). Total width is Higgs-dominated to ~0.1%.
# General non-minimal coupling: C -> (1 - 6 xi)^2 / (48 pi).
SCALARON_DECAY_COEFFICIENT = 1.0 / (48.0 * np.pi)

# Higgs curvature coupling at the scalaron decay scale. xi = 0 is minimal
# coupling (tree Higgs channel open); xi = 1/6 is conformal and switches that
# channel off, leaving only the gauge trace anomaly. This is a matter boundary
# condition, NOT a consequence of the gravitational sector, and it shifts N_* by
# about one e-fold -- i.e. one rung of the p ladder.
XI_HIGGS = 0.0

# Gauge trace-anomaly bracket sum_i b_i^2 alpha_i^2(M/2) N_i^adj / (4 pi^2)
# with b_i = (41/6, -19/6, -7) and N_i^adj = (1, 3, 8), one-loop run to M/2.
# Compare the Higgs bracket 4(1-6 xi)^2 = 4 at xi = 0.
ANOMALY_BRACKET = 0.00779

# --- Stochastic exit -------------------------------------------------------
#
# Width of the Born-seeded initial amplitude of the real growing coefficient,
# sigma_C / H_H, from the local classicalizing thimble: sigma_C = |c1 + i r_E
# c2| sigma_Y with r_E = -6.27855862, c1 = 1.15455467, c2 = 0.54066085,
# theta = 71.215906 deg, sigma_Y = 0.0713437 H (|Psi|^2 widths). Property of
# the mu^2 = 3 horizon; independent of p and alpha. Theory doc 6.3/6.5;
# independently reproduced to 8 digits.
BORN_SEED_SIGMA_OVER_H = 0.255806

# Equation of state during scalaron-dominated reheating. V ~ M^2 phi^2 / 2 near
# the minimum, so coherent oscillations are matter-like. This is now the
# DOMINANT uncertainty: w = 0 -> 0.05 moves N_* by +0.84 e-folds, comparable to
# the 0.99 e-fold spacing between adjacent p. Scan it.
W_REHEATING = 0.0

# --- Boltzmann stage -------------------------------------------------------
#
# Late-time cosmology held fixed while the primordial spectrum varies. Planck
# 2018 TT,TE,EE+lowE+lensing base-LCDM. Only the primordial input is model
# dependent, so these are nuisance parameters for the feature ratio and cancel
# to first order in it.
FIDUCIAL_COSMOLOGY = {
    "H0": 67.36,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "tau": 0.0544,
}

# --- Default model ---------------------------------------------------------

# The primary candidate of Paper I: p = 66 on the invariant wall.
DEFAULT_P = 66
