"""Reheating: the calculation that breaks the p / N_* degeneracy.

These tests exist because the whole selection of p rests on N_*, and a
systematic error there shifts which integer the theory picks rather than
merely blurring it.
"""

import numpy as np
import pytest

from frozen_horizon import FrozenHorizonModel, background, config, reheating


def test_matching_constant_derived_from_first_principles():
    """D must equal -ln k_* + ln T_0 + (1/3)ln g_s0 - (1/2)ln 3 + (1/4)ln(pi^2/30).

    Recomputing it here rather than trusting the tabulated value: an error in D
    enters N_* at unit weight and would move the selected p.
    """
    hbar_c = 1.9733e-14          # GeV cm
    mpc_cm = 3.0857e24
    mpc_inverse_GeV = hbar_c / mpc_cm
    T0 = 2.7255 * 8.617333e-14   # GeV
    g_s0 = 2.0 + 21.0 / 11.0

    for k_pivot, tabulated in config.N_STAR_MATCHING_D.items():
        derived = (
            -np.log(k_pivot * mpc_inverse_GeV)
            + np.log(T0)
            + np.log(g_s0) / 3.0
            - 0.5 * np.log(3.0)
            + 0.25 * np.log(np.pi**2 / 30.0)
        )
        assert derived == pytest.approx(tabulated, abs=1e-4)


def test_matching_constant_is_hubble_independent():
    """D absorbs -ln[k_*/(a_0 H_0)], so H_0 must cancel exactly.

    Keeping the two pieces separate would let an h mismatch inject a systematic
    offset; this pins the grouping that prevents it.
    """
    speed_of_light = 2.99792458e5      # km/s
    hbar_c = 1.9733e-14                # GeV cm
    mpc_cm = 3.0857e24
    mpc_inverse_GeV = hbar_c / mpc_cm
    T0 = 2.7255 * 8.617333e-14         # GeV
    g_s0 = 2.0 + 21.0 / 11.0
    k_pivot = 0.05                     # Mpc^-1

    values = []
    for H0_km_s_Mpc in (66.0, 67.36, 70.0, 73.0):
        H0_Mpc = H0_km_s_Mpc / speed_of_light          # Mpc^-1
        H0_GeV = H0_Mpc * mpc_inverse_GeV              # GeV

        # C_match carries +ln(T_0/H_0) with H_0 in GeV ...
        c_match = (
            -0.5 * np.log(3.0)
            - 0.25 * np.log(30.0 / np.pi**2)
            + np.log(T0 / H0_GeV)
            + np.log(g_s0) / 3.0
        )
        # ... and the pivot term carries -ln H_0 with H_0 in Mpc^-1.
        values.append(c_match - np.log(k_pivot / H0_Mpc))

    assert max(values) - min(values) == pytest.approx(0.0, abs=1e-12)
    # and the H_0-free value is the tabulated D
    assert values[0] == pytest.approx(config.N_STAR_MATCHING_D[k_pivot], abs=1e-4)


@pytest.mark.parametrize("p", [63, 67])
def test_decay_rate_scales_as_M_cubed(p):
    model = FrozenHorizonModel(p)
    assert reheating.scalaron_decay_rate(2.0e-5) / reheating.scalaron_decay_rate(1.0e-5) \
        == pytest.approx(8.0, rel=1e-12)


def test_decay_coefficient_matches_gorbunov_panin():
    """Gamma = N_h M^3 / (192 pi M_Pl^2) with N_h = 4, i.e. 1/(48 pi)."""
    assert config.SCALARON_DECAY_COEFFICIENT == pytest.approx(
        4.0 / (192.0 * np.pi), rel=1e-14
    )


@pytest.mark.slow
def test_n_star_is_p_blind():
    """The premise of the whole argument: reheating cannot see p.

    Reheating happens at low curvature, where every member of the family is the
    same Starobinsky theory. If this ever fails, p and N_* are degenerate again
    and no p can be selected.
    """
    values = []
    for p in (62, 63, 64):
        bg = background.prepare(FrozenHorizonModel(p))
        values.append(reheating.solve_n_star(bg)["N_star"])
    spread = max(values) - min(values)
    assert spread < 0.01, f"N_* varies by {spread:.4f} e-folds across p"


@pytest.mark.slow
def test_n_star_matches_starobinsky_literature():
    """Cross-check against the published Starobinsky value at k = 0.05.

    Beware: the commonly quoted 54.37 is at the WMAP pivot k = 0.002. At
    k = 0.05 it is 54.37 - ln 25 = 51.15. Validating against 54 here would bake
    in a 3.2 e-fold error, which is over three model spacings.
    """
    bg = background.prepare(FrozenHorizonModel(63))
    n_star = reheating.solve_n_star(bg)["N_star"]
    assert n_star == pytest.approx(51.15, abs=0.5)


@pytest.mark.slow
def test_n_star_sensitivity_to_decay_rate_is_one_sixth():
    """dN_*/dln Gamma = 1/6, so Gamma barely matters."""
    bg = background.prepare(FrozenHorizonModel(63))
    base = reheating.solve_n_star(bg)["N_star"]

    def with_coefficient(factor):
        original = config.SCALARON_DECAY_COEFFICIENT
        config.SCALARON_DECAY_COEFFICIENT = original * factor
        try:
            return reheating.solve_n_star(bg)["N_star"]
        finally:
            config.SCALARON_DECAY_COEFFICIENT = original

    shifted = with_coefficient(np.e)
    assert shifted - base == pytest.approx(1.0 / 6.0, abs=0.02)
