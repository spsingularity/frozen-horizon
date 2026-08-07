"""Derivatives, positivity, and the Starobinsky limit."""

import numpy as np
import pytest

from frozen_horizon import FrozenHorizonModel, background


@pytest.fixture(scope="module")
def model():
    return FrozenHorizonModel(63)


@pytest.mark.parametrize("fraction", [0.3, 0.6, 0.85, 0.97])
def test_first_derivative_matches_finite_difference(model, fraction):
    x = model.q * fraction
    step = x * 1e-7
    numeric = (model.geometry(x + step)[0] - model.geometry(x - step)[0]) / (2 * step)
    assert model.geometry(x)[1] == pytest.approx(numeric, rel=1e-6)


@pytest.mark.parametrize("fraction", [0.3, 0.6, 0.85, 0.97])
def test_second_derivative_matches_finite_difference(model, fraction):
    x = model.q * fraction
    step = x * 1e-6
    numeric = (model.geometry(x + step)[1] - model.geometry(x - step)[1]) / (2 * step)
    assert model.geometry(x)[2] == pytest.approx(numeric, rel=1e-5)


def test_low_curvature_limit_is_starobinsky(model):
    """Far below the horizon the wall terms must be utterly negligible."""
    x = 1.0
    g, gp, gpp, _, _ = model.geometry(x)
    assert g == pytest.approx(x + x * x / 6.0, rel=1e-12)
    assert gp == pytest.approx(1.0 + x / 3.0, rel=1e-12)
    assert gpp == pytest.approx(1.0 / 3.0, rel=1e-12)


@pytest.mark.slow
def test_no_ghost_along_the_trajectory(model):
    """f_R > 0 and f_RR > 0 everywhere, so the scalaron is not a ghost."""
    bg = background.prepare(model)
    x = bg._x(bg.N)
    _, gp, gpp, _, _ = model.geometry(x)
    assert (gp > 0).all(), "f_R changed sign: graviton kinetic term flipped"
    assert (gpp > 0).all(), "f_RR changed sign: scalaron became a ghost"


@pytest.mark.slow
def test_min_f_R_is_the_trajectory_end_not_the_slow_roll_end(model):
    """Guard against reinstating the handout's 2.1547.

    2.1547 = g'(2 sqrt 3) is where the *slow-roll* epsilon_V = 1. The exact
    trajectory ends later, at epsilon = 1, where g' is smaller. Quoting the
    former alongside e-fold counts computed with the latter mixes two different
    end-of-inflation criteria.
    """
    bg = background.prepare(model)
    on_trajectory = bg.stability()["min_f_R"]
    slow_roll = model.geometry(2.0 * np.sqrt(3.0))[1]
    assert slow_roll == pytest.approx(2.1547005, rel=1e-6)
    assert on_trajectory < slow_roll
    assert on_trajectory == pytest.approx(1.6518, rel=1e-3)


# --- cubic coefficient and the linear-treatment self-consistency ---------

@pytest.mark.slow
def test_potential_second_derivative_is_minus_mu2_H2():
    """U''(phi_H) = -mu^2 H_H^2 validates the parametric (phi, U) construction.

    This is the check that makes W_3 trustworthy: if the phi parametrization
    or the mass-scale conversion were wrong, the second derivative would not
    land on the tachyonic mass the bootstrap fixes independently.
    """
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    run = json.loads((root / "results" / "invwall_p66" / "summary.json").read_text())
    model = FrozenHorizonModel(66, alpha=2.809064)
    potential, second, _ = model.potential_taylor(
        run["observables"]["M_over_Mpl_corrected"])
    hubble_squared = potential / 3.0                 # 3 H^2 M_Pl^2 = U
    assert second == pytest.approx(-model.mu_squared * hubble_squared, rel=1e-6)


@pytest.mark.slow
def test_nonlinearity_numbers_are_mutually_consistent():
    """The two quantities Paper II quotes must sit on the same linear relation.

    ratio(X_c) = X_c / dphi_x by construction, so quoting a ratio and a
    crossover suppression independently is a chance to be inconsistent -- and
    the manuscript was, by a factor of about four, until both were computed.
    """
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    run = json.loads((root / "results" / "invwall_p66" / "summary.json").read_text())
    duration = run["duration"]
    model = FrozenHorizonModel(66, alpha=2.809064)
    result = model.nonlinearity(
        run["observables"]["M_over_Mpl_corrected"],
        duration["classicality_boundary_over_Mpl"],
        0.255806 * duration["H_horizon_over_Mpl"])

    assert result["cubic_over_linear_at_threshold"] == pytest.approx(
        duration["classicality_boundary_over_Mpl"] / result["crossover_amplitude"])
    assert result["cubic_over_linear_at_threshold"] == pytest.approx(1.1e-5, rel=0.05)
    assert result["minus_ln_P_at_crossover"] == pytest.approx(2.0e10, rel=0.05)
    # The whole point: nonlinearity is irrelevant where the state has support.
    assert result["minus_ln_P_at_crossover"] > 1.0e9
