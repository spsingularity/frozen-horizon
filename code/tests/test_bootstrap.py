"""The algebra the handout states in prose, asserted so it can fail."""

import numpy as np
import pytest

from frozen_horizon import FrozenHorizonModel, bootstrap


def test_sum_rule_fixes_wall_coefficient():
    """int_0^1 [S(z) - 1] dz = 0 has the unique solution alpha = 3."""
    assert bootstrap.WALL_COEFFICIENT == pytest.approx(3.0, abs=1e-12)


@pytest.mark.parametrize("p", [62, 63, 64, 66, 67, 68])
def test_unit_gap_gives_mu_squared_three(p):
    """mu^2 = 3 must come out of the algebra, not be asserted.

    The handout prints a literal 3.0 here, which hides the fact that q was
    *defined* to make this true. Computing it at least verifies the inversion.
    """
    model = FrozenHorizonModel(p)
    assert model.mu_squared == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("p", [62, 63, 64, 66, 67, 68])
def test_horizon_is_an_exact_root(p):
    """R f_R - 2f = 0 at x = q, since S(1) = 0."""
    model = FrozenHorizonModel(p)
    assert model.horizon_residual() == pytest.approx(0.0, abs=1e-9 * model.q)


@pytest.mark.parametrize("p", [62, 63, 67])
@pytest.mark.parametrize("fraction", [0.2, 0.5, 0.9, 0.99, 1.0])
def test_trace_slope_is_exactly_the_wall(p, fraction):
    """(2g - x g')/x = (1-z)(1+3z) identically, for all x and all p.

    This is an identity, not a coincidence: the R^2/6 term cancels out of the
    trace slope exactly, which is why the Starobinsky limit does not disturb
    the wall.
    """
    model = FrozenHorizonModel(p)
    numeric, wall = model.trace_slope_check(model.q * fraction)
    assert numeric == pytest.approx(wall, abs=1e-12)


def test_q_matches_published_value_for_p67():
    """Regression pin against the original handout's q."""
    assert FrozenHorizonModel(67).q == pytest.approx(262.0464798359535, rel=1e-15)


def test_exit_exponent_matches_closed_form():
    s = FrozenHorizonModel(63).exit_exponent
    assert s == pytest.approx((np.sqrt(21.0) - 3.0) / 2.0, rel=1e-14)


def test_euclidean_spectrum_has_exactly_one_negative_mode():
    """lambda_L = L(L+3) - mu^2 with mu^2 = 3: one negative mode, unit gap."""
    L, eigenvalues = bootstrap.euclidean_eigenvalues(max_L=5)
    assert eigenvalues[0] == pytest.approx(-3.0)
    assert eigenvalues[1] == pytest.approx(1.0)      # the unit gap
    assert (eigenvalues[2:] > 0).all()
    assert (eigenvalues < 0).sum() == 1


@pytest.mark.parametrize("p", [62, 63, 64])
def test_curvature_powers_follow_from_p(p):
    """The exponents are p+1 and 2p+1; nothing is hard-coded."""
    model = FrozenHorizonModel(p)
    assert model.m_power == p + 1
    assert model.n_power == 2 * p + 1
