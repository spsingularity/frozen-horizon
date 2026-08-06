"""Tests for the Paper II quantum-state module.

The central check is that the closed form theta = pi s / 2 agrees with direct
numerical integration of the Euclidean and Lorentzian mode equations. Those two
routes share no code: theta_exact() evaluates Gamma functions, while
classicalizing_thimble() integrates two ODEs and takes an argument.
"""

import numpy as np
import pytest
from scipy.special import gamma

from frozen_horizon import quantum as q

WINDOW = [0.5, 1.0, 1.75, 2.5, 3.0, 3.5, 3.99]


@pytest.mark.parametrize("mu2", WINDOW)
def test_theta_closed_form_matches_integration(mu2):
    """theta = pi s / 2 exactly, against the ODE solve."""
    numeric = q.classicalizing_thimble(mu2=mu2)["theta_deg"]
    assert numeric == pytest.approx(np.degrees(q.theta_exact(mu2)), abs=1.0e-8)


@pytest.mark.parametrize("mu2", WINDOW)
def test_connection_coefficients(mu2):
    """The hypergeometric connection formulae behind the closed form."""
    thimble = q.classicalizing_thimble(mu2=mu2)
    s = q.growth_exponent(mu2)
    ratio = 0.5 * gamma((s + 3) / 2) * gamma((s + 1) / 2) / (
        gamma((s + 4) / 2) * gamma((s + 2) / 2))
    r_e = 2.0 * gamma((s + 4) / 2) * gamma((1 - s) / 2) / (
        gamma((s + 3) / 2) * gamma(-s / 2))
    assert thimble["c2"] / thimble["c1"] == pytest.approx(ratio, rel=1.0e-9)
    assert thimble["r_E"] == pytest.approx(r_e, rel=1.0e-9)


def test_growth_exponent_solves_its_defining_equation():
    for mu2 in WINDOW:
        s = q.growth_exponent(mu2)
        assert s * (s + 3.0) == pytest.approx(mu2)


def test_threshold_is_s_one_half():
    """The window edge is exactly where theta = 45 deg."""
    assert q.growth_exponent(q.MU2_NORMALIZABLE_MIN) == pytest.approx(0.5)
    assert np.degrees(q.theta_exact(q.MU2_NORMALIZABLE_MIN)) == pytest.approx(45.0)
    assert not q.normalizability_window(1.74)["normalizable"]
    assert q.normalizability_window(1.76)["normalizable"]


def test_unit_gap_lies_inside_the_window():
    window = q.normalizability_window(3.0)
    assert window["one_negative_mode"] and window["normalizable"]
    assert window["theta_deg"] == pytest.approx(71.215906, abs=1.0e-6)
    assert window["cos_2theta"] == pytest.approx(-0.79262829, abs=1.0e-8)


def test_shell_crossings_are_log_n_plus_one():
    """N_n = ln(n+1) exactly, from cosh(arcsinh(sqrt(n(n+2)))) = n+1."""
    for row in q.shell_crossings(max_n=8):
        assert row["N_n"] == pytest.approx(np.log(row["n"] + 1.0), abs=1.0e-13)


def test_first_shell_exceeds_continuum_threefold():
    shells = q.shell_crossings(max_n=8)
    continuum = q.continuum_kick()
    assert shells[0]["Q_effective"] / continuum == pytest.approx(3.09, abs=0.01)


def test_shell_sequence_crosses_the_continuum_value():
    """It does not converge monotonically from above: it overshoots, crosses
    near n = 4, and undershoots by about 2.5 percent before returning."""
    shells = q.shell_crossings(max_n=48)
    continuum = q.continuum_kick()
    ratios = [row["Q_effective"] / continuum for row in shells]
    assert ratios[3] > 1.0 > ratios[4]
    assert min(ratios) == pytest.approx(0.975, abs=0.002)
    assert ratios[-1] == pytest.approx(0.991, abs=0.002)


def test_euclidean_kernel_has_exactly_one_negative_mode():
    _, kernel = q.fluctuation_spectrum(max_n=8)
    assert kernel[0] < 0.0
    assert all(kernel[n] > 0.0 for n in range(1, 9))
