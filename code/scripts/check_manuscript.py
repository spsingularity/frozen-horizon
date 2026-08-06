"""Verify every number quoted in the manuscript prose against the pipeline.

Generated tables and figures cannot drift: they are build artifacts. Prose is
different. A number typed into a sentence stays there when the code beneath it
changes, and nothing fails. That is not hypothetical -- Paper II shipped with
N_tot = 74.403, a value belonging to the superseded p = 67 run, while its own
primary branch gives 70.35.

This script closes that hole. Each entry below recomputes a quantity from the
package (or reads it from a stored result), renders it the way the manuscript
does, and asserts the string is present in the .tex. Numbers that are inputs,
identities, or cited from the literature are listed as such and skipped, so
that the exemption is explicit rather than silent.

    ./.venv/bin/python scripts/check_manuscript.py          # check
    ./.venv/bin/python scripts/check_manuscript.py -v       # show every hit

Exit status is nonzero if any claim is missing, so this belongs in CI.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frozen_horizon import (background, bootstrap, config,  # noqa: E402
                            quantum, reheating, stochastic)
from frozen_horizon.model import FrozenHorizonModel  # noqa: E402

PAPER1 = ROOT.parent / "paper" / "nonsingular-fR-starobinsky-completion.tex"
PAPER2 = (ROOT.parent / "paper" / "paper2"
          / "quantum-state-frozen-horizon.tex")

PRIMARY_P = 66
PRIMARY_ALPHA = 2.809064


def strip(text):
    """Drop comments and collapse whitespace so line breaks do not hide a hit."""
    return re.sub(r"\s+", " ", re.sub(r"(?<!\\)%.*", "", text))


class Report:
    def __init__(self, verbose):
        self.verbose, self.failures, self.checked = verbose, [], 0

    def claim(self, paper, label, value, fmt="{:.4f}"):
        """Assert that `value`, rendered with `fmt`, appears in `paper`.

        `fmt` may be a tuple, in which case every rendering must appear. Papers
        quote the same quantity at several precisions (71.216 in the abstract,
        71.215906 in the text) and each occurrence is an independent chance to
        go stale, so each is checked.
        """
        formats = fmt if isinstance(fmt, tuple) else (fmt,)
        body = strip(paper.read_text())
        ok = True
        for one in formats:
            rendered = one.format(value) if not isinstance(value, str) else value
            self.checked += 1
            if rendered in body:
                if self.verbose:
                    print(f"  ok    {label:<44} {rendered}")
                continue
            ok = False
            self.failures.append((paper.name, label, rendered))
            print(f"  FAIL  {label:<44} {rendered} not found in {paper.name}")
        return ok

    def exempt(self, label, why):
        if self.verbose:
            print(f"  --    {label:<44} exempt: {why}")


def paper1_claims(report):
    print("\nPaper I")
    model = FrozenHorizonModel(PRIMARY_P, alpha=PRIMARY_ALPHA)

    # Wall coefficients: three exact, one a fixed point.
    report.claim(PAPER1, "alpha, d ln R measure",
                 bootstrap.wall_coefficient_in_measure("dlnR"), "{:.0f}")
    report.claim(PAPER1, "alpha, dR measure at p=63",
                 bootstrap.wall_coefficient_in_measure("dR", 63), "{:.6f}")
    report.claim(PAPER1, "alpha, dR measure at p=66",
                 bootstrap.wall_coefficient_in_measure("dR", 66), "{:.6f}")
    # Canonical measure: the quoted range across the viable integers.
    canonical = [bootstrap.canonical_wall_coefficient(p)
                 for p in (64, 65, 66, 67)]      # the invariant-wall rungs
    report.claim(PAPER1, "canonical alpha, upper end", max(canonical), "{:.4f}")
    report.claim(PAPER1, "canonical alpha, lower end", min(canonical), "{:.4f}")

    # Stability numbers on the trajectory.
    report.claim(PAPER1, "g'(2 sqrt 3), the slow-roll end value",
                 model.geometry(2.0 * np.sqrt(3.0))[1], "{:.4f}")
    report.claim(PAPER1, "exit exponent s",
                 model.exit_exponent, ("{:.4f}", "{:.3f}"))

    # S_max = (1+alpha)^2 / 4 alpha for each measure.
    for alpha, name in ((2.0, "d ln R"), (2.0 + 1.0 / PRIMARY_P, "dR"),
                        (PRIMARY_ALPHA, "canonical")):
        report.claim(PAPER1, f"S_max, {name} wall",
                     (1.0 + alpha) ** 2 / (4.0 * alpha), "{:.3f}")

    # Reheating anchors, at the CONVERGED fixed point rather than the input.
    run = json.loads((ROOT / "results" / f"invwall_p{PRIMARY_P}"
                      / "summary.json").read_text())
    for xi, label in ((0.0, "xi=0"), (1.0 / 6.0, "xi=1/6")):
        n_star = run["inputs"]["n_star"]
        for _ in range(8):                      # converges in one step
            solved = reheating.solve_n_star_resolved(
                background.prepare(model, n_star=n_star), xi_higgs=xi)
            if abs(solved["N_star"] - n_star) < 1.0e-9:
                break
            n_star = solved["N_star"]
        report.claim(PAPER1, f"N_* fixed point, {label}", n_star, "{:.3f}")

    # The p-blindness claim, which the whole degeneracy argument rests on.
    # This is an inequality rather than a printed number, so assert it directly.
    converged = {}
    for p in (64, 65, 66, 67):
        rung, n_star = FrozenHorizonModel(p, alpha=PRIMARY_ALPHA), 50.6
        for _ in range(8):
            solved = reheating.solve_n_star_resolved(
                background.prepare(rung, n_star=n_star), xi_higgs=1.0 / 6.0)
            if abs(solved["N_star"] - n_star) < 1.0e-10:
                break
            n_star = solved["N_star"]
        converged[p] = n_star
    spread = max(abs(converged[p + 1] - converged[p]) for p in (64, 65, 66))
    report.checked += 1
    if spread < 1.0e-3:
        if report.verbose:
            print(f"  ok    {'N_* p-blindness < 1e-3 e-folds':<44} "
                  f"{spread:.2e}")
    else:
        report.failures.append((PAPER1.name, "N_* p-blindness", f"{spread:.2e}"))
        print(f"  FAIL  {'N_* p-blindness':<44} {spread:.2e} exceeds 1e-3")

    # Likelihood results, read from the evaluation the tables also read.
    likelihood = {row["run"]: row["dTOT"] for row in json.loads(
        (ROOT / "results" / "lowl_likelihood.json").read_text())}
    report.claim(PAPER1, "excluded rung, coordinate branch p=62",
                 likelihood["resolved_p62"], "{:+.1f}")
    report.claim(PAPER1, "excluded rung, invariant branch p=64",
                 likelihood["invwall_p64"], "{:+.1f}")
    best = min(likelihood[r] for r in
               ("invwall_p66", "resolved_p64", "invwall_p65", "resolved_p65"))
    report.claim(PAPER1, "best preferred rung", best, "{:.1f}")

    report.exempt("A_s, Planck cosmology, chi_*", "observational inputs")
    report.exempt("N=54.4 of Bezrukov-Gorbunov", "cited from literature")
    report.exempt("marginalized evidence +1.0", "read from evidence.tex")


def paper2_claims(report):
    print("\nPaper II")
    thimble = quantum.classicalizing_thimble()

    # The closed form and the state at the unit gap.
    report.claim(PAPER2, "theta at unit gap (deg)",
                 np.degrees(quantum.theta_exact()), "{:.6f}")
    report.claim(PAPER2, "cos 2 theta", np.cos(np.pi * quantum.growth_exponent()),
                 "{:.8f}")
    report.claim(PAPER2, "r_E = K_0 / 2 pi^2",
                 quantum.euclidean_mode_exact(0), "{:.8f}")
    report.claim(PAPER2, "c_1", thimble["c1"], "{:.8f}")
    report.claim(PAPER2, "c_2", thimble["c2"], "{:.8f}")
    report.claim(PAPER2, "K_0 H^2",
                 2.0 * np.pi**2 * quantum.euclidean_mode_exact(0), "{:.4f}")
    report.claim(PAPER2, "sigma_Y / H", thimble["sigma_Y_over_H"], "{:.5f}")
    report.claim(PAPER2, "sigma_Cg / H", thimble["sigma_Cg_over_H"], "{:.5f}")
    report.claim(PAPER2, "kappa_C", thimble["kappa_C"], "{:.5f}")

    # Shells.
    shells = quantum.shell_crossings(max_n=8)
    continuum = quantum.continuum_kick()
    report.claim(PAPER2, "continuum Q_nu(1)", continuum, "{:.4f}")
    report.claim(PAPER2, "sqrt Q_nu(1)", np.sqrt(continuum), "{:.3f}")
    report.claim(PAPER2, "Q_1 effective", shells[0]["Q_effective"], "{:.2f}")
    report.claim(PAPER2, "ln 2, first crossing", np.log(2.0), "{:.3f}")

    # Duration, from the primary branch and nothing else.
    model = FrozenHorizonModel(PRIMARY_P, alpha=PRIMARY_ALPHA)
    run = json.loads((ROOT / "results" / f"invwall_p{PRIMARY_P}"
                      / "summary.json").read_text())
    duration = stochastic.duration(
        model, run["observables"]["M_over_Mpl_corrected"])
    report.claim(PAPER2, "N_stochastic mean (Born-seeded)",
                 duration["N_stochastic_mean"], "{:.3f}")
    report.claim(PAPER2, "N_stochastic std", duration["N_stochastic_std"],
                 "{:.3f}")
    report.claim(PAPER2, "N_stochastic sharp mean",
                 duration["N_stochastic_sharp_mean"], "{:.3f}")
    report.claim(PAPER2, "N_total", duration["N_total_mean"], "{:.3f}")

    # The double-count the paper warns about.
    kick = stochastic.tachyonic_kick_factor(1.0, mu2=model.mu_squared)
    report.claim(PAPER2, "double-count ln sqrt(Q)/s",
                 np.log(kick) / model.exit_exponent, "{:.3f}")

    report.exempt("WKB bound 9/4, window 7/4 and 4", "exact rationals in text")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    report = Report(args.verbose)
    paper1_claims(report)
    paper2_claims(report)

    print(f"\n{report.checked} claims checked, {len(report.failures)} failed")
    if report.failures:
        print("\nEach failure means the manuscript states a number the code "
              "does not produce.")
        for paper, label, rendered in report.failures:
            print(f"  {paper}: {label} -> {rendered}")
        return 1
    print("Every checked number in the prose is reproduced by the pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
