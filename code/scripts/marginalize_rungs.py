"""Rung-marginalized evidence: the honest summary statistic for the ladder.

Selecting the best rung after inspection incurs uncontrolled look-elsewhere
freedom. The statistically honest alternative is to treat the viable integer
ladder as a discrete parameter with a flat prior and report the marginalized
likelihood ratio against the smooth baseline:

    L_marg / L_smooth = (1/N) sum_p exp(-Delta_chi2_p / 2)
    evidence          = 2 ln(L_marg / L_smooth)     (>0 favours the feature)

This penalizes the family automatically: rungs that fit worse than smooth
pull the average down, so a ladder with one mildly-good rung and several bad
ones scores worse than its best member -- exactly the look-elsewhere charge.

Values are read from results/lowl_likelihood.json, the output of
lowl_likelihood_eval.py, so this script cannot drift out of step with the
likelihood evaluation or with the manuscript tables.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Branch definitions: (label, run prefix, integers, hard-exclusion threshold).
BRANCHES = [
    ("coordinate wall (alpha=3, xi_H=0)", "resolved_p", [62, 63, 64, 65]),
    ("invariant wall (alpha=2.809, xi_H=1/6)", "invwall_p", [64, 65, 66, 67]),
]
EXCLUDED_ABOVE = 6.0


def marginalized_evidence(delta_chi2_values):
    weights = np.exp(-0.5 * np.asarray(delta_chi2_values))
    return 2.0 * np.log(np.mean(weights))


def load():
    path = ROOT / "results" / "lowl_likelihood.json"
    rows = json.loads(path.read_text())
    return {r["run"]: r["dTOT"] for r in rows}


def main():
    totals = load()
    viable = {}

    for label, prefix, rungs in BRANCHES:
        print(f"\n{label}")
        print(f"  {'p':>4} {'dTT+EE':>9}")
        live = []
        for p in rungs:
            key = f"{prefix}{p}"
            if key not in totals:
                print(f"  {p:>4} {'(missing)':>9}")
                continue
            value = totals[key]
            flag = "  excluded" if value > EXCLUDED_ABOVE else ""
            print(f"  {p:>4} {value:>9.2f}{flag}")
            if value <= EXCLUDED_ABOVE:
                live.append(value)
        viable[label] = live
        print(f"  evidence over viable rungs: "
              f"{marginalized_evidence(live):+.2f}")

    pooled = [v for live in viable.values() for v in live]
    best = min(pooled)
    print(f"\npooled, {len(pooled)} viable integers: "
          f"{marginalized_evidence(pooled):+.2f}")
    print(f"best single integer (no penalty):  {-best:+.2f}")
    print("\nread: |evidence| of order unity is indistinguishable from noise; "
          "the difference from the best-single-integer value is the "
          "look-elsewhere penalty.")


if __name__ == "__main__":
    main()
