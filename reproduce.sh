#!/usr/bin/env bash
#
# Reproduce both manuscripts end to end, from a clean checkout to two PDFs.
#
#   ./reproduce.sh              full run: every spectrum recomputed (~2 h)
#   ./reproduce.sh --fast       skip the pipeline, rebuild from stored runs
#   ./reproduce.sh --check      verify only; recompute nothing
#
# The full run recomputes every number in both papers. Nothing is read from a
# previous invocation except the Planck likelihood data, which is downloaded
# once. On the way out it runs check_manuscript.py, which recomputes each
# number quoted in the prose and fails if the .tex disagrees, so a green run
# means the manuscripts and the code actually agree.
#
# On N_*: the pivot e-fold count is not a free input. The background fixes the
# reheating history, which fixes N_*, which fixes the background. Physics runs
# therefore pass --n-star auto to solve that fixed point. The exceptions are
# deliberate and marked below: the reheat_* runs are the different-N_* control
# that calibrates the shape statistic's null, and the alpha comparison runs
# probe wall shape, whose depth Sec. 5 of Paper I shows is N_*-independent.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$ROOT/code"
PY="$CODE/.venv/bin/python"
MODE="${1:-full}"

XI_CONFORMAL=0.16666666666666666      # conformally coupled Higgs, invariant wall
XI_MINIMAL=0.0                        # minimally coupled matter, coordinate wall

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

[ -x "$PY" ] || { echo "No venv at $PY. Run:"; \
  echo "  cd code && python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"; \
  exit 1; }

if [ "$MODE" != "--check" ] && [ ! -d "$CODE/cobaya_packages" ]; then
  say "Planck 2018 low-l likelihood data (~16 MB, once)"
  "$CODE/.venv/bin/cobaya-install" planck_2018_lowl.TT planck_2018_lowl.EE \
      --packages-path "$CODE/cobaya_packages"
fi

cd "$CODE"

if [ "$MODE" = "full" ]; then
  say "Primordial spectra: invariant wall (primary branch), N_* solved"
  for p in 64 65 66 67; do
    "$PY" scripts/run_pipeline.py --p "$p" --alpha 2.809064 \
        --n-star auto --xi "$XI_CONFORMAL" \
        --k-points 160 --outdir "results/invwall_p$p"
  done

  say "Primordial spectra: coordinate wall (robustness branch), N_* solved"
  for p in 62 63 64 65; do
    "$PY" scripts/run_pipeline.py --p "$p" --alpha 3.0 \
        --n-star auto --xi "$XI_MINIMAL" \
        --k-points 160 --outdir "results/resolved_p$p"
  done

  # Deliberate fixed N_*: the pair (reheat_pN, resolved_pN) is the same p at
  # two different N_*, which calibrates the error floor of the shape statistic.
  # Solving the fixed point here would collapse the control onto its partner.
  say "Control runs at offset N_* (null calibration for the shape statistic)"
  for spec in "62 51.4223" "63 51.4214" "64 51.4212"; do
    set -- $spec
    "$PY" scripts/run_pipeline.py --p "$1" --alpha 3.0 --n-star "$2" \
        --k-points 160 --outdir "results/reheat_p$1"
  done

  # Wall-shape comparisons. Paper I Sec. 5 establishes the depth is
  # N_*-independent, so these are run at a common reference value.
  say "Wall-shape comparison runs (alpha = 0 and alpha = 2)"
  "$PY" scripts/run_pipeline.py --p 246 --alpha 0.0 --n-star 51.421 \
      --k-points 100 --outdir results/alpha0_p246
  "$PY" scripts/run_pipeline.py --p 84 --alpha 2.0 --n-star 51.421 \
      --k-points 100 --outdir results/alpha2_p84

  say "Planck 2018 low-l likelihoods"
  "$PY" scripts/lowl_likelihood_eval.py \
      invwall_p64 invwall_p65 invwall_p66 invwall_p67 \
      resolved_p62 resolved_p63 resolved_p64 resolved_p65
fi

if [ "$MODE" != "--check" ]; then
  say "Ledger and manuscript assets"
  "$PY" scripts/make_ledger.py --p 66 --alpha 2.809064 \
      --xi "$XI_CONFORMAL" --results invwall_p
  "$PY" scripts/make_paper_tables.py
  "$PY" scripts/make_figures.py
  "$PY" scripts/make_paper2_assets.py
fi

say "Tests"
"$PY" -m pytest tests/ -q --run-slow

say "Manuscript numbers against the pipeline"
"$PY" scripts/check_manuscript.py

if [ "$MODE" != "--check" ]; then
  say "Typesetting"
  make -C "$ROOT/paper"
  make -C "$ROOT/paper/paper2"
fi

say "Done"
echo "  $ROOT/paper/nonsingular-fR-starobinsky-completion.pdf"
echo "  $ROOT/paper/paper2/quantum-state-frozen-horizon.pdf"
