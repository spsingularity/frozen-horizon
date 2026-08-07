# Frozen-horizon cosmology

Code and numerical results for two companion papers on a metric `f(R)`
extension of Starobinsky inflation possessing an exact constant-curvature
solution — a "frozen horizon" — with a single unstable scalaron mode.

**Paper I** — *A nonsingular metric f(R) completion of Starobinsky inflation
with a reheating-determined low-multipole feature.* The construction, its
reheating closure, the primordial spectra, and the official Planck 2018
low-`ℓ` likelihood evaluation.

**Paper II** — *The quantum state of the frozen horizon: Neumann contour,
classicalizing thimble, and the discrete mode-crossing kernel.* The Euclidean
saddle, the contour over its unstable direction, the admissible window
`7/4 < μ² < 4`, and the discrete `S³` mode-crossing kernel.

Every table and figure in both manuscripts is a build artifact of this
pipeline. Numbers quoted in the prose are checked against it by
`scripts/check_manuscript.py`, which recomputes each one and fails if the
manuscript disagrees:

```bash
./.venv/bin/python scripts/check_manuscript.py -v
```

This runs as part of the test suite. It exists because generated tables
cannot drift but prose can: the first release of Paper II quoted a total
duration belonging to a superseded run, and nothing caught it.

## Install

```bash
cd code
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Reproducing the manuscripts

One script goes from a clean checkout to both PDFs:

```bash
./reproduce.sh              # full: every spectrum recomputed (~2 h)
./reproduce.sh --fast       # rebuild assets and papers from the stored runs
./reproduce.sh --check      # verify only; recompute nothing
```

It fetches the Planck low-`ℓ` likelihood data on first use (~16 MB), runs
the thirteen pipeline configurations, evaluates the likelihoods, regenerates
every table and figure, runs the tests, checks the manuscript prose against
the pipeline, and typesets both papers. A green run means the manuscripts
and the code agree.

The individual steps, if you want them separately, are in the script; it is
short and ordered. Two conventions it encodes are worth stating here.

**`N_*` is not a free input.** The background fixes the reheating history,
which fixes `N_*`, which fixes the background, so the physics runs pass
`--n-star auto` to solve that fixed point. A hand-supplied value goes stale
silently when the reheating model changes, and did: the first release
computed the spectra at `50.5852` against a converged `50.5966`. The map is
nearly flat, so `auto` converges in one step. Two sets of runs keep a fixed
`N_*` deliberately — the `reheat_*` control, which calibrates the shape
statistic's null by sitting at a different `N_*` than its partner, and the
`α = 0` and `α = 2` wall-shape comparisons, whose depth Paper I Sec. 5 shows
is `N_*`-independent.

**The wall coefficient is derived, not assumed.** Three of the four zero-work
measures are exact (`3`, `2`, and `2 + 1/p`); the canonical field-space
measure is a fixed point, `bootstrap.canonical_wall_coefficient(66)`
`-> 2.809064`.

## Layout

| Path | Purpose |
|---|---|
| `code/frozen_horizon/config.py` | every physical and numerical constant, with sources |
| `code/frozen_horizon/bootstrap.py` | `C(p)`, `q(p)`, `μ²(p)`; wall coefficient from the zero-work condition in each measure |
| `code/frozen_horizon/model.py` | `g`, `g'`, `g''` from `p` and `α` alone |
| `code/frozen_horizon/background.py` | Einstein-frame trajectory, stability, energy densities |
| `code/frozen_horizon/modes.py` | scalar and tensor mode integration |
| `code/frozen_horizon/observables.py` | `n_s`, `α_s`, `r`, amplitude normalization |
| `code/frozen_horizon/reheating.py` | scalaron decay, resolved reheating, `N_*` as a fixed point |
| `code/frozen_horizon/stochastic.py` | tachyonic first-passage exit, duration |
| `code/frozen_horizon/quantum.py` | Euclidean kernel, classicalizing contour, `S³` shells (Paper II) |
| `code/frozen_horizon/projection.py` | Sachs–Wolfe estimate, retained as a cross-check |
| `code/frozen_horizon/boltzmann.py` | CAMB propagation of the exact primordial spectrum |
| `code/scripts/` | pipeline, analysis, and manuscript-asset generation |
| `code/results/` | stored runs used by the manuscripts |

### Validation scripts

| Script | Establishes |
|---|---|
| `independent_solver.py` | conformal-time Mukhanov–Sasaki solver, structurally distinct from the pipeline; transfer agrees to rms `3.9e-4` |
| `dimension_check.py` | `d`-dimensional form of the horizon conditions |
| `eos_reheating.py` | scalaron equation of state measured, not assumed |
| `shape_statistic.py` | translation-invariant `D_pp'`: `p` is not observable beyond feature location |
| `marginalize_rungs.py` | rung-marginalized evidence (reads the likelihood JSON; no embedded numbers) |
| `check_manuscript.py` | every number in the manuscript prose, recomputed and matched against the `.tex` |
| `ee_diagonal_diagnostic.py` | why the exact SimAll likelihood is used for `EE` rather than diagonal errors |
| `class_crosscheck.py` | independent CLASS reproduction of the CAMB feature ratios |

## Notes on the stored results

`code/results/` contains only the runs the manuscripts use. Superseded and
exploratory runs — earlier `N_* = 55` benchmarks, calibration scans, and
configurations that failed on inflationary duration — are kept out of this
repository. Any run whose numbers appear in either paper is present here and
was produced by the current pipeline.

Two numerical hazards are worth flagging for anyone extending this code, as
both are silent when they occur and both were encountered during
development. The regular Euclidean modes behave as `f_n ~ u^n`, which
underflows a solver's absolute tolerance for `n ≳ 6`; `quantum.py` normalizes
`f_n(u_0) = 1` instead. And the Born width of the classicalizing contour
follows from `|Ψ|²`, not `Ψ`, a factor of `√2` that propagates into the exit
seed and the inflationary duration.

## License

MIT — see `LICENSE`. If you use this code, please cite the papers and the
archived release:

> Pandev, S. *Frozen-horizon cosmology: pipeline, results, and manuscripts.*
> Zenodo, [doi:10.5281/zenodo.21829379](https://doi.org/10.5281/zenodo.21829379)
> (concept DOI — always resolves to the latest version).
