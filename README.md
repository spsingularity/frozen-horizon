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
pipeline. No number in either paper is transcribed by hand.

## Install

```bash
python3 -m venv .venv
./.venv/bin/pip install -r code/requirements.txt
```

The Planck low-`ℓ` likelihood data (~16 MB) is fetched separately:

```bash
./.venv/bin/cobaya-install planck_2018_lowl.TT planck_2018_lowl.EE \
    --packages-path code/cobaya_packages
```

## Reproducing the manuscripts

From `code/`:

```bash
# Paper I: ledger, tables, figures
./.venv/bin/python scripts/make_ledger.py --p 66 --alpha 2.809064 \
    --xi 0.16666666666666666 --results invwall_p
./.venv/bin/python scripts/make_paper_tables.py
./.venv/bin/python scripts/make_figures.py

# Paper II: tables and figures (analytic; no stored results needed)
./.venv/bin/python scripts/make_paper2_assets.py

./.venv/bin/python -m pytest tests/ --run-slow
```

Then `make` in `paper/` and `paper/paper2/`.

To regenerate the underlying runs rather than use the stored ones (each
160-point spectrum takes a few minutes):

```bash
./.venv/bin/python scripts/run_pipeline.py --p 66 --alpha 2.809064 \
    --n-star 50.5852 --k-points 160 --outdir results/invwall_p66
./.venv/bin/python scripts/lowl_likelihood_eval.py invwall_p66      # etc.
```

## Layout

| Path | Purpose |
|---|---|
| `code/frozen_horizon/config.py` | every physical and numerical constant, with sources |
| `code/frozen_horizon/bootstrap.py` | `C(p)`, `q(p)`, `μ²(p)`; wall coefficient from the zero-work condition |
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
| `independent_solver.py` | conformal-time Mukhanov–Sasaki solver, structurally distinct from the pipeline; transfer agrees to rms `4e-4` |
| `dimension_check.py` | `d`-dimensional form of the horizon conditions |
| `eos_reheating.py` | scalaron equation of state measured, not assumed |
| `shape_statistic.py` | translation-invariant `D_pp'`: `p` is not observable beyond feature location |
| `marginalize_rungs.py` | rung-marginalized evidence (reads the likelihood JSON; no embedded numbers) |

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
