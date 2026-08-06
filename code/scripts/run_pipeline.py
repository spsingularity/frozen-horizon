"""End-to-end pipeline for one value of p.

    python3 scripts/run_pipeline.py --p 67

Writes every number the paper quotes to results/p<P>/. Nothing is transcribed
by hand: downstream stages read the JSON artifacts written by upstream ones.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frozen_horizon import (  # noqa: E402
    FrozenHorizonModel,
    background,
    config,
    modes,
    observables,
    projection,
    stochastic,
)


def notch_location(k_ratios, transfer, k_pivot=None):
    """Locate the transfer minimum by a parabolic fit in ln k around the grid min."""
    k_pivot = config.K_PIVOT if k_pivot is None else k_pivot
    index = int(np.argmin(transfer))
    if 0 < index < len(transfer) - 1:
        window = slice(index - 1, index + 2)
        coefficients = np.polyfit(np.log(k_ratios[window]), transfer[window], 2)
        log_k = -coefficients[1] / (2.0 * coefficients[0])
        depth = np.polyval(coefficients, log_k)
    else:
        log_k = np.log(k_ratios[index])
        depth = transfer[index]
    return {
        "notch_depth": float(depth),
        "notch_k_over_kpivot": float(np.exp(log_k)),
        "notch_k_Mpc": float(np.exp(log_k) * k_pivot),
        "grid_min_depth": float(transfer[index]),
        "grid_min_k_over_kpivot": float(k_ratios[index]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=config.DEFAULT_P)
    parser.add_argument("--alpha", type=float, default=None,
                        help="wall coefficient; default 3 (dz sum rule), "
                             "2 is the RG measure, 0 the linear single-operator wall")
    parser.add_argument("--n-star", type=float, default=config.N_STAR)
    parser.add_argument("--k-min", type=float, default=2.0e-5)
    parser.add_argument("--k-max", type=float, default=10.0)
    parser.add_argument("--k-points", type=int, default=32)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    outdir = args.outdir or ROOT / "results" / f"p{args.p}"
    outdir.mkdir(parents=True, exist_ok=True)

    fh_model = FrozenHorizonModel(args.p, alpha=args.alpha)
    print(f"[1/6] {fh_model}", flush=True)

    bootstrap_block = {
        "p": fh_model.p,
        "q": fh_model.q,
        "alpha": fh_model.alpha,
        "wall_coefficient_alpha": fh_model.alpha,
        "n_power": fh_model.n_power,
        "m_power": fh_model.m_power,
        "horizon_slope_F_H": float(fh_model.horizon_slope),
        "mu_squared": float(fh_model.mu_squared),
        "exit_exponent_s": float(fh_model.exit_exponent),
        "horizon_residual": float(fh_model.horizon_residual()),
    }

    print("[2/6] background", flush=True)
    bg = background.prepare(fh_model, n_star=args.n_star)
    stability = bg.stability()
    flow = modes.hubble_flow_tilts(bg)
    background_block = {
        "N_span_from_offset_start": float(bg.total_efolds),
        "end_N": bg.end_N,
        "pivot_N": bg.pivot_N,
        "n_star": bg.n_star,
        "x_pivot": float(bg.quantities(bg.pivot_N)[0]),
        "M_over_Mpl_slowroll": float(bg.mass_scale),
        "H_pivot_over_Mpl": float(bg.H_pivot),
        **stability,
        **flow,
    }

    print(f"[3/6] modes ({args.k_points} k values, both polarizations)", flush=True)
    k_ratios = np.geomspace(args.k_min, args.k_max, args.k_points)
    k_ratios, scalar, tensor = modes.transfer_table(bg, k_ratios)

    pivot_scalar = float(np.exp(np.interp(0.0, np.log(k_ratios), np.log(scalar))))
    pivot_tensor = float(np.exp(np.interp(0.0, np.log(k_ratios), np.log(tensor))))
    scalar_ratio = scalar / modes.smooth_reference(
        k_ratios, pivot_scalar, flow["ns_first_order"] - 1.0
    )
    tensor_ratio = tensor / modes.smooth_reference(
        k_ratios, pivot_tensor, -2.0 * flow["epsilon_pivot"]
    )

    transfer_path = outdir / "primordial_transfer.csv"
    np.savetxt(
        transfer_path,
        np.column_stack([k_ratios, scalar_ratio, tensor_ratio]),
        delimiter=",",
        header="k_over_kpivot,scalar_transfer,tensor_transfer",
        comments="",
        fmt="%.9e",
    )

    # Absolute dimensionless spectra on a physical k grid, for the Boltzmann
    # stage. P_R here is the actual curvature power, not a ratio, so the
    # Boltzmann code needs no separate amplitude normalization.
    np.savetxt(
        outdir / "primordial_power.csv",
        np.column_stack([k_ratios * config.K_PIVOT, scalar, tensor]),
        delimiter=",",
        header="k_Mpc,P_R,P_t",
        comments="",
        fmt="%.9e",
    )

    notch = notch_location(k_ratios, scalar_ratio, k_pivot=config.K_PIVOT)
    notch["tensor_max_deviation"] = float(np.max(np.abs(tensor_ratio - 1.0)))

    print("[4/6] local observables", flush=True)
    local = observables.local_fit(bg)

    print("[5/6] stochastic duration", flush=True)
    duration = stochastic.duration(fh_model, local["M_over_Mpl_corrected"])

    print("[6/6] Sachs-Wolfe projection", flush=True)
    sw = projection.projection_table(
        k_ratios, scalar_ratio, local["n_s"], local["alpha_s"]
    )
    np.savetxt(
        outdir / "sachs_wolfe_ratios.csv",
        np.column_stack([list(sw.keys()), list(sw.values())]),
        delimiter=",",
        header="ell,C_ell_over_smooth",
        comments="",
        fmt=["%d", "%.9f"],
    )

    summary = {
        "bootstrap": bootstrap_block,
        "background": background_block,
        "observables": local,
        "notch": notch,
        "duration": duration,
        "sachs_wolfe": {"C2_over_smooth": sw[2], "C3_over_smooth": sw[3]},
        "inputs": {
            "A_s_obs": config.A_S_OBS,
            "k_pivot_Mpc": config.K_PIVOT,
            "chi_star_Mpc": config.CHI_STAR,
            "n_star": args.n_star,
            "horizon_offset": config.HORIZON_OFFSET,
            "k_range": [args.k_min, args.k_max, args.k_points],
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {outdir}/summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
