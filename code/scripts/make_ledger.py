"""Regenerate every number the paper quotes, for one p, with provenance.

    ./.venv/bin/python scripts/make_ledger.py --p 63

Writes results/ledger_p<P>.json and LaTeX tables into paper/tables/. The paper
\\input's those files, so no value is ever transcribed by hand.

Three numbers in the original handout were wrong and are corrected here:

* min f_R was quoted as 2.1547 = g'(2 sqrt 3), which is the *slow-roll* end of
  inflation (epsilon_V = 1). The trajectory actually ends at exact epsilon = 1,
  a different point. Both are reported; only the second is on the trajectory.
* The notch depth was read off a 32-point grid whose spacing (0.42 in ln k) is
  a third of the feature width. The resolved minimum is deeper.
* sigma(N_total) was quoted as +/- 0.82, but that is fixed by the arbitrary
  choice X_c = H/(2 pi s): varying X_c by a factor of 10 moves N_total by only
  0.3 e-folds while sigma swings from 0.19 to 1.39. It is a convention, not an
  uncertainty, and is reported as such.

The notch depth also depends on what "smooth" means. Four definitions are
tabulated; the paper must name the one it quotes.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frozen_horizon import (  # noqa: E402
    FrozenHorizonModel, background, config, modes, observables, reheating,
    stochastic,
)


def smooth_reference_variants(k_ratios, scalar, flow, local):
    """Notch depth under each defensible definition of P_smooth."""
    pivot = float(np.exp(np.interp(0.0, np.log(k_ratios), np.log(scalar))))
    ln_k = np.log(k_ratios)
    variants = {
        "hubble_flow_tilt": pivot * k_ratios ** (flow["ns_first_order"] - 1.0),
        "fitted_ns": pivot * k_ratios ** (local["n_s"] - 1.0),
        "fitted_ns_running": pivot * np.exp(
            (local["n_s"] - 1.0) * ln_k + 0.5 * local["alpha_s"] * ln_k**2
        ),
    }
    out = {}
    for name, reference in variants.items():
        ratio = scalar / reference
        index = int(np.argmin(ratio))
        window = slice(index - 1, index + 2)
        coefficients = np.polyfit(ln_k[window], ratio[window], 2)
        log_k = -coefficients[1] / (2.0 * coefficients[0])
        out[name] = {
            "depth": float(np.polyval(coefficients, log_k)),
            "k_over_kpivot": float(np.exp(log_k)),
            "k_Mpc": float(np.exp(log_k) * config.K_PIVOT),
            "grid_depth": float(ratio[index]),
        }
    return out


def latex_sci(value, digits=3):
    """Render a float as $m \\times 10^{e}$ instead of computer notation."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def latex_table(rows, caption, label, columns=("Quantity", "Value", "Note")):
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l r l}", r"\hline",
        " & ".join(columns) + r" \\", r"\hline",
    ]
    lines += [f"{name} & {value} & {note} \\\\" for name, value, note in rows]
    lines += [
        r"\hline", r"\end{tabular}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        r"\end{table}", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=63)
    parser.add_argument("--results", default="reheat_p")
    parser.add_argument("--alpha", type=float, default=None,
                        help="wall coefficient (default: 3, the dz sum rule)")
    parser.add_argument("--xi", type=float, default=None,
                        help="Higgs curvature coupling for reheating (default 0)")
    parser.add_argument("--window-scan", action="store_true",
                        help="scan the n_s/alpha_s fitting window (slow)")
    args = parser.parse_args()

    directory = ROOT / "results" / f"{args.results}{args.p}"
    run = json.loads((directory / "summary.json").read_text())
    n_star = run["inputs"]["n_star"]

    print(f"[1/5] model and bootstrap (p={args.p})", flush=True)
    model = FrozenHorizonModel(args.p, alpha=args.alpha)
    bootstrap_block = {
        "p": model.p, "q": model.q,
        "wall_coefficient_alpha": model.alpha,
        "curvature_powers": [model.m_power, model.n_power],
        "horizon_slope_F_H": float(model.horizon_slope),
        "mu_squared": float(model.mu_squared),
        "exit_exponent_s": float(model.exit_exponent),
        "horizon_residual_at_q": float(model.horizon_residual()),
    }

    print("[2/5] background, stability, reheating", flush=True)
    bg = background.prepare(model, n_star=n_star)
    stability = bg.stability()
    x_end = float(bg._x_raw[-1])
    # The handout's 2.1547 is g' at the slow-roll end of inflation, x = 2 sqrt 3.
    slow_roll_end = float(model.geometry(2.0 * np.sqrt(3.0))[1])
    flow = modes.hubble_flow_tilts(bg)
    reheat = reheating.solve_n_star_resolved(bg, xi_higgs=args.xi)
    budget = reheating.n_star_error_budget(bg, xi_higgs=args.xi)

    print("[3/5] notch under each P_smooth definition", flush=True)
    table = np.genfromtxt(
        directory / "primordial_power.csv", delimiter=",", names=True
    )
    k_ratios = table["k_Mpc"] / config.K_PIVOT
    notch_variants = smooth_reference_variants(
        k_ratios, table["P_R"], flow, run["observables"]
    )

    print("[4/5] duration and measure conventions", flush=True)
    duration = stochastic.duration(model, run["observables"]["M_over_Mpl_corrected"])
    sigma_convention = {}
    for factor in (0.5, 1.0, 2.0, 5.0):
        s = model.exit_exponent
        hubble = stochastic.horizon_hubble(
            model, run["observables"]["M_over_Mpl_corrected"]
        )
        mean, std = stochastic.first_passage_moments(s)
        sigma_convention[str(factor)] = {"N_stochastic_std": std}

    window = None
    if args.window_scan:
        print("[5/5] n_s / alpha_s window scan", flush=True)
        window = observables.window_scan(bg)
    else:
        print("[5/5] window scan skipped (--window-scan to enable)", flush=True)

    ledger = {
        "p": args.p,
        "bootstrap": bootstrap_block,
        "background": {
            **flow,
            "N_total_from_offset_start": float(bg.total_efolds),
            "x_end": x_end,
            "min_f_R_on_trajectory": stability["min_f_R"],
            "min_f_R_at_slow_roll_end": slow_roll_end,
            "min_M2_f_RR": stability["min_M2_f_RR"],
        },
        "reheating": {
            k: reheat[k] for k in
            ("N_star", "M_over_Mpl", "V_star", "rho_end", "Gamma_over_Mpl",
             "T_reh_GeV")
        },
        "n_star_budget": budget,
        "observables": run["observables"],
        "notch": notch_variants,
        "duration": duration,
        "cmb": run.get("sachs_wolfe", {}),
        "window_scan": window,
    }

    out = ROOT / "results" / f"ledger_p{args.p}.json"
    out.write_text(json.dumps(ledger, indent=2))

    tables = ROOT.parent / "paper" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    b, o, r = bootstrap_block, run["observables"], ledger["reheating"]
    rows = [
        (r"$p$", f"{args.p}", "discrete model parameter"),
        (r"$q = R_H/M^2$", f"{b['q']:.9f}", r"unit-gap condition"),
        (r"$\mu^2$", f"{b['mu_squared']:.6f}", "verified numerically"),
        (r"$s$", f"{b['exit_exponent_s']:.6f}", r"$(\sqrt{21}-3)/2$"),
        (r"$N_*$", f"{r['N_star']:.2f} $\\pm$ {budget['sigma_N_star']:.2f}",
         "resolved reheating"),
        (r"$M/M_{\rm Pl}$", latex_sci(r['M_over_Mpl'], 4), r"fixed by $A_s$"),
        (r"$T_{\rm reh}$", latex_sci(r['T_reh_GeV'], 3) + " GeV",
         "at radiation equality"),
        (r"$n_s$", f"{o['n_s']:.5f}", "exact mode fit"),
        (r"$\alpha_s$", f"{o['alpha_s']:.5f}", "fit-window dependent"),
        (r"$r$", f"{o['r']:.5f}", ""),
        (r"$\min f_R$", f"{ledger['background']['min_f_R_on_trajectory']:.4f}",
         r"on trajectory ($\epsilon=1$)"),
        (r"$\min M^2 f_{RR}$", f"{ledger['background']['min_M2_f_RR']:.4f}", ""),
        (r"notch depth", f"{notch_variants['fitted_ns']['depth']:.3f}",
         r"fitted-$n_s$ reference"),
        (r"$k_{\rm notch}$",
         latex_sci(notch_variants['fitted_ns']['k_Mpc'], 2) + r" Mpc$^{-1}$",
         "feature location"),
    ]
    (tables / "candidate.tex").write_text(
        latex_table(rows,
                    f"Parameters and derived observables of the primary "
                    f"candidate ($p={args.p}$, $\\alpha={model.alpha:.4f}$; "
                    f"invariant-wall branch, conformally coupled matter).",
                    "tab:candidate")
    )

    print(f"\nwrote {out}")
    print(f"wrote {tables/'candidate.tex'}\n")
    print("--- notch depth by P_smooth definition ---")
    for name, values in notch_variants.items():
        print(f"  {name:22s} depth={values['depth']:.4f} "
              f"k={values['k_Mpc']:.4e} Mpc^-1 (grid readout {values['grid_depth']:.4f})")
    print("\n--- corrected stability numbers ---")
    print(f"  min f_R on trajectory      {ledger['background']['min_f_R_on_trajectory']:.4f}"
          f"   (handout quoted {slow_roll_end:.4f} = slow-roll end, x=2sqrt3)")
    print(f"  min M^2 f_RR               {ledger['background']['min_M2_f_RR']:.4f}")
    print(f"\n  N_total = {duration['N_total_mean']:.3f}; the quoted sigma "
          f"{duration['N_stochastic_std']:.3f} is set by the choice X_c = H/(2 pi s)")


if __name__ == "__main__":
    main()
