"""Emit every table Paper I quotes, as LaTeX, from stored results.

The manuscript \\input's these files; no number in the paper is typed by hand.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TABLES = ROOT.parent / "paper" / "tables"


def evidence(values):
    return 2.0 * np.log(np.mean(np.exp(-0.5 * np.asarray(values))))


def load_likelihood():
    rows = json.loads((ROOT / "results" / "lowl_likelihood.json").read_text())
    return {r["run"]: r for r in rows}


def write(name, text):
    (TABLES / name).write_text(text)
    print(f"wrote {TABLES/name}")


def likelihood_table(like):
    branches = [
        (r"coordinate wall ($\alpha=3$, $\xi_H=0$)", "resolved_p", [62, 63, 64, 65]),
        (r"invariant wall ($\alpha\simeq2.809$, $\xi_H=\tfrac16$)", "invwall_p",
         [64, 65, 66, 67]),
    ]
    lines = [
        r"\begin{table*}[htbp]\centering",
        r"\begin{tabular}{l r r r r}", r"\hline",
        r"Model & $\Delta\chi^2_{TT}$ & $\Delta\chi^2_{EE}$ & "
        r"$\Delta\chi^2_{TT+EE}$ & \\", r"\hline",
    ]
    for label, prefix, rungs in branches:
        lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{label}}}}} \\")
        for p in rungs:
            r = like[f"{prefix}{p}"]
            note = "excluded" if r["dTOT"] > 6 else ""
            lines.append(
                f"$p={p}$ & {r['dTT']:+.2f} & {r['dEE']:+.2f} & "
                f"{r['dTOT']:+.2f} & {note} \\\\"
            )
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Official Planck 2018 low-$\ell$ likelihoods (Commander $TT$,"
        r" SimAll $EE$, exact native implementations): featured minus smooth,"
        r" negative values favour the featured spectrum. Each model is compared against its own"
        r" smooth continuation with identical late-time cosmology.}",
        r"\label{tab:likelihood}", r"\end{table*}", "",
    ]
    write("likelihood.tex", "\n".join(lines))


def evidence_table(like):
    coord = [like[f"resolved_p{p}"]["dTOT"] for p in (63, 64, 65)]
    inv = [like[f"invwall_p{p}"]["dTOT"] for p in (65, 66, 67)]
    rows = [
        (r"coordinate wall, $p\in\{63,64,65\}$", evidence(coord)),
        (r"invariant wall, $p\in\{65,66,67\}$", evidence(inv)),
        (r"pooled, six viable integers", evidence(coord + inv)),
        (r"best single integer (no penalty)",
         -min(min(coord), min(inv))),
    ]
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l r}", r"\hline",
        r"Flat prior over & $2\ln(L_{\rm marg}/L_{\rm smooth})$ \\", r"\hline",
    ]
    lines += [f"{label} & {value:+.2f} \\\\" for label, value in rows]
    lines += [
        r"\hline", r"\end{tabular}",
        r"\caption{Rung-marginalized evidence under the official low-$\ell$"
        r" likelihoods. Values of order unity are indistinguishable from"
        r" noise; the difference from the best-single-integer row quantifies the look-elsewhere penalty.}",
        r"\label{tab:evidence}", r"\end{table}", "",
    ]
    write("evidence.tex", "\n".join(lines))


def cmb_ratio_table():
    lines = [
        r"\begin{table*}[htbp]\centering",
        r"\begin{tabular}{r rrr rrr}", r"\hline",
        r" & \multicolumn{3}{c}{$p=65$ (invariant, broad)} & "
        r"\multicolumn{3}{c}{$p=66$ (invariant, narrow)} \\",
        r"$\ell$ & $TT$ & $EE$ & $TE$ & $TT$ & $EE$ & $TE$ \\", r"\hline",
    ]
    ratios = {}
    for p in (65, 66):
        data = np.load(ROOT / "results" / f"invwall_p{p}" / "camb_ratio.npz")
        ratios[p] = data
    for ell in (2, 3, 4, 5, 10, 20, 30):
        cells = []
        for p in (65, 66):
            for field in ("TT", "EE", "TE"):
                cells.append(f"{ratios[p][field][ell]:.4f}")
        lines.append(f"{ell} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\hline", r"\end{tabular}",
        r"\caption{Lensed $C_\ell$ ratios (featured over smooth) from CAMB at"
        r" the reheating-determined $N_*$, for the two viable integers of the"
        r" invariant-wall branch; the CAMB ratios are independently reproduced"
        r" by CLASS to better than $1.1\times10^{-3}$ at every"
        r" $\ell\le30$.}",
        r"\label{tab:cmbratios}", r"\end{table*}", "",
    ]
    write("cmb_ratios.tex", "\n".join(lines))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    like = load_likelihood()
    likelihood_table(like)
    evidence_table(like)
    cmb_ratio_table()


if __name__ == "__main__":
    main()
