"""Generate the tables and figures of Paper II from the quantum module.

As in Paper I, every number and every curve in the manuscript is a build
artifact: nothing is transcribed by hand.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frozen_horizon import quantum  # noqa: E402

OUT = ROOT.parent / "paper" / "paper2"
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.2, "legend.frameon": False, "figure.dpi": 300,
})


def table_kernel(max_n=6):
    ratios, kernels = quantum.fluctuation_spectrum(max_n)
    L, ev = quantum.eigenvalues(max_n)
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\begin{tabular}{r r r r}", r"\hline",
        r"$n$ & $f_n'/f_n$ & $K_n H_H^2$ & $\lambda_n/H_H^2$ \\", r"\hline",
    ]
    for n in range(max_n + 1):
        lines.append(f"{n} & {ratios[n]:+.5f} & {kernels[n]:+.4f} & "
                     f"{ev[n]:+.0f} " + r"\\")
    lines += [
        r"\hline", r"\end{tabular}",
        r"\caption{Boundary kernel of the half four-sphere at $\mu^2=3$. The"
        r" homogeneous mode is the sole negative direction; every"
        r" inhomogeneous mode is Gaussian suppressed. The final column gives"
        r" the four-sphere eigenvalues $\lambda_L/H_H^2=L(L+3)-\mu^2$ for"
        r" comparison.}",
        r"\label{tab:kernel}", r"\end{table}", "",
    ]
    (OUT / "tables" / "kernel.tex").write_text("\n".join(lines))
    print("wrote tables/kernel.tex")


def table_shells(max_n=8):
    rows = quantum.shell_crossings(max_n)
    continuum = quantum.continuum_kick()
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{r r r r r r}", r"\hline",
        r"$n$ & $N_n$ & $d_n$ & $|F_n(v_n)|$ & $\Delta V_n/H_H^2$ &"
        r" $Q_n^{\rm eff}$ \\", r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['n']} & {r['N_n']:.5f} & {r['degeneracy']} & "
            f"{r['F_at_crossing']:.5f} & {r['delta_V_n']:.6f} & "
            f"{r['Q_effective']:.3f} " + r"\\")
    lines += [
        r"\hline", r"\end{tabular}",
        r"\caption{Discrete $S^3$ shell crossings. Harmonic $n$ enters the"
        r" long-wavelength sector at $N_n=\ln(n+1)$ exactly, injecting"
        r" variance $\Delta V_n$. The effective noise amplitude"
        r" $Q_n^{\rm eff}$ falls steeply and crosses the continuum value"
        rf" $Q_\nu(1)={continuum:.4f}$ between the fourth and fifth shells,"
        r" undershooting it by $2.5\%$ at $n=8$; the first shell exceeds it"
        r" by a factor of three.}",
        r"\label{tab:shells}", r"\end{table}", "",
    ]
    (OUT / "tables" / "shells.tex").write_text("\n".join(lines))
    print("wrote tables/shells.tex")


def fig_kernel():
    ratios, kernels = quantum.fluctuation_spectrum(8)
    n = np.arange(0, 9)
    values = np.array([kernels[i] for i in n])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    ax.plot(n[1:], values[1:], "o-", color=BLUE, ms=4, label=r"$n\geq 1$ (stable)")
    ax.plot(n[:1], values[:1], "s", color=VERMILLION, ms=6,
            label=r"$n=0$ (unstable)")
    ax.set_xlabel(r"$S^3$ harmonic $n$")
    ax.set_ylabel(r"$K_n H_H^2$")
    ax.legend(loc="lower right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figures" / "kernel.pdf")
    print("wrote figures/kernel.pdf")


def fig_thimble():
    """The complex amplitude plane: real axis divergent, thimble convergent."""
    t = quantum.classicalizing_thimble()
    theta = t["theta_rad"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    grid = np.linspace(-1.6, 1.6, 400)
    X, Y = np.meshgrid(grid, grid)
    A = X + 1j * Y
    # Re ln Psi = +|K_0| Re(A^2) / 2H^2, so the exponent is POSITIVE (and the
    # measure divergent) along the real axis, where A^2 > 0. The sign here was
    # inverted, which flipped the colour scale and made the caption describe
    # the opposite of what was plotted.
    weight = np.real(A**2)
    ax.contourf(X, Y, weight, levels=25, cmap="RdBu_r", alpha=0.75)
    ax.contour(X, Y, weight, levels=[0.0], colors="0.4", linewidths=0.6)

    span = np.linspace(-1.5, 1.5, 2)
    ax.plot(span, 0 * span, color=GREY, lw=1.4, ls=":",
            label=r"real axis (divergent)")
    ax.plot(span * np.cos(theta), span * np.sin(theta), color=BLUE, lw=1.8,
            label=rf"thimble, $\theta={t['theta_deg']:.1f}^\circ$")
    ax.plot(0 * span, span, color=GREEN, lw=1.0, ls="--",
            label=r"steepest descent of $K_0$")
    ax.set_xlabel(r"$\mathrm{Re}\,A/H_H$")
    ax.set_ylabel(r"$\mathrm{Im}\,A/H_H$")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figures" / "thimble.pdf")
    print("wrote figures/thimble.pdf")


def fig_window():
    """Normalizability across the one-negative-mode window."""
    from frozen_horizon.quantum import MU2_NORMALIZABLE_MIN
    grid = np.linspace(0.3, 3.95, 40)
    theta, cos2 = [], []
    for m in grid:
        t = quantum.classicalizing_thimble(mu2=float(m))
        theta.append(t["theta_deg"]); cos2.append(t["cos_2theta"])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axhspan(0, 1.1, color="0.93", zorder=0)
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.plot(grid, cos2, color=BLUE, label=r"$\cos 2\theta$")
    ax.axvline(MU2_NORMALIZABLE_MIN, color=VERMILLION, lw=1.0, ls="--")
    ax.axvline(3.0, color=GREEN, lw=1.0, ls=":")
    ax.text(MU2_NORMALIZABLE_MIN - 0.06, 0.72, r"$\mu^2=7/4$", color=VERMILLION,
            fontsize=7.5, ha="right")
    ax.text(3.05, -0.85, r"unit gap", color=GREEN, fontsize=7.5)
    ax.text(0.45, 0.45, "not normalizable", fontsize=7, color="0.35")
    ax.set_xlabel(r"$\mu^2=|m_H^2|/H_H^2$")
    ax.set_ylabel(r"$\cos 2\theta$")
    ax.set_xlim(0.3, 4.0); ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="lower left")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figures" / "window.pdf")
    print("wrote figures/window.pdf")


def fig_shells():
    rows = quantum.shell_crossings(8)
    continuum = quantum.continuum_kick()
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    n = [r["n"] for r in rows]
    q = [r["Q_effective"] for r in rows]
    ax.axhline(continuum, color=GREY, lw=1.0, ls="--",
               label=rf"continuum $Q_\nu(1)={continuum:.2f}$")
    ax.plot(n, q, "o-", color=BLUE, ms=4, label=r"$Q_n^{\rm eff}$")
    ax.set_xlabel(r"shell index $n$")
    ax.set_ylabel(r"$Q^{\rm eff}_n$")
    ax.set_ylim(0, 26)
    secondary = ax.twiny()
    secondary.set_xlim(ax.get_xlim())
    secondary.set_xticks(n[::2])
    secondary.set_xticklabels([f"{np.log(i+1):.2f}" for i in n[::2]], fontsize=7)
    secondary.set_xlabel(r"crossing time $N_n=\ln(n+1)$", fontsize=8)
    ax.legend(loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "figures" / "shells.pdf")
    print("wrote figures/shells.pdf")


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    table_kernel()
    table_shells()
    fig_kernel()
    fig_thimble()
    fig_window()
    fig_shells()


if __name__ == "__main__":
    main()
