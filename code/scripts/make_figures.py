"""Generate every figure in Paper I from stored results.

Like the tables, figures are build artifacts: nothing is drawn by hand.
Styling follows standard journal conventions: serif/Computer Modern math
fonts matching the SVJour3 body text, a colorblind-safe palette, panel
labels, and no in-axes titles.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIGURES = ROOT.parent / "paper" / "figures"

# Okabe-Ito colorblind-safe palette
BLUE, VERMILLION, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "figure.dpi": 300,
})

QUADRUPOLE_K = 2.5 / 13870.0   # (l + 1/2)/chi_*  at l = 2


def fig_wall():
    """S(z) for the wall family."""
    z = np.linspace(0.0, 1.0, 400)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for alpha, label, color, style, width in (
        (2.809, r"$d\phi$ ($\alpha=2.81$)", BLUE, "-", 1.5),
        (3.0, r"$dz$ ($\alpha=3$)", VERMILLION, "--", 1.1),
        (2.0, r"$d\ln R$ ($\alpha=2$)", GREEN, "-.", 1.1),
        (0.0, r"$\alpha=0$", GREY, ":", 1.1),
    ):
        ax.plot(z, (1 - z) * (1 + alpha * z), style, color=color,
                lw=width, label=label)
    ax.axhline(1.0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(r"$z=(R/R_H)^{\,p}$")
    ax.set_ylabel(r"$S(z)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.45)
    ax.legend(loc="lower left", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "wall.pdf")
    print("wrote figures/wall.pdf")


def fig_transfer():
    """Scalar transfer for the viable invariant-wall rungs; tensor for p=66."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for p, color, style in ((65, GREEN, "--"), (66, BLUE, "-"),
                            (67, VERMILLION, "-.")):
        d = np.genfromtxt(ROOT / "results" / f"invwall_p{p}" /
                          "primordial_transfer.csv", delimiter=",", names=True)
        k = d["k_over_kpivot"] * 0.05
        ax.plot(k, d["scalar_transfer"], style, color=color,
                label=fr"scalar, $p={p}$")
    d66 = np.genfromtxt(ROOT / "results" / "invwall_p66" /
                        "primordial_transfer.csv", delimiter=",", names=True)
    ax.plot(d66["k_over_kpivot"] * 0.05, d66["tensor_transfer"],
            color=GREY, lw=0.9, label=r"tensor, $p=66$")
    ax.axvline(QUADRUPOLE_K, color="0.8", lw=0.7, zorder=0)
    ax.annotate(r"$\ell=2$", xy=(QUADRUPOLE_K, 0.435),
                xytext=(QUADRUPOLE_K * 1.25, 0.435),
                color="0.4", fontsize=7.5, va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(2e-5, 5e-1)
    ax.set_ylim(0.4, 1.45)
    ax.set_xlabel(r"$k\;[\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P(k)/P^{\rm smooth}(k)$")
    ax.legend(loc="upper right", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "transfer.pdf")
    print("wrote figures/transfer.pdf")


def fig_clratios():
    """Lensed C_l ratios for the broad/narrow invariant rungs vs cosmic variance."""
    from frozen_horizon import boltzmann

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), sharey=True)
    ells = np.arange(2, 31)

    cv = np.sqrt(2.0 / (2.0 * ells + 1.0))
    for ax, panel in zip(axes, ("(a)  $TT$", "(b)  $EE$")):
        ax.fill_between(ells, 1 - cv, 1 + cv, color="0.93", zorder=0)
        ax.axhline(1.0, color="0.8", lw=0.6)
        ax.text(0.04, 0.92, panel, transform=ax.transAxes, va="top",
                fontsize=9)

    for p, color, style, label in (
        (65, GREEN, "--", r"$p=65$ (broad)"),
        (66, BLUE, "-", r"$p=66$ (narrow)"),
    ):
        cache = ROOT / "results" / f"invwall_p{p}" / "camb_ratio.npz"
        if not cache.exists():
            d = np.genfromtxt(ROOT / "results" / f"invwall_p{p}" /
                              "primordial_power.csv", delimiter=",", names=True)
            obs = json.loads((ROOT / "results" / f"invwall_p{p}" /
                              "summary.json").read_text())["observables"]
            ratio, _, _ = boltzmann.feature_ratio(
                d["k_Mpc"], d["P_R"], obs["n_s"], obs["alpha_s"], lmax=300)
            np.savez(cache, **ratio)
        data = np.load(cache)
        axes[0].plot(ells, data["TT"][2:31], style, color=color, label=label)
        axes[1].plot(ells, data["EE"][2:31], style, color=color, label=label)

    # legend entry for the band, once
    axes[0].fill_between([], [], [], color="0.93", label="cosmic variance")
    axes[0].set_ylabel(r"$C_\ell/C_\ell^{\rm smooth}$")
    for ax in axes:
        ax.set_xlabel(r"$\ell$")
        ax.set_xlim(2, 30)
    axes[0].set_ylim(0.55, 1.25)
    axes[0].legend(loc="lower right", handlelength=2.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "clratios.pdf")
    print("wrote figures/clratios.pdf")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_wall()
    fig_transfer()
    fig_clratios()


if __name__ == "__main__":
    main()
