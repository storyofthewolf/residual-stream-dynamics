"""markov_plots.py — Figures for the Markov entropy-rate analysis.

All functions take arrays from markov_compute and return a Figure; saving goes
through _save (creates parent dirs, dpi=150). No file I/O beyond that.

Colors: categorical slot 1 (blue) = Markov sequence, slot 2 (orange) =
shuffled control; entropy levels use a single-hue blue ramp (light = low H).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C_BASE, C_SHUF = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
BLUE_RAMP = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]


def _save(fig, save_path):
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_rho_vs_layer(panels: list[dict], model_name: str, save_path=None):
    """
    panels: list of {"title", "base": spearman dict, "shuf": spearman dict}
    Spearman ρ(H, residual entropy) per layer with 95% pair-bootstrap bands.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        x = np.arange(len(p["base"]["rho"]))
        for key, c, lab in (("base", C_BASE, "Markov"), ("shuf", C_SHUF, "Shuffled")):
            s = p[key]
            ax.fill_between(x, s["ci_lo"], s["ci_hi"], color=c, alpha=0.18, lw=0)
            ax.plot(x, s["rho"], color=c, lw=2, marker="o", ms=4, label=lab)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(p["title"], color=INK, fontsize=11)
        ax.set_xlabel("Layer", color=INK)
        _style(ax)
    axes[0].set_ylabel("Spearman ρ (H, residual entropy)", color=INK)
    axes[0].legend(frameon=False)
    fig.suptitle(f"{model_name}: does residual entropy track the source entropy rate?",
                 color=INK, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_resid_by_level(levels, base_mean, shuf_mean, model_name, title, save_path=None):
    """Residual entropy vs layer, one line per H level (Markov); shuffled pooled mean dashed."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(base_mean.shape[1])
    for i, h in enumerate(levels):
        ax.plot(x, base_mean[i], color=BLUE_RAMP[i % len(BLUE_RAMP)], lw=2,
                label=f"H = {h:.2f}")
    ax.plot(x, shuf_mean.mean(axis=0), color=C_SHUF, lw=2, ls="--", label="Shuffled (all H)")
    ax.set_xlabel("Layer", color=INK)
    ax.set_ylabel("Residual entropy, energy norm (bits)", color=INK)
    ax.set_title(f"{model_name}, {title}", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_output_vs_rate(panels: list[dict], model_name: str, save_path=None):
    """
    panels: list of {"title", "levels", "H_alpha": {base, shuf}, "xent": {...}, "mass": {...}}
    Each value is a level_means_with_ci dict. Rows: lengths; cols: measures.
    """
    measures = [("H_alpha", "Output entropy over the 20 words (bits)"),
                ("xent", "Cross-entropy on next word (bits)"),
                ("mass", "Probability on the 20 words")]
    fig, axes = plt.subplots(len(panels), 3, figsize=(14, 3.8 * len(panels)), squeeze=False)
    for r, p in enumerate(panels):
        for c, (key, lab) in enumerate(measures):
            ax = axes[r, c]
            for role, col, name in (("base", C_BASE, "Markov"), ("shuf", C_SHUF, "Shuffled")):
                s = p[key][role]
                ax.fill_between(s["levels"], s["ci_lo"], s["ci_hi"], color=col, alpha=0.18, lw=0)
                ax.plot(s["levels"], s["mean"], color=col, lw=2, marker="o", ms=5, label=name)
            if key != "mass":
                lv = p[key]["base"]["levels"]
                ax.plot(lv, lv, color=MUTED, lw=1, ls=":", label="y = H (true rate)")
            else:
                ax.set_ylim(0, 1.02)
            ax.set_xlabel("True entropy rate H (bits/token)", color=INK)
            ax.set_ylabel(lab, color=INK, fontsize=9)
            ax.set_title(p["title"], color=INK, fontsize=10)
            _style(ax)
        axes[r, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{model_name}: final-layer output vs. source entropy rate "
                 f"(second-half positions, 95% CI)", color=INK, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
