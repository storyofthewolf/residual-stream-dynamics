"""toy_plots.py — Figures for toy transformers trained on in-context Markov chains.

Functions take a history dict from toy_training and return a Figure; saving via
_save (creates parent dirs, dpi=150).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1–3
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


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


def plot_training_stages(hist: dict, title: str, save_path=None):
    """Three panels: loss vs references; KL to unigram/bigram; attention signatures."""
    step = np.maximum(hist["step"], 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

    ax = axes[0]
    ax.plot(step, hist["xent"], color=C1, lw=2, label="Model")
    for key, lab, ls in (("xent_uniform", "Uniform", ":"), ("xent_unigram", "In-context unigram", "--"),
                         ("xent_bigram", "Bayes-optimal bigram", "-.")):
        ax.axhline(hist[key][0], color=MUTED, lw=1.2, ls=ls)
        ax.text(step[-1], hist[key][0], f" {lab}", color=MUTED, va="center", fontsize=8)
    ax.set_ylim(hist["xent_bigram"][0] - 0.05, hist["xent_uniform"][0] + 0.1)
    ax.set_ylabel("Cross-entropy (bits/token)", color=INK)
    ax.set_title("Loss vs. reference strategies", color=INK, fontsize=11)

    ax = axes[1]
    ax.plot(step, hist["kl_unigram"], color=C2, lw=2, label="KL(unigram ‖ model)")
    ax.plot(step, hist["kl_bigram"], color=C1, lw=2, label="KL(bigram ‖ model)")
    ax.set_yscale("log")
    ax.set_ylabel("KL divergence (bits)", color=INK)
    ax.set_title("Which strategy does the model match?", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.plot(step, hist["prev_token_score"], color=C3, lw=2, label="Layer 0: previous-token")
    ax.plot(step, hist["induction_score"], color=C1, lw=2, label="Layer 1: statistical induction")
    ax.plot(step, hist["induction_baseline"], color=C1, lw=1, ls=":", label="Induction, uniform-attention baseline")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Attention score", color=INK)
    ax.set_title("Attention signatures", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Training step", color=INK)
        _style(ax)
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_attention_patterns(patterns: list[np.ndarray], tokens: np.ndarray, title: str,
                            n_show: int = 30, save_path=None):
    """Attention pattern per layer for one sequence (first n_show positions)."""
    fig, axes = plt.subplots(1, len(patterns), figsize=(5.2 * len(patterns), 4.8))
    axes = np.atleast_1d(axes)
    labels = [str(t) for t in tokens[:n_show]]
    for i, (ax, A) in enumerate(zip(axes, patterns)):
        im = ax.imshow(A[:n_show, :n_show], cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Layer {i}", color=INK, fontsize=11)
        ax.set_xlabel("Key position (token)", color=INK)
        ax.set_ylabel("Query position (token)", color=INK)
        ax.set_xticks(range(n_show)); ax.set_xticklabels(labels, fontsize=6)
        ax.set_yticks(range(n_show)); ax.set_yticklabels(labels, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, label="Attention weight")
    fig.suptitle(title, color=INK, fontsize=12)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
