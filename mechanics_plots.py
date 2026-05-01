"""mechanics_plots.py — Visualization for residual stream mechanical quantities.

Consumes MechanicsRecords from mechanics_compute.py.
No model, no torch, no forward passes — pure visualization.

Pipeline position:
    extraction.py -> mechanics_compute.py -> VISUALIZATION (this file)

Plot functions:
    Corpus / statistical:
        plot_mechanics_overview()   — five-panel mean +/- 1σ across all corpus pairs
        plot_mechanics_category()   — per-category base vs contrast, pair curves + mean
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mechanics_compute import MechanicsRecord


# ============================================================================
# SHARED STYLE
# ============================================================================

PAIR_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors[:5])

# (record attribute, x-axis label, y-axis label, title, y-limits or None)
_PANELS = [
    ("speed",
     "Layer transition", "||ΔX_l||₂", "Speed", None),
    ("acceleration_magnitude",
     "Layer transition", "||ΔV_l||₂", "Acceleration magnitude", None),
    ("cosine_sim_state",
     "Layer transition", "cos(X_l, X_{l+1})", "State cosine similarity", (-1.05, 1.05)),
    ("cosine_sim_update_state",
     "Layer transition", "cos(ΔX_l, X_l)", "Update–state alignment", (-1.05, 1.05)),
    ("cosine_sim_update_update",
     "Layer transition", "cos(ΔX_l, ΔX_{l+1})", "Update coherence", (-1.05, 1.05)),
]


def _save(fig, path: Path, filename: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    out = path / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def _build_title(base: str, model_name: str) -> str:
    return f"{base}  [{model_name}]" if model_name else base


# ============================================================================
# CORPUS: MEAN +/- 1 SIGMA ACROSS ALL PAIRS
# ============================================================================

def plot_mechanics_overview(
    records:    list,
    output_dir: Path,
    corpus_tag: str,
    run_tag:    str,
    model_name: str = "",
) -> None:
    """
    Five-panel overview: one subplot per mechanical quantity.
    Each subplot: mean +/- 1σ at the final token position, base vs contrast.

    Mirrors plot_overall_mean() in entropy_plots.py.

    Args:
        records:    list of MechanicsRecord
        output_dir: directory for saved figure
        corpus_tag: corpus name (used in filename)
        run_tag:    optional runtime tag to avoid filename collisions
        model_name: used in figure title and filename
    """
    if not records:
        print("  plot_mechanics_overview: no records, skipping")
        return

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        _build_title("Residual stream mechanics — all categories (mean ± 1σ, final token)",
                     model_name),
        fontsize=11,
    )

    for ax, (attr, xlabel, ylabel, title, ylim) in zip(axes, _PANELS):
        for role, color, ls in [("base",     "#1565C0", "-"),
                                  ("contrast", "#B71C1C", "--")]:
            curves = [getattr(r, attr) for r in records if r.role == role]
            if not curves:
                continue
            arr  = np.array(curves)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            xs   = list(range(len(mean)))
            ax.plot(xs, mean, color=color, linestyle=ls,
                    linewidth=2.0, label=f"{role} mean")
            ax.fill_between(xs, mean - std, mean + std,
                            color=color, alpha=0.15)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(getattr(records[0], attr))))
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
            if ylim[1] > 1.0:
                ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    axes[0].legend(fontsize=8)
    plt.tight_layout()

    suffix = f"_{model_name}" if model_name else ""
    _save(fig, output_dir,
          f"mechanics_overview{suffix}_{corpus_tag}{run_tag}.png")


# ============================================================================
# CORPUS: PER-CATEGORY BASE VS CONTRAST
# ============================================================================

def plot_mechanics_category(
    records:    list,
    category:   str,
    output_dir: Path,
    corpus_tag: str,
    run_tag:    str,
    model_name: str = "",
) -> None:
    """
    Per-category five-panel plot.
    Individual pair curves shown as faint lines; bold lines show the mean.
    Base=solid, contrast=dashed.

    Mirrors plot_category() in entropy_plots.py.

    Args:
        records:    list of MechanicsRecord
        category:   corpus category to filter to
        output_dir: directory for saved figure
        corpus_tag: corpus name (used in filename)
        run_tag:    optional runtime tag
        model_name: used in figure title and filename
    """
    cat_records = [r for r in records if r.category == category]
    if not cat_records:
        print(f"  plot_mechanics_category: no records for category='{category}', skipping")
        return

    by_pair: dict = defaultdict(dict)
    for r in cat_records:
        if r.pair_id and r.role:
            by_pair[r.pair_id][r.role] = r

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        _build_title(f"Residual stream mechanics — {category} (final token)", model_name),
        fontsize=11,
    )

    for ax, (attr, xlabel, ylabel, title, ylim) in zip(axes, _PANELS):
        all_base:     list = []
        all_contrast: list = []

        for pair_idx, (pair_id, roles) in enumerate(sorted(by_pair.items())):
            color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
            for role, ls in [("base", "-"), ("contrast", "--")]:
                if role not in roles:
                    continue
                curve = getattr(roles[role], attr)
                ax.plot(range(len(curve)), curve,
                        color=color, linestyle=ls, linewidth=1.0, alpha=0.5)
                (all_base if role == "base" else all_contrast).append(curve)

        if all_base:
            mean_b = np.mean(all_base, axis=0)
            ax.plot(range(len(mean_b)), mean_b,
                    color="black", linestyle="-", linewidth=2.2,
                    label="mean base", zorder=5)
        if all_contrast:
            mean_c = np.mean(all_contrast, axis=0)
            ax.plot(range(len(mean_c)), mean_c,
                    color="dimgray", linestyle="--", linewidth=2.2,
                    label="mean contrast", zorder=5)

        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ref_curve = getattr(cat_records[0], attr)
        ax.set_xticks(range(len(ref_curve)))
        ax.grid(alpha=0.3)
        if ylim is not None:
            ax.set_ylim(*ylim)
            ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
            if ylim[1] > 1.0:
                ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    axes[0].legend(fontsize=7)
    plt.tight_layout()

    suffix = f"_{model_name}" if model_name else ""
    _save(fig, output_dir,
          f"mechanics_{category}{suffix}_{corpus_tag}{run_tag}.png")
