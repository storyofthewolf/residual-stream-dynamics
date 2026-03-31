"""ablation_plots.py — Visualization for ablation results.

Consumes AblationRecords from ablation_compute.py.
No model, no torch, no forward passes — pure visualization.

Pipeline position (parallel to the entropy visualization path):
    extraction.py → ablation_compute.py → VISUALIZATION (this file)

Plot functions:
    Posthoc ablation (Stage 1):
        plot_kl_vs_layer()            — KL divergence vs layer, one panel per k
        plot_kl_vs_k()                — KL divergence vs k, one panel per layer
        plot_top1_preservation()      — top-1 token preservation vs layer
        plot_entropy_change_vs_layer()— entropy change vs layer

    Intervention ablation (Stage 2):
        plot_intervention_heatmap()   — 2D heatmap of (layer, k) effects

All functions accept a list of AblationRecord objects and filter internally
to the appropriate ablation_type (posthoc for Figures 1-4, intervention
for Figure 5). All return the matplotlib Figure object.

Visual conventions match entropy_plots.py:
    - Base prompts: solid blue (#1f77b4)
    - Contrast prompts: dashed red (#d62728)
    - Shaded bands: mean ± 1 SEM, alpha=0.2
    - Grid: True, alpha=0.3
    - Titles: "Figure title  [model_name]"
    - Save: PNG at 150 DPI
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# STYLE CONSTANTS
# Match entropy_plots.py conventions for visual consistency.
# ============================================================================

COLOR_BASE     = "#1f77b4"
COLOR_CONTRAST = "#d62728"
STYLE_BASE     = {"color": COLOR_BASE,     "linestyle": "-",  "linewidth": 2.0}
STYLE_CONTRAST = {"color": COLOR_CONTRAST, "linestyle": "--", "linewidth": 2.0}
GRID_ALPHA     = 0.3
BAND_ALPHA     = 0.2
DPI            = 150


def _safe_model_name(model_name: str) -> str:
    """Replace spaces and hyphens with underscores for filenames."""
    return model_name.replace(" ", "_").replace("-", "_")


def _save(fig, save_path: str) -> None:
    """Save figure and close."""
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _panel_title(k: int, ev_dict: dict | None) -> str:
    """Panel subtitle showing k and optional explained variance."""
    if ev_dict is not None and k in ev_dict:
        return f"k={k}  (EV={ev_dict[k]*100:.1f}%)"
    return f"k={k}"


def _panel_layout(n_panels: int):
    """
    Determine (n_rows, n_cols) for multi-panel figures.
    Single row for ≤4 panels, otherwise two rows.
    """
    if n_panels <= 4:
        return 1, n_panels
    n_cols = (n_panels + 1) // 2
    return 2, n_cols


# ============================================================================
# DATA AGGREGATION HELPER
# Groups AblationRecords by (role, k) and computes mean ± SEM
# across prompts for each layer. Called internally by all plot functions.
# ============================================================================

def _aggregate_ablation_records(
    records:       list,
    ablation_type: str = "posthoc",
) -> dict:
    """
    Aggregate a list of AblationRecord objects into arrays suitable
    for plotting.

    Returns a nested dict:
        result[role][k] = {
            "kl_mean":   np.ndarray [n_layers],
            "kl_sem":    np.ndarray [n_layers],
            "ent_mean":  np.ndarray [n_layers],
            "ent_sem":   np.ndarray [n_layers],
            "top1_mean": np.ndarray [n_layers],   # fraction preserved
            "top1_sem":  np.ndarray [n_layers],
            "n_prompts": int,
        }

    role is "base" or "contrast" (from ActivationRecord.role).
    k is the integer subspace rank.
    Filters to ablation_type before aggregating.

    SEM = std / sqrt(n_prompts), computed across prompts for each
    (layer, k, role) cell.
    """
    # Filter to requested ablation type
    filtered = [r for r in records if r.ablation_type == ablation_type]

    # Group by (role, k)
    grouped = defaultdict(list)
    for r in filtered:
        grouped[(r.role, r.k)].append(r)

    result = defaultdict(dict)
    for (role, k), recs in grouped.items():
        n = len(recs)

        # Stack arrays: each is [n_layers] for posthoc
        kl_stack   = np.array([r.kl_divergence   for r in recs])   # [n_prompts, n_layers]
        ent_stack  = np.array([r.entropy_change   for r in recs])
        top1_stack = np.array([r.top1_preserved.astype(float) for r in recs])

        sqrt_n = np.sqrt(n) if n > 1 else 1.0

        result[role][k] = {
            "kl_mean":   kl_stack.mean(axis=0),
            "kl_sem":    kl_stack.std(axis=0, ddof=1) / sqrt_n if n > 1
                         else np.zeros_like(kl_stack.mean(axis=0)),
            "ent_mean":  ent_stack.mean(axis=0),
            "ent_sem":   ent_stack.std(axis=0, ddof=1) / sqrt_n if n > 1
                         else np.zeros_like(ent_stack.mean(axis=0)),
            "top1_mean": top1_stack.mean(axis=0),
            "top1_sem":  top1_stack.std(axis=0, ddof=1) / sqrt_n if n > 1
                         else np.zeros_like(top1_stack.mean(axis=0)),
            "n_prompts": n,
        }

    return dict(result)


# ============================================================================
# FIGURE 1: KL DIVERGENCE VS LAYER
# One panel per k value. Primary figure for the paper.
# ============================================================================

def plot_kl_vs_layer(
    records:    list,
    model_name: str,
    k_values:   list[int],
    ev_dict:    dict | None = None,
    save_path:  str | None  = None,
) -> plt.Figure:
    """
    KL divergence vs layer for posthoc ablation, one panel per k value.

    Primary figure. Tests whether base prompts diverge more from the full
    prediction than contrast prompts when r⊥ is removed, at each layer.
    The expected result: base > contrast, especially at the entropy
    crossover peak layer.

    Args:
        records:    list of AblationRecord (mixed types OK; filters to posthoc)
        model_name: used in title and filename
        k_values:   which k values to show as separate panels
        ev_dict:    optional {k: explained_variance_fraction} for panel titles
        save_path:  output file path, or None to skip saving

    Returns:
        matplotlib Figure object
    """
    agg = _aggregate_ablation_records(records, ablation_type="posthoc")
    n_rows, n_cols = _panel_layout(len(k_values))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, 5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Posthoc ablation: KL divergence vs layer  [{model_name}]",
        fontsize=12,
    )

    # Determine shared y-axis range across all panels
    y_max = 0.0
    for k in k_values:
        for cat in ["base", "contrast"]:
            if cat in agg and k in agg[cat]:
                vals = agg[cat][k]["kl_mean"] + agg[cat][k]["kl_sem"]
                y_max = max(y_max, vals.max())
    y_max = y_max * 1.1 if y_max > 0 else 1.0

    for idx, k in enumerate(k_values):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row][col]

        # Infer n_layers from whichever category is present
        n_layers = None
        for cat in ["base", "contrast"]:
            if cat in agg and k in agg[cat]:
                n_layers = len(agg[cat][k]["kl_mean"])
                break
        if n_layers is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        layers = np.arange(n_layers)

        for cat, style, label in [
            ("base",     STYLE_BASE,     "base"),
            ("contrast", STYLE_CONTRAST, "contrast"),
        ]:
            if cat not in agg or k not in agg[cat]:
                continue
            d = agg[cat][k]
            ax.plot(layers, d["kl_mean"], label=label, **style)
            ax.fill_between(layers,
                            d["kl_mean"] - d["kl_sem"],
                            d["kl_mean"] + d["kl_sem"],
                            color=style["color"], alpha=BAND_ALPHA)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(_panel_title(k, ev_dict), fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("KL divergence from full prediction (nats)")
        ax.set_xticks(layers)
        ax.set_ylim(bottom=0, top=y_max)
        ax.grid(alpha=GRID_ALPHA)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Hide unused axes
    total_axes = n_rows * n_cols
    for idx in range(len(k_values), total_axes):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        _save(fig, save_path)

    return fig


# ============================================================================
# FIGURE 2: KL DIVERGENCE VS K AT FIXED LAYERS
# One panel per fixed layer. Shows how degradation grows as k decreases.
# ============================================================================

def plot_kl_vs_k(
    records:      list,
    model_name:   str,
    fixed_layers: list[int],
    k_values:     list[int],
    ev_dict:      dict | None = None,
    save_path:    str | None  = None,
) -> plt.Figure:
    """
    KL divergence vs subspace rank k at fixed layers.

    Shows how prediction degradation grows as k decreases (more r⊥ removed),
    and whether base and contrast prompts diverge in their degradation curves.
    The falsifiable prediction: base curve lies above contrast curve,
    especially at layers near the entropy crossover peak.

    Args:
        records:      list of AblationRecord
        model_name:   used in title
        fixed_layers: one panel per layer (must be a parameter, not hardcoded)
        k_values:     x-axis values, must be sorted ascending
        ev_dict:      optional {k: EV fraction}; if provided, adds secondary x-axis
        save_path:    output file path, or None

    Returns:
        matplotlib Figure object
    """
    agg = _aggregate_ablation_records(records, ablation_type="posthoc")
    n_rows, n_cols = _panel_layout(len(fixed_layers))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, 5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Posthoc ablation: KL divergence vs subspace rank k  [{model_name}]",
        fontsize=12,
    )

    k_sorted = sorted(k_values)

    for idx, L in enumerate(fixed_layers):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row][col]

        for cat, style, label in [
            ("base",     STYLE_BASE,     "base"),
            ("contrast", STYLE_CONTRAST, "contrast"),
        ]:
            if cat not in agg:
                continue

            means = []
            sems  = []
            for k in k_sorted:
                if k in agg[cat]:
                    means.append(agg[cat][k]["kl_mean"][L])
                    sems.append(agg[cat][k]["kl_sem"][L])
                else:
                    means.append(np.nan)
                    sems.append(0.0)

            means = np.array(means)
            sems  = np.array(sems)
            ax.plot(k_sorted, means, label=label, **style)
            ax.fill_between(k_sorted,
                            means - sems, means + sems,
                            color=style["color"], alpha=BAND_ALPHA)

        ax.set_title(f"Layer {L}", fontsize=10)
        ax.set_xlabel("Subspace rank k  (← more ablation    less ablation →)")
        ax.set_ylabel("KL divergence from full prediction (nats)")
        ax.grid(alpha=GRID_ALPHA)
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

        # Add 99% explained variance vertical line if ev_dict provided
        if ev_dict is not None:
            for k in k_sorted:
                if k in ev_dict and ev_dict[k] >= 0.99:
                    ax.axvline(k, color="gray", linewidth=1.0, linestyle=":",
                               alpha=0.7)
                    ax.text(k, ax.get_ylim()[1] * 0.95, "99% EV",
                            ha="center", va="top", fontsize=7, color="gray")
                    break

            # Secondary x-axis showing explained variance
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ev_ticks = [k for k in k_sorted if k in ev_dict]
            ax2.set_xticks(ev_ticks)
            ax2.set_xticklabels(
                [f"{ev_dict[k]*100:.0f}%" for k in ev_ticks],
                fontsize=7,
            )
            ax2.set_xlabel("Explained variance (%)", fontsize=8)

    # Hide unused axes
    total_axes = n_rows * n_cols
    for idx in range(len(fixed_layers), total_axes):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        _save(fig, save_path)

    return fig


# ============================================================================
# FIGURE 3: TOP-1 TOKEN PRESERVATION RATE VS LAYER
# One panel per k value. Simple communicable result for the paper.
# ============================================================================

def plot_top1_preservation(
    records:    list,
    model_name: str,
    k_values:   list[int],
    ev_dict:    dict | None = None,
    save_path:  str | None  = None,
) -> plt.Figure:
    """
    Top-1 token preservation rate vs layer after posthoc ablation.

    Simple communicable result: if base prompts show lower preservation
    (top token changes more often when r⊥ is removed), that is direct
    evidence that r⊥ is load-bearing for base prompt predictions.

    Args:
        records:    list of AblationRecord
        model_name: used in title
        k_values:   one panel per k value
        ev_dict:    optional {k: EV fraction} for panel titles
        save_path:  output file path, or None

    Returns:
        matplotlib Figure object
    """
    agg = _aggregate_ablation_records(records, ablation_type="posthoc")
    n_rows, n_cols = _panel_layout(len(k_values))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, 5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Posthoc ablation: top-1 token preservation  [{model_name}]",
        fontsize=12,
    )

    for idx, k in enumerate(k_values):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row][col]

        n_layers = None
        for cat in ["base", "contrast"]:
            if cat in agg and k in agg[cat]:
                n_layers = len(agg[cat][k]["top1_mean"])
                break
        if n_layers is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        layers = np.arange(n_layers)

        for cat, style, label in [
            ("base",     STYLE_BASE,     "base"),
            ("contrast", STYLE_CONTRAST, "contrast"),
        ]:
            if cat not in agg or k not in agg[cat]:
                continue
            d = agg[cat][k]
            ax.plot(layers, d["top1_mean"], label=label, **style)
            ax.fill_between(layers,
                            d["top1_mean"] - d["top1_sem"],
                            d["top1_mean"] + d["top1_sem"],
                            color=style["color"], alpha=BAND_ALPHA)

        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_title(_panel_title(k, ev_dict), fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Fraction of prompts preserving top-1 token")
        ax.set_xticks(layers)
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=GRID_ALPHA)
        if idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    # Hide unused axes
    total_axes = n_rows * n_cols
    for idx in range(len(k_values), total_axes):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        _save(fig, save_path)

    return fig


# ============================================================================
# FIGURE 4: ENTROPY CHANGE VS LAYER
# One panel per k value. Connects ablation to the existing entropy framework.
# ============================================================================

def plot_entropy_change_vs_layer(
    records:    list,
    model_name: str,
    k_values:   list[int],
    ev_dict:    dict | None = None,
    save_path:  str | None  = None,
) -> plt.Figure:
    """
    Entropy change (H_ablated - H_full) vs layer after posthoc ablation.

    Connects ablation results to the existing entropy framework. Prediction:
    base prompts show larger positive entropy change (removing r⊥ makes
    their token distributions more diffuse), consistent with base prompts
    having invested more structured content in r⊥.

    Args:
        records:    list of AblationRecord
        model_name: used in title
        k_values:   one panel per k value
        ev_dict:    optional {k: EV fraction} for panel titles
        save_path:  output file path, or None

    Returns:
        matplotlib Figure object
    """
    agg = _aggregate_ablation_records(records, ablation_type="posthoc")
    n_rows, n_cols = _panel_layout(len(k_values))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(14, 5 * n_rows),
                             squeeze=False)
    fig.suptitle(
        f"Posthoc ablation: entropy change vs layer  [{model_name}]",
        fontsize=12,
    )

    for idx, k in enumerate(k_values):
        row = idx // n_cols
        col = idx % n_cols
        ax  = axes[row][col]

        n_layers = None
        for cat in ["base", "contrast"]:
            if cat in agg and k in agg[cat]:
                n_layers = len(agg[cat][k]["ent_mean"])
                break
        if n_layers is None:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        layers = np.arange(n_layers)

        for cat, style, label in [
            ("base",     STYLE_BASE,     "base"),
            ("contrast", STYLE_CONTRAST, "contrast"),
        ]:
            if cat not in agg or k not in agg[cat]:
                continue
            d = agg[cat][k]
            ax.plot(layers, d["ent_mean"], label=label, **style)
            ax.fill_between(layers,
                            d["ent_mean"] - d["ent_sem"],
                            d["ent_mean"] + d["ent_sem"],
                            color=style["color"], alpha=BAND_ALPHA)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title(_panel_title(k, ev_dict), fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("ΔEntropy: H(ablated) − H(full)  (bits)")
        ax.set_xticks(layers)
        ax.grid(alpha=GRID_ALPHA)

        # Sign annotation
        ax.text(0.97, 0.95,
                "↑ more diffuse / ↓ more concentrated",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="gray")

        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Hide unused axes
    total_axes = n_rows * n_cols
    for idx in range(len(k_values), total_axes):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        _save(fig, save_path)

    return fig


# ============================================================================
# FIGURE 5: INTERVENTION HEATMAP (Stage 2 only)
# 2D heatmap of (intervention_layer, k) effects with base/contrast/difference.
# ============================================================================

def _aggregate_intervention_records(records: list) -> dict:
    """
    Aggregate intervention AblationRecords into a structure suitable for
    heatmap plotting.

    Returns:
        result[role][(L, k)] = {
            "kl_mean":   float,
            "ent_mean":  float,
            "top1_mean": float,
            "n_prompts": int,
        }

    where L is the intervention_layer and k is the subspace rank.
    """
    filtered = [r for r in records if r.ablation_type == "intervention"]

    grouped = defaultdict(list)
    for r in filtered:
        grouped[(r.role, r.intervention_layer, r.k)].append(r)

    result = defaultdict(dict)
    for (role, L, k), recs in grouped.items():
        n = len(recs)
        result[role][(L, k)] = {
            "kl_mean":   np.mean([r.kl_divergence[0]  for r in recs]),
            "ent_mean":  np.mean([r.entropy_change[0]  for r in recs]),
            "top1_mean": np.mean([r.top1_preserved[0].astype(float) for r in recs]),
            "n_prompts": n,
        }

    return dict(result)


def plot_intervention_heatmap(
    records:    list,
    model_name: str,
    metric:     str = "kl_divergence",
    save_path:  str | None = None,
) -> plt.Figure:
    """
    Intervention heatmap: 2D view of (intervention_layer, k) effects.

    Three panels: base prompts, contrast prompts, and base − contrast
    difference. Provides a complete view of Stage 2 results across all
    (layer, k) combinations simultaneously.

    Args:
        records:    list of AblationRecord (filters to intervention type)
        model_name: used in title
        metric:     "kl_divergence", "entropy_change", or "top1_preserved"
        save_path:  output file path, or None

    Returns:
        matplotlib Figure object
    """
    agg = _aggregate_intervention_records(records)

    # Determine metric key and labels
    metric_key_map = {
        "kl_divergence":  "kl_mean",
        "entropy_change": "ent_mean",
        "top1_preserved": "top1_mean",
    }
    metric_label_map = {
        "kl_divergence":  "KL divergence (nats)",
        "entropy_change": "ΔEntropy (bits)",
        "top1_preserved": "Top-1 preservation rate",
    }
    mkey   = metric_key_map[metric]
    mlabel = metric_label_map[metric]

    # Collect all unique layers and k values
    all_keys = set()
    for cat_data in agg.values():
        all_keys.update(cat_data.keys())

    if not all_keys:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No intervention records found",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    layers_sorted = sorted(set(L for L, k in all_keys))
    k_sorted      = sorted(set(k for L, k in all_keys))

    n_layers_int = len(layers_sorted)
    n_k          = len(k_sorted)

    # Build 2D grids for each category
    def build_grid(cat_data):
        grid = np.full((n_k, n_layers_int), np.nan)
        for i, k in enumerate(k_sorted):
            for j, L in enumerate(layers_sorted):
                if (L, k) in cat_data:
                    grid[i, j] = cat_data[(L, k)][mkey]
        return grid

    grid_base     = build_grid(agg.get("base", {}))
    grid_contrast = build_grid(agg.get("contrast", {}))
    grid_diff     = grid_base - grid_contrast

    # Choose colormap
    if metric == "top1_preserved":
        cmap_main = "Blues"
        cmap_diff = "RdBu_r"
    else:
        cmap_main = "RdBu_r"
        cmap_diff = "RdBu_r"

    # Symmetric vmin/vmax for main panels
    vmax_main = max(np.nanmax(np.abs(grid_base)),
                    np.nanmax(np.abs(grid_contrast)))
    if metric == "top1_preserved":
        vmin_main, vmax_main_plot = 0.0, 1.0
    else:
        vmin_main, vmax_main_plot = -vmax_main, vmax_main

    vmax_diff = np.nanmax(np.abs(grid_diff))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Intervention ablation: {metric} heatmap  [{model_name}]",
        fontsize=12,
    )

    # Base panel
    im1 = ax1.imshow(grid_base, aspect="auto", origin="lower",
                     cmap=cmap_main, vmin=vmin_main, vmax=vmax_main_plot)
    ax1.set_title("Base prompts", fontsize=10)

    # Contrast panel
    im2 = ax2.imshow(grid_contrast, aspect="auto", origin="lower",
                     cmap=cmap_main, vmin=vmin_main, vmax=vmax_main_plot)
    ax2.set_title("Contrast prompts", fontsize=10)

    # Shared colorbar for base/contrast
    fig.colorbar(im2, ax=[ax1, ax2], label=mlabel, shrink=0.8)

    # Difference panel
    im3 = ax3.imshow(grid_diff, aspect="auto", origin="lower",
                     cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff)
    ax3.set_title("Base − Contrast", fontsize=10)
    fig.colorbar(im3, ax=ax3, label=f"Δ {mlabel}", shrink=0.8)

    # Axis labels and ticks
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Intervention layer")
        ax.set_ylabel("Subspace rank k")
        ax.set_xticks(range(n_layers_int))
        ax.set_xticklabels([str(L) for L in layers_sorted], fontsize=8)
        ax.set_yticks(range(n_k))
        ax.set_yticklabels([str(k) for k in k_sorted], fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        _save(fig, save_path)

    return fig
