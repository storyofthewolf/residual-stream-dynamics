"""dashboard_viz.py — Figure-generating functions for the dashboard.

All functions take numpy arrays and return matplotlib.Figure.
No file I/O, no filtering, no model loading, no torch.

Each function handles the zero-records case by returning an empty figure
with an informative title rather than raising.

Visual conventions match post_process_plots.py:
    COLOR_BASE     = "#2166ac"   blue
    COLOR_CONTRAST = "#d6604d"   red
    BAND_ALPHA     = 0.2
    GRID_ALPHA     = 0.3
    DPI            = 130
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ── Style constants ────────────────────────────────────────────────────────────
COLOR_BASE     = "#2166ac"
COLOR_CONTRAST = "#d6604d"
BAND_ALPHA     = 0.2
GRID_ALPHA     = 0.3
DPI            = 130


# ── Internal helpers ───────────────────────────────────────────────────────────

def _mean_sem(profiles: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer-wise mean and SEM for a list of variable-length 1D arrays.

    Profiles with different lengths are aligned to the shortest length so that
    the x-axis is always 0-based integer layer indices (not fractional depth).
    This preserves the absolute layer numbering that is scientifically meaningful.

    Returns
    -------
    x    : 1D int array — layer indices 0 .. min_len-1
    mean : 1D float array
    sem  : 1D float array
    """
    if not profiles:
        return np.array([]), np.array([]), np.array([])
    min_len = min(len(p) for p in profiles)
    arr  = np.array([p[:min_len] for p in profiles], dtype=np.float64)
    n    = arr.shape[0]
    mean = arr.mean(axis=0)
    sem  = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(min_len)
    x    = np.arange(min_len)
    return x, mean, sem


def _empty_fig(msg: str) -> Figure:
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 4), dpi=DPI)
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=12, color="#666666")
    ax.set_axis_off()
    plt.tight_layout()
    return fig


def _style_ax(ax, xlabel: str = "Layer", ylabel: str = "") -> None:
    ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_entropy_curves(
    base_profiles: list,
    contrast_profiles: list,
    model_name: str,
    norm_key: str,
    alpha: float,
    position_mode: str,
) -> Figure:
    """
    Mean ± SEM entropy vs layer for base (blue) and contrast (red).

    Parameters
    ----------
    base_profiles     : list of 1D arrays, one per prompt
    contrast_profiles : list of 1D arrays, one per prompt
    model_name        : displayed in title
    norm_key          : displayed in title
    alpha             : Rényi alpha, displayed in title
    position_mode     : "final" or "mean", displayed in subtitle
    """
    if not base_profiles and not contrast_profiles:
        return _empty_fig(f"No entropy records found\n{model_name}  ·  {norm_key}  ·  α={alpha}")

    alpha_label = "Shannon (α→1)" if abs(alpha - 1.0) < 1e-4 else f"α={alpha}"
    pos_label   = "final token" if position_mode == "final" else "mean over tokens"
    title = f"Entropy — {model_name}  ·  {norm_key}  ·  {alpha_label}  ·  {pos_label}"

    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)

    for profiles, label, color in [
        (base_profiles,     f"Base (n={len(base_profiles)})",     COLOR_BASE),
        (contrast_profiles, f"Contrast (n={len(contrast_profiles)})", COLOR_CONTRAST),
    ]:
        if not profiles:
            continue
        x, mean, sem = _mean_sem(profiles)
        ax.plot(x, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=BAND_ALPHA)

    ax.set_title(title, fontsize=10, pad=8)
    _style_ax(ax, ylabel="Entropy (bits)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_wu_subspace_curves(
    base_profiles: list,
    contrast_profiles: list,
    model_name: str,
    direction: str,
    k: int,
    alpha: float,
) -> Figure:
    """
    Mean ± SEM WU subspace entropy vs layer, base vs contrast.

    Parameters
    ----------
    direction : "parallel" or "orthogonal"
    k         : subspace rank
    """
    if not base_profiles and not contrast_profiles:
        return _empty_fig(
            f"No WU subspace records found\n{model_name}  ·  {direction}  k={k}  ·  α={alpha}"
        )

    alpha_label = "Shannon (α→1)" if abs(alpha - 1.0) < 1e-4 else f"α={alpha}"
    dir_symbol  = "r‖" if direction == "parallel" else "r⊥"
    title = (f"WU Subspace ({dir_symbol})  —  {model_name}  ·  k={k}  ·  {alpha_label}")

    plt.close('all')
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)

    for profiles, label, color in [
        (base_profiles,     f"Base (n={len(base_profiles)})",     COLOR_BASE),
        (contrast_profiles, f"Contrast (n={len(contrast_profiles)})", COLOR_CONTRAST),
    ]:
        if not profiles:
            continue
        x, mean, sem = _mean_sem(profiles)
        ax.plot(x, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=BAND_ALPHA)

    ax.set_title(title, fontsize=10, pad=8)
    _style_ax(ax, ylabel="Entropy (bits)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_ablation_posthoc(
    base_top1: list,
    contrast_top1: list,
    base_kl: list,
    contrast_kl: list,
    model_name: str,
    k: int,
) -> Figure:
    """
    Two-panel figure: top-1 token preservation (left) and KL divergence (right).

    Parameters
    ----------
    base_top1     : list of 1D float arrays (0.0/1.0 per layer)
    contrast_top1 : list of 1D float arrays
    base_kl       : list of 1D float arrays
    contrast_kl   : list of 1D float arrays
    """
    has_data = base_top1 or contrast_top1 or base_kl or contrast_kl
    if not has_data:
        return _empty_fig(f"No posthoc ablation records\n{model_name}  ·  k={k}")

    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI)
    fig.suptitle(f"Posthoc Ablation — {model_name}  ·  k={k}", fontsize=11)

    # Left: top-1 preservation
    ax = axes[0]
    for profiles, label, color in [
        (base_top1,     f"Base (n={len(base_top1)})",     COLOR_BASE),
        (contrast_top1, f"Contrast (n={len(contrast_top1)})", COLOR_CONTRAST),
    ]:
        if not profiles:
            continue
        x, mean, sem = _mean_sem(profiles)
        ax.plot(x, mean * 100, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, (mean - sem) * 100, (mean + sem) * 100,
                         color=color, alpha=BAND_ALPHA)
    ax.set_title("Top-1 Token Preserved (%)", fontsize=10)
    ax.set_ylim(0, 105)
    _style_ax(ax, ylabel="%")
    ax.legend(fontsize=9)

    # Right: KL divergence
    ax = axes[1]
    for profiles, label, color in [
        (base_kl,     f"Base (n={len(base_kl)})",     COLOR_BASE),
        (contrast_kl, f"Contrast (n={len(contrast_kl)})", COLOR_CONTRAST),
    ]:
        if not profiles:
            continue
        x, mean, sem = _mean_sem(profiles)
        ax.plot(x, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=BAND_ALPHA)
    ax.set_title("KL Divergence from Full Model", fontsize=10)
    _style_ax(ax, ylabel="KL (nats)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_ablation_heatmap(
    top1_base: np.ndarray,
    top1_contrast: np.ndarray,
    k_values: list,
    layer_indices: list,
    model_name: str,
) -> Figure:
    """
    Three-panel heatmap: Base | Contrast | Base−Contrast difference.

    Parameters
    ----------
    top1_base     : ndarray [n_k, n_layers] — mean top-1 preservation rate
    top1_contrast : ndarray [n_k, n_layers]
    k_values      : list of int  (y-axis ticks)
    layer_indices : list of int  (x-axis ticks)
    """
    if top1_base.size == 0 or top1_contrast.size == 0:
        return _empty_fig(f"No intervention ablation records\n{model_name}")

    plt.close('all')
    diff = top1_base - top1_contrast
    vmin, vmax = 0.0, 1.0
    dv = max(float(np.nanmax(np.abs(diff))), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(k_values) * 0.4 + 2)), dpi=DPI)
    fig.suptitle(f"Intervention Ablation — Top-1 Preservation  [{model_name}]", fontsize=11)

    extent = [-0.5, len(layer_indices) - 0.5, len(k_values) - 0.5, -0.5]

    for ax, grid, title, cmap, vn, vx, clabel in [
        (axes[0], top1_base,     "Base",           "Blues",   vmin, vmax, "Top-1 preserved"),
        (axes[1], top1_contrast, "Contrast",       "Reds",    vmin, vmax, "Top-1 preserved"),
        (axes[2], diff,          "Base − Contrast","RdBu_r",  -dv,  dv,   "Δ Top-1"),
    ]:
        im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vn, vmax=vx,
                       interpolation="nearest", extent=extent, origin="upper")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Intervention Layer", fontsize=9)
        ax.set_ylabel("Subspace Rank k", fontsize=9)
        ax.set_xticks(range(len(layer_indices)))
        ax.set_xticklabels([str(l) for l in layer_indices], fontsize=7)
        ax.set_yticks(range(len(k_values)))
        ax.set_yticklabels([str(k) for k in k_values], fontsize=7)
        fig.colorbar(im, ax=ax, label=clabel, fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def plot_mechanics_curves(
    base: dict,
    contrast: dict,
    model_name: str,
) -> Figure:
    """
    Five-panel figure: one subplot per mechanical quantity, base vs contrast.

    Each panel shows mean ± SEM across prompts.

    Parameters
    ----------
    base     : dict returned by DashboardLoader.query_mechanics(..., role="base")
    contrast : dict returned by DashboardLoader.query_mechanics(..., role="contrast")
    model_name : displayed in title
    """
    has_data = any(base.get(k) for k in ("speed",)) or any(contrast.get(k) for k in ("speed",))
    if not has_data:
        return _empty_fig(
            f"No mechanics records found for {model_name}\n"
            "(run workflows/mechanics_analysis.py --save-data to generate mechanics/ files)"
        )

    plt.close('all')
    fig, axes = plt.subplots(1, 5, figsize=(25, 5), dpi=DPI)
    fig.suptitle(
        f"Residual Stream Mechanics — {model_name}  ·  mean ± SEM  ·  final token",
        fontsize=11,
    )

    panels = [
        ("speed",                    "Layer transition", "||ΔX_l||₂",         "Speed"),
        ("acceleration_magnitude",   "Layer transition", "||ΔV_l||₂",         "Acceleration magnitude"),
        ("cosine_sim_state",         "Layer transition", "cos(X_l, X_{l+1})", "State cosine similarity"),
        ("cosine_sim_update_state",  "Layer transition", "cos(ΔX_l, X_l)",    "Update–state alignment"),
        ("cosine_sim_update_update", "Layer transition", "cos(ΔX_l, ΔX_{l+1})", "Update coherence"),
    ]
    cosine_keys = {"cosine_sim_state", "cosine_sim_update_state", "cosine_sim_update_update"}

    for ax, (key, xlabel, ylabel, title) in zip(axes, panels):
        for role_data, label, color in [
            (base,     f"Base (n={len(base.get(key, []))})",     COLOR_BASE),
            (contrast, f"Contrast (n={len(contrast.get(key, []))})", COLOR_CONTRAST),
        ]:
            profiles = role_data.get(key, [])
            if not profiles:
                continue
            x, mean, sem = _mean_sem(profiles)
            ax.plot(x, mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=BAND_ALPHA)

        ax.set_title(title, fontsize=9)
        _style_ax(ax, xlabel=xlabel, ylabel=ylabel)
        if key in cosine_keys:
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(0.0, color="#888888", linewidth=0.7, linestyle="--")

    axes[0].legend(fontsize=8)
    plt.tight_layout()
    return fig


def plot_ck_spectrum(
    ck_mean: np.ndarray | None,
    ck_sem: np.ndarray | None,
    singular_values: np.ndarray | None,
    model_name: str,
    layer_idx: int,
    role: str,
) -> Figure:
    """
    Two-panel figure: c_k spectrum (top) and singular value spectrum (bottom).

    Top panel: mean |c_k| ± SEM vs singular direction index k.
    Bottom panel: σ_k vs k (same x-axis).

    Parameters
    ----------
    ck_mean         : 1D array [d_model] or None
    ck_sem          : 1D array [d_model] or None
    singular_values : 1D array [d_model] or None
    """
    if ck_mean is None:
        return _empty_fig("No c_k data loaded\n(run ck_analysis.py --save-data to generate ck/ files)")

    plt.close('all')
    color = COLOR_BASE if role == "base" else COLOR_CONTRAST
    d_model = len(ck_mean)
    k = np.arange(d_model)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=DPI, sharex=True)
    fig.suptitle(
        f"c_k Spectrum — {model_name}  ·  Layer {layer_idx}  ·  {role.capitalize()}",
        fontsize=11,
    )

    # Top: |c_k| mean ± SEM
    ax = axes[0]
    ax.plot(k, np.abs(ck_mean), color=color, linewidth=1.2, label=f"{role} mean |c_k|")
    if ck_sem is not None:
        ax.fill_between(k,
                         np.abs(ck_mean) - ck_sem,
                         np.abs(ck_mean) + ck_sem,
                         color=color, alpha=BAND_ALPHA)
    ax.set_ylabel("|c_k|", fontsize=10)
    ax.set_title("c_k = σ_k · (v_k · r)  spectrum", fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    # Bottom: σ_k
    ax = axes[1]
    if singular_values is not None:
        ax.plot(k, singular_values, color="#555555", linewidth=1.0, label="σ_k")
        ax.fill_between(k, 0, singular_values, color="#aaaaaa", alpha=0.3)
    ax.set_xlabel("Singular direction index k  (descending σ_k)", fontsize=10)
    ax.set_ylabel("σ_k", fontsize=10)
    ax.set_title("Singular value spectrum", fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)

    plt.tight_layout()
    return fig
