"""ck_spectrum_plots.py — Visualization for c_k spectrum results.

Consumes CkRecords from ck_spectrum_compute.py.
No model, no torch, no forward passes — pure visualization.

Pipeline position:
    extraction.py → ck_spectrum_compute.py → VISUALIZATION (this file)

Plot functions:
    Single-prompt diagnostic:
        plot_single_prompt_diagnostic()     — 2×2 figure: projections, c_k spectrum

    Corpus / statistical:
        plot_heatmap_lasttoken()            — mean |c_k| at last token, log scale
        plot_heatmap_alltokens()            — mean |c_k| all tokens, log scale
        plot_com_vs_layer()                 — spectral CoM vs layer, mean ± std
        plot_cumpower_vs_k()                — cumulative power fraction F(K) vs k
        plot_delta_ck_heatmap()             — layer-to-layer |Δc_k| heatmap
        plot_heatmap_variance_lasttoken()   — prompt-variance heatmap + log ratio
        plot_variance_ratio_vs_k()          — log10 variance ratio vs k, by layer
        plot_com_variance_vs_layer()        — CoM of prompt-variance vs layer

All functions accept CkRecord objects or lists thereof.
All return the matplotlib Figure object.

Visual conventions match ablation_plots.py:
    - Base prompts: solid blue (#1f77b4)
    - Contrast prompts: dashed red (#d62728)
    - Shaded bands: mean ± 1 std, alpha=0.2
    - Grid: True, alpha=0.3
    - Titles: "Figure title  [model_name]"
    - Save: PNG at 150 DPI via _save()
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import warnings
import logging

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

from ck_spectrum_compute import CkRecord


# ── Style constants ────────────────────────────────────────────────────────────
COLOR_BASE     = "#1f77b4"
COLOR_CONTRAST = "#d62728"
GRID_ALPHA     = 0.3
DPI            = 150


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# PLOT 1: SINGLE-PROMPT 2×2 DIAGNOSTIC
# ============================================================================

def plot_single_prompt_diagnostic(
    ck_rec:    CkRecord,
    layer:     int,
    token:     int,
    save_path: str | None = None,
) -> plt.Figure:
    """
    2×2 diagnostic figure for one (prompt, layer, token) triple.

    Panel 1: Projections v_k · r in the singular-vector basis, k-ordered
    Panel 2: Sorted |v_k · r| descending vs rank
    Panel 3: c_k spectrum — c_k vs singular direction index k
    Panel 4: Sorted |c_k| descending vs rank

    Args:
        ck_rec:    CkRecord for a single prompt
        layer:     layer index to inspect
        token:     token position index (negative indexing supported)
        save_path: output path, or None to skip saving
    """
    ck_vec = ck_rec.ck_spectrum[layer, token, :]       # [d_model]
    S      = ck_rec.singular_values                     # [d_model]

    # Recover raw projections (v_k · r) = c_k / σ_k where σ_k > 0
    raw_proj = np.where(S > 1e-10, ck_vec / S, 0.0)   # [d_model]

    tok_label = f"token {token}" if token >= 0 else f"token {len(ck_rec.str_tokens) + token}"
    title_base = (f"c_k diagnostic  [{ck_rec.model_name}  "
                  f"layer {layer}  {tok_label}]")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title_base, fontsize=11)

    ax = axes[0, 0]
    ax.plot(raw_proj, color=COLOR_BASE, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title("Projections $v_k \\cdot r$ (singular basis, k-ordered)", fontsize=9)
    ax.set_xlabel("Singular direction index k")
    ax.set_ylabel("$v_k \\cdot r$")
    ax.grid(alpha=GRID_ALPHA)

    ax = axes[0, 1]
    sorted_proj = np.sort(np.abs(raw_proj))[::-1]
    ax.plot(sorted_proj, color=COLOR_BASE, linewidth=0.8)
    ax.set_title("Sorted $|v_k \\cdot r|$ descending", fontsize=9)
    ax.set_xlabel("Rank")
    ax.set_ylabel("$|v_k \\cdot r|$")
    ax.grid(alpha=GRID_ALPHA)

    ax = axes[1, 0]
    ax.plot(ck_vec, color=COLOR_CONTRAST, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title("$c_k = \\sigma_k (v_k \\cdot r)$ spectrum", fontsize=9)
    ax.set_xlabel("Singular direction index k  (descending $\\sigma_k$)")
    ax.set_ylabel("$c_k$")
    ax.grid(alpha=GRID_ALPHA)

    ax = axes[1, 1]
    sorted_ck = np.sort(np.abs(ck_vec))[::-1]
    ax.plot(sorted_ck, color=COLOR_CONTRAST, linewidth=0.8)
    ax.set_title("Sorted $|c_k|$ descending", fontsize=9)
    ax.set_xlabel("Rank")
    ax.set_ylabel("$|c_k|$")
    ax.grid(alpha=GRID_ALPHA)

    plt.tight_layout()

    if save_path is not None:
        _save(fig, save_path)

    return fig


# ============================================================================
# SHARED GRID HELPERS
# ============================================================================

def _build_heatmap_grids(
    ck_records:      list,
    last_token_only: bool,
    skip_layer0:     bool,
):
    """
    Compute mean |c_k| grids for base and contrast roles.

    Returns:
        (grid_base, grid_contrast) each [n_layers_plotted, d_model] or None.
    """
    def _mean_grid(records):
        if not records:
            return None
        per_prompt = []
        for r in records:
            if last_token_only:
                arr = np.abs(r.ck_spectrum[:, -1, :])    # [n_layers, d_model]
            else:
                arr = np.abs(r.ck_spectrum).mean(axis=1)  # [n_layers, d_model]
            if skip_layer0:
                arr = arr[1:, :]
            per_prompt.append(arr)
        return np.mean(per_prompt, axis=0)                # [n_layers_plotted, d_model]

    base_recs     = [r for r in ck_records if r.role == "base"]
    contrast_recs = [r for r in ck_records if r.role == "contrast"]
    return _mean_grid(base_recs), _mean_grid(contrast_recs)


def _render_heatmap_figure(
    grid_base:     np.ndarray | None,
    grid_contrast: np.ndarray | None,
    title:         str,
    save_path:     str | None = None,
) -> plt.Figure:
    """
    Render a 1–3 panel heatmap figure from pre-computed grids.

    Base and Contrast panels use log-scale (LogNorm). Difference panel uses
    symmetric linear scale since differences can be negative.
    """
    if grid_base is None and grid_contrast is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No records found", ha="center", va="center",
                transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    has_diff = grid_base is not None and grid_contrast is not None
    n_panels = 3 if has_diff else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=11)

    all_vals = np.concatenate([
        g.ravel() for g in [grid_base, grid_contrast] if g is not None
    ])
    vmax = float(all_vals.max())
    vmin_log = max(1e-6, vmax * 1e-4)
    log_norm = LogNorm(vmin=vmin_log, vmax=vmax)

    y_label = "Layer (layer 0 excluded)"
    x_label = "Singular direction index k  (descending σ_k)"

    panel_idx = 0
    for grid, label in [(grid_base, "Base"), (grid_contrast, "Contrast")]:
        if grid is None:
            continue
        ax = axes[panel_idx]
        im = ax.imshow(grid, aspect="auto", origin="upper",
                       cmap="viridis", norm=log_norm)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, label="mean |c_k|", fraction=0.046, pad=0.04)
        panel_idx += 1

    if has_diff:
        diff = grid_base - grid_contrast
        vd   = float(np.abs(diff).max()) or 1e-10
        ax   = axes[panel_idx]
        im   = ax.imshow(diff, aspect="auto", origin="upper",
                         cmap="RdBu_r", vmin=-vd, vmax=vd)
        ax.set_title("Base − Contrast", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, label="Δ mean |c_k|", fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


# ============================================================================
# PLOTS 2 & 3: LAYER-EVOLUTION HEATMAPS
# ============================================================================

def plot_heatmap_lasttoken(
    ck_records:  list,
    model_name:  str,
    skip_layer0: bool = True,
    save_path:   str | None = None,
) -> plt.Figure:
    """Heatmap of mean |c_k| at the last token only, log scale, layers 1–end."""
    grid_b, grid_c = _build_heatmap_grids(ck_records, last_token_only=True,
                                           skip_layer0=skip_layer0)
    title = f"c_k spectrum: mean |c_k| at last token  [{model_name}]"
    return _render_heatmap_figure(grid_b, grid_c, title, save_path)


def plot_heatmap_alltokens(
    ck_records:  list,
    model_name:  str,
    skip_layer0: bool = True,
    save_path:   str | None = None,
) -> plt.Figure:
    """Heatmap of mean |c_k| averaged over all tokens, log scale, layers 1–end."""
    grid_b, grid_c = _build_heatmap_grids(ck_records, last_token_only=False,
                                           skip_layer0=skip_layer0)
    title = f"c_k spectrum: mean |c_k| all tokens  [{model_name}]"
    return _render_heatmap_figure(grid_b, grid_c, title, save_path)


# ============================================================================
# PLOT 4: SPECTRAL CENTER OF MASS vs. LAYER
# ============================================================================

def _compute_spectral_com(
    ck_records:  list,
    skip_layer0: bool = True,
):
    """
    Spectral CoM at the last token: CoM = Σ_k k·c_k² / Σ_k c_k².

    Returns:
        (com_base, com_contrast) each np.ndarray [n_prompts, n_layers_plotted],
        or None if no records for that role.
    """
    def _com_for_role(records):
        if not records:
            return None
        d_model   = records[0].ck_spectrum.shape[2]
        k_indices = np.arange(d_model, dtype=np.float64)
        coms = []
        for r in records:
            arr   = r.ck_spectrum[:, -1, :]           # [n_layers, d_model]
            if skip_layer0:
                arr = arr[1:, :]
            power = arr ** 2                           # [n_layers_plotted, d_model]
            total = power.sum(axis=1)                  # [n_layers_plotted]
            com   = (power @ k_indices) / np.where(total > 0, total, 1.0)
            coms.append(com)
        return np.array(coms)                          # [n_prompts, n_layers_plotted]

    base_recs     = [r for r in ck_records if r.role == "base"]
    contrast_recs = [r for r in ck_records if r.role == "contrast"]
    return _com_for_role(base_recs), _com_for_role(contrast_recs)


def plot_com_vs_layer(
    ck_records:  list,
    model_name:  str,
    skip_layer0: bool = True,
    save_path:   str | None = None,
) -> plt.Figure:
    """Spectral CoM vs. layer, mean ± 1 std across prompts, base vs. contrast."""
    com_base, com_contrast = _compute_spectral_com(ck_records, skip_layer0=skip_layer0)

    if com_base is None and com_contrast is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No records found", ha="center", va="center",
                transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    n_layers_plotted = (com_base if com_base is not None else com_contrast).shape[1]
    layer_start = 1 if skip_layer0 else 0
    x = np.arange(layer_start, layer_start + n_layers_plotted)

    fig, ax = plt.subplots(figsize=(9, 5))
    for com, label, color in [
        (com_base,     "Base",     COLOR_BASE),
        (com_contrast, "Contrast", COLOR_CONTRAST),
    ]:
        if com is None:
            continue
        mean = com.mean(axis=0)
        std  = com.std(axis=0)
        ax.plot(x, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title(f"Spectral CoM vs. layer (last token)  [{model_name}]", fontsize=11)
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"CoM  ($\Sigma_k\, k \cdot c_k^2\, /\, \Sigma_k\, c_k^2$)")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


# ============================================================================
# PLOT 5: CUMULATIVE POWER FRACTION vs. k
# ============================================================================

def _compute_cumulative_power(
    ck_records:  list,
    skip_layer0: bool = True,
):
    """
    Cumulative power fraction F(K) = Σ_{k<K} c_k² / Σ_k c_k² at the last token.

    Uses c_k² (squared) — exact power decomposition; logit variance = Σ_k c_k².

    Returns:
        (base_cum, contrast_cum) each dict mapping layer_index -> mean F array [d_model].
        Layer indices are 1-based when skip_layer0=True.
    """
    def _cum_for_role(records):
        if not records:
            return None
        layer_sums  = {}
        layer_count = {}
        for r in records:
            arr = r.ck_spectrum[:, -1, :]          # [n_layers, d_model]
            if skip_layer0:
                arr = arr[1:, :]
            layer_start = 1 if skip_layer0 else 0
            power = arr ** 2                        # [n_layers_plotted, d_model]
            total = power.sum(axis=1, keepdims=True)
            frac  = power / np.where(total > 0, total, 1.0)
            cum   = np.cumsum(frac, axis=1)        # [n_layers_plotted, d_model]
            for i, l in enumerate(range(layer_start, layer_start + cum.shape[0])):
                layer_sums[l]  = layer_sums.get(l, 0.0) + cum[i]
                layer_count[l] = layer_count.get(l, 0) + 1
        return {l: layer_sums[l] / layer_count[l] for l in layer_sums}

    base_recs     = [r for r in ck_records if r.role == "base"]
    contrast_recs = [r for r in ck_records if r.role == "contrast"]
    return _cum_for_role(base_recs), _cum_for_role(contrast_recs)


def plot_cumpower_vs_k(
    ck_records:     list,
    model_name:     str,
    summary_layers: list,
    skip_layer0:    bool = True,
    save_path:      str | None = None,
) -> plt.Figure:
    """
    Cumulative power fraction F(K) vs K for selected layers.

    One subplot per layer in summary_layers; base (solid) and contrast (dashed).
    """
    base_cum, contrast_cum = _compute_cumulative_power(ck_records, skip_layer0=skip_layer0)

    available = set()
    if base_cum:
        available |= set(base_cum.keys())
    if contrast_cum:
        available |= set(contrast_cum.keys())

    layers_to_plot = [l for l in summary_layers if l in available]
    skipped = [l for l in summary_layers if l not in available]
    if skipped:
        print(f"  [cumpower] Skipping layers not in data: {skipped}")

    if not layers_to_plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid summary layers", ha="center", va="center",
                transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    n_sub = len(layers_to_plot)
    fig, axes = plt.subplots(1, n_sub, figsize=(5 * n_sub, 4), sharey=True)
    if n_sub == 1:
        axes = [axes]
    fig.suptitle(
        r"Cumulative power fraction $F(K) = \Sigma_{k<K}\, c_k^2\, /\, \Sigma_k\, c_k^2$"
        f"  [{model_name}]",
        fontsize=11,
    )

    for i, (ax, l) in enumerate(zip(axes, layers_to_plot)):
        for src, label, color, ls in [
            (base_cum,     "Base",     COLOR_BASE,     "solid"),
            (contrast_cum, "Contrast", COLOR_CONTRAST, "dashed"),
        ]:
            if src is None or l not in src:
                continue
            cum = src[l]
            k = np.arange(len(cum))
            ax.plot(k, cum, color=color, linestyle=ls,
                    label=label if i == 0 else None, linewidth=1.2)
        ax.set_title(f"Layer {l}", fontsize=10)
        ax.set_xlabel("k")
        if i == 0:
            ax.set_ylabel("F(K)")
            ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


# ============================================================================
# PLOT 6: LAYER-TO-LAYER |Δc_k| HEATMAP
# ============================================================================

def _compute_delta_ck(
    ck_records:  list,
    skip_layer0: bool = True,
):
    """
    Layer-to-layer absolute spectral change at the last token.

    With skip_layer0=True, arr starts at layer 1; diff gives transitions 1→2, 2→3, …

    Returns:
        (delta_base, delta_contrast) each np.ndarray [n_prompts, n_transitions, d_model]
        or None if no records for that role.
    """
    def _delta_for_role(records):
        if not records:
            return None
        deltas = []
        for r in records:
            arr = r.ck_spectrum[:, -1, :]          # [n_layers, d_model]
            if skip_layer0:
                arr = arr[1:, :]
            delta = np.abs(np.diff(arr, axis=0))   # [n_layers_plotted-1, d_model]
            deltas.append(delta)
        return np.array(deltas)                     # [n_prompts, n_transitions, d_model]

    base_recs     = [r for r in ck_records if r.role == "base"]
    contrast_recs = [r for r in ck_records if r.role == "contrast"]
    return _delta_for_role(base_recs), _delta_for_role(contrast_recs)


def plot_delta_ck_heatmap(
    ck_records:  list,
    model_name:  str,
    skip_layer0: bool = True,
    save_path:   str | None = None,
) -> plt.Figure:
    """Layer-to-layer |Δc_k| heatmap at the last token, stratified by role."""
    delta_base, delta_contrast = _compute_delta_ck(ck_records, skip_layer0=skip_layer0)

    if delta_base is None and delta_contrast is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No records found", ha="center", va="center",
                transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    mean_base     = delta_base.mean(axis=0)     if delta_base     is not None else None
    mean_contrast = delta_contrast.mean(axis=0) if delta_contrast is not None else None

    has_diff = mean_base is not None and mean_contrast is not None
    n_panels = 3 if has_diff else 1

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"Layer-to-layer |Δc_k| at last token  [{model_name}]", fontsize=11)

    all_vals = np.concatenate([
        g.ravel() for g in [mean_base, mean_contrast] if g is not None
    ])
    vmax = float(all_vals.max()) or 1e-10

    ref = mean_base if mean_base is not None else mean_contrast
    n_trans = ref.shape[0]
    layer_start = 1 if skip_layer0 else 0
    ytick_labels = [f"{layer_start + i}→{layer_start + i + 1}" for i in range(n_trans)]

    x_label = "Singular direction index k  (descending σ_k)"
    y_label = "Layer transition (l → l+1)"

    panel_idx = 0
    for grid, label in [(mean_base, "Base"), (mean_contrast, "Contrast")]:
        if grid is None:
            continue
        ax = axes[panel_idx]
        im = ax.imshow(grid, aspect="auto", origin="upper",
                       cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yticks(np.arange(n_trans))
        ax.set_yticklabels(ytick_labels, fontsize=7)
        fig.colorbar(im, ax=ax, label="mean |Δc_k|", fraction=0.046, pad=0.04)
        panel_idx += 1

    if has_diff:
        diff = mean_base - mean_contrast
        vd   = float(np.abs(diff).max()) or 1e-10
        ax   = axes[panel_idx]
        im   = ax.imshow(diff, aspect="auto", origin="upper",
                         cmap="RdBu_r", vmin=-vd, vmax=vd)
        ax.set_title("Base − Contrast", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_yticks(np.arange(n_trans))
        ax.set_yticklabels(ytick_labels, fontsize=7)
        fig.colorbar(im, ax=ax, label="Δ mean |Δc_k|", fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


# ============================================================================
# PLOTS 7–9: PROMPT-VARIANCE DIAGNOSTICS
# ============================================================================

def _compute_variance_grids(
    ck_records:  list,
    skip_layer0: bool = True,
):
    """
    Compute per-(layer, k) variance of |c_k| across prompts at the last token.

    Returns:
        (var_base, var_contrast) each np.ndarray [n_layers_plotted, d_model] or None.
    """
    def _var_grid(records):
        if not records:
            return None
        per_prompt = []
        for r in records:
            arr = np.abs(r.ck_spectrum[:, -1, :])   # [n_layers, d_model]
            if skip_layer0:
                arr = arr[1:, :]
            per_prompt.append(arr)
        stack = np.stack(per_prompt, axis=0)         # [n_prompts, n_layers_plotted, d_model]
        return stack.var(axis=0)                     # [n_layers_plotted, d_model]

    base_recs     = [r for r in ck_records if r.role == "base"]
    contrast_recs = [r for r in ck_records if r.role == "contrast"]
    return _var_grid(base_recs), _var_grid(contrast_recs)


def plot_heatmap_variance_lasttoken(
    ck_records:  list,
    model_name:  str,
    skip_layer0: bool = True,
    save_path:   str | None = None,
) -> plt.Figure:
    """
    Three-panel variance heatmap at the last token.

    Panel 1: Var_base(layer, k)     — log scale
    Panel 2: Var_contrast(layer, k) — same log-norm as Panel 1
    Panel 3: log10(Var_base / Var_contrast) — diverging linear scale, clipped to [-2, 2]
    """
    var_base, var_contrast = _compute_variance_grids(ck_records, skip_layer0=skip_layer0)

    if var_base is None and var_contrast is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No records found", ha="center", va="center",
                transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    has_ratio = var_base is not None and var_contrast is not None
    n_panels  = 3 if has_ratio else (1 if var_base is None or var_contrast is None else 2)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"c_k prompt variance at last token  [{model_name}]",
        fontsize=11,
    )

    all_var_vals = np.concatenate([
        g.ravel() for g in [var_base, var_contrast] if g is not None
    ])
    vmax_var = float(all_var_vals.max())
    vmin_log = max(1e-12, vmax_var * 1e-4)
    log_norm = LogNorm(vmin=vmin_log, vmax=vmax_var)

    x_label = "Singular direction index k  (descending σ_k)"
    y_label = "Layer (layer 0 excluded)"

    panel_idx = 0
    for grid, label in [(var_base, "Base"), (var_contrast, "Contrast")]:
        if grid is None:
            continue
        ax = axes[panel_idx]
        im = ax.imshow(grid, aspect="auto", origin="upper",
                       cmap="viridis", norm=log_norm)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, label="Var(|c_k|) across prompts",
                     fraction=0.046, pad=0.04)
        panel_idx += 1

    if has_ratio:
        ratio = np.log10(var_base / (var_contrast + 1e-8))
        ratio_clipped = np.clip(ratio, -2.0, 2.0)
        ax = axes[panel_idx]
        im = ax.imshow(ratio_clipped, aspect="auto", origin="upper",
                       cmap="RdBu_r", vmin=-2.0, vmax=2.0)
        ax.set_title("log₁₀(Var_base / Var_contrast)", fontsize=10)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(im, ax=ax, label="log₁₀ variance ratio  (clipped ±2)",
                     fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def plot_variance_ratio_vs_k(
    ck_records:     list,
    model_name:     str,
    summary_layers: list,
    skip_layer0:    bool = True,
    save_path:      str | None = None,
) -> plt.Figure:
    """
    log10(Var_base / Var_contrast) vs k for representative layers.

    One line per layer in summary_layers, all on the same axes.
    Horizontal dashed line at zero marks equal variance.
    """
    var_base, var_contrast = _compute_variance_grids(ck_records, skip_layer0=skip_layer0)

    if var_base is None or var_contrast is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "Need both base and contrast records", ha="center",
                va="center", transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    n_layers_plotted = var_base.shape[0]
    layer_start      = 1 if skip_layer0 else 0
    available        = set(range(layer_start, layer_start + n_layers_plotted))
    layers_to_plot   = [l for l in summary_layers if l in available]
    skipped          = [l for l in summary_layers if l not in available]
    if skipped:
        print(f"  [variance_ratio] Skipping layers not in data: {skipped}")

    if not layers_to_plot:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.text(0.5, 0.5, "No valid summary layers", ha="center", va="center",
                transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    log_ratio = np.log10(var_base / (var_contrast + 1e-8))   # [n_layers_plotted, d_model]

    d_model = var_base.shape[1]
    k       = np.arange(d_model)
    cmap    = plt.cm.viridis
    colors  = [cmap(i / max(len(layers_to_plot) - 1, 1)) for i in range(len(layers_to_plot))]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=0)
    for color, l in zip(colors, layers_to_plot):
        row = l - layer_start
        ax.plot(k, log_ratio[row], color=color, linewidth=1.0, label=f"Layer {l}")

    ax.set_title(
        f"log₁₀(Var_base / Var_contrast) vs k  [{model_name}]", fontsize=11
    )
    ax.set_xlabel("Singular direction index k  (descending σ_k)")
    ax.set_ylabel("log₁₀ variance ratio  (> 0 → base more variable)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=GRID_ALPHA)
    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig


def _compute_variance_com(
    ck_records:  list,
    skip_layer0: bool = True,
):
    """
    Spectral CoM of the prompt-variance: CoM_var = Σ_k k·Var(|c_k|) / Σ_k Var(|c_k|).

    Computed at the last token, separately for base and contrast.

    Returns:
        (com_base, com_contrast) each np.ndarray [n_layers_plotted] or None.
    """
    var_base, var_contrast = _compute_variance_grids(ck_records, skip_layer0=skip_layer0)

    def _com_of_var(var_grid):
        if var_grid is None:
            return None
        d_model   = var_grid.shape[1]
        k_indices = np.arange(d_model, dtype=np.float64)
        total     = var_grid.sum(axis=1)
        return (var_grid @ k_indices) / np.where(total > 0, total, 1.0)

    return _com_of_var(var_base), _com_of_var(var_contrast)


def plot_com_variance_vs_layer(
    ck_records:  list,
    model_name:  str,
    skip_layer0: bool = True,
    save_path:   str | None = None,
) -> plt.Figure:
    """
    CoM of the prompt-variance vs. layer, base vs. contrast.

    Unlike plot_com_vs_layer (which shows mean ± std of per-prompt CoM),
    this collapses variance across prompts first, then computes CoM — so
    there is no per-prompt scatter to shade. Plotted as lines only.
    """
    com_base, com_contrast = _compute_variance_com(ck_records, skip_layer0=skip_layer0)

    if com_base is None and com_contrast is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No records found", ha="center", va="center",
                transform=ax.transAxes)
        if save_path is not None:
            _save(fig, save_path)
        return fig

    ref          = com_base if com_base is not None else com_contrast
    n_layers_plotted = len(ref)
    layer_start  = 1 if skip_layer0 else 0
    x            = np.arange(layer_start, layer_start + n_layers_plotted)

    fig, ax = plt.subplots(figsize=(9, 5))
    for com, label, color in [
        (com_base,     "Base",     COLOR_BASE),
        (com_contrast, "Contrast", COLOR_CONTRAST),
    ]:
        if com is None:
            continue
        ax.plot(x, com, color=color, label=label, linewidth=1.5)

    ax.set_title(
        f"CoM of prompt variance vs. layer  [{model_name}]", fontsize=11
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel(r"CoM_var  ($\Sigma_k\, k \cdot \mathrm{Var}(|c_k|)\, /\, \Sigma_k\, \mathrm{Var}(|c_k|)$)")
    ax.grid(alpha=GRID_ALPHA)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        _save(fig, save_path)
    return fig
