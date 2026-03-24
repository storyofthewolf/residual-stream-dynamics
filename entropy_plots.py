"""entropy_plots.py — Visualization for entropy surfaces.

Consumes EntropyRecords from computation.py.
No model, no torch, no forward passes — pure visualization.

Pipeline position:
    extraction.py → computation.py → VISUALIZATION (this file)

Plot functions:
    Single-prompt exploratory:
        plot_fixed_position()   — entropy vs layer, one curve per token position
        plot_fixed_layer()      — entropy vs token position, one curve per layer
        plot_2d_surface()       — full entropy surface as heatmap
        plot_hook_comparison()  — overlay multiple hook types on one axes

    Corpus / statistical:
        plot_overall_mean()     — mean ± 1σ across all corpus pairs
        plot_category()         — per-category base vs contrast curves

All functions accept EntropyRecord objects or lists thereof.
Hook type, normalization method, and alpha are read from the record —
no implicit assumptions about what was computed.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from computation import EntropyRecord, NORM_METHODS, filter_records
from extraction import HOOK_LABELS

# ============================================================================
# SHARED STYLE
# ============================================================================

COLORS      = plt.cm.tab10.colors
PAIR_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors[:5])

# Distinct colors for hook type overlays
HOOK_COLORS = {
    "resid_pre":  "#1565C0",   # dark blue
    "resid_mid":  "#6A1B9A",   # purple
    "resid_post": "#1B5E20",   # dark green
    "attn_out":   "#E65100",   # deep orange
    "mlp_out":    "#B71C1C",   # dark red
    "mlp_pre":    "#F57F17",   # amber
    "mlp_post":   "#4E342E",   # brown
}


def _alpha_label(alpha: float) -> str:
    return "Shannon" if abs(alpha - 1.0) < 1e-6 else f"Rényi α={alpha}"


def _hook_label(hook_type: str) -> str:
    return HOOK_LABELS.get(hook_type, hook_type)


def _dim_annotation(d_model: int) -> str:
    """Annotate when d_model is non-standard (MLP internal space)."""
    return f"  [d={d_model}]" if d_model not in (768, 1024, 2048, 4096) else ""


def _save(fig, path: Path, filename: str) -> None:
    out = path / filename
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def _bos_slice(surface: np.ndarray, str_tokens: list, skip_bos: bool):
    """Return (surface_slice, token_labels) with optional BOS removal."""
    if skip_bos and len(str_tokens) > 0 and str_tokens[0] == '<|endoftext|>':
        return surface[:, 1:], str_tokens[1:]
    return surface, str_tokens


# ============================================================================
# 1D: ENTROPY VS LAYER  (fixed token position, vary layer)
#
# Physical meaning:
#   Each curve traces how ONE token position's representation evolves
#   through network depth. Final-token curve → next-token prediction.
#   Earlier-token curves → contextual elaboration under causal masking.
# ============================================================================

def plot_fixed_position(
    record:     EntropyRecord,
    output_dir: Path,
    filename:   Optional[str] = None,
    skip_bos:   bool = True,
) -> None:
    """
    Entropy vs layer, one curve per token position.

    Args:
        record:     single EntropyRecord (one norm_key, one alpha)
        output_dir: directory for saved figure
        filename:   output filename (auto-generated if None)
        skip_bos:   skip the BOS token position (default True)
    """
    surface, str_tokens = _bos_slice(record.surface, record.str_tokens, skip_bos)
    seq_len  = surface.shape[1]
    n_layers = surface.shape[0]
    layers   = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(9, 5))
    title = (f"Entropy vs layer — each curve = one token position\n"
             f"'{record.prompt}'  |  {_hook_label(record.hook_type)}"
             f"{_dim_annotation(record.d_model)}  |  {_alpha_label(record.alpha)}")
    fig.suptitle(title, fontsize=10)

    for t in range(seq_len):
        curve  = surface[:, t]
        lw     = 2.2 if t == seq_len - 1 else 1.2
        zorder = 5   if t == seq_len - 1 else 2
        ax.plot(layers, curve,
                color=COLORS[t % len(COLORS)],
                linewidth=lw, marker="o", markersize=3,
                label=repr(str_tokens[t]), zorder=zorder)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (bits)")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    if filename is None:
        safe = record.prompt.replace(" ", "_")[:25]
        filename = (f"entropy_vs_layer_{record.hook_type}_"
                    f"{record.norm_key}_a{record.alpha}_{safe}.png")
    _save(fig, output_dir, filename)


# ============================================================================
# 1D: ENTROPY VS TOKEN POSITION  (fixed layer, vary token position)
#
# Physical meaning:
#   Snapshot at one processing depth across the sequence. At layer 0 this
#   is raw embedding geometry. Deeper layers reflect contextual integration.
# ============================================================================

def plot_fixed_layer(
    record:     EntropyRecord,
    output_dir: Path,
    filename:   Optional[str] = None,
    skip_bos:   bool = True,
) -> None:
    """
    Entropy vs token position, one curve per layer.

    Args:
        record:     single EntropyRecord
        output_dir: directory for saved figure
        filename:   output filename (auto-generated if None)
        skip_bos:   skip BOS token (default True)
    """
    surface, str_tokens = _bos_slice(record.surface, record.str_tokens, skip_bos)
    seq_len  = surface.shape[1]
    n_layers = surface.shape[0]
    positions = list(range(seq_len))

    fig, ax = plt.subplots(figsize=(9, 5))
    title = (f"Entropy vs token position — each curve = one layer\n"
             f"'{record.prompt}'  |  {_hook_label(record.hook_type)}"
             f"{_dim_annotation(record.d_model)}  |  {_alpha_label(record.alpha)}")
    fig.suptitle(title, fontsize=10)

    layer_colors = plt.cm.viridis(np.linspace(0.15, 0.95, n_layers))
    for layer in range(n_layers):
        curve = surface[layer, :]
        ax.plot(positions, curve,
                color=layer_colors[layer],
                linewidth=1.2, marker="o", markersize=4,
                label=f"L{layer}", alpha=0.8)

    ax.set_xlabel("Token position")
    ax.set_ylabel("Entropy (bits)")
    ax.set_xticks(positions)
    ax.set_xticklabels([repr(t) for t in str_tokens],
                       rotation=30, ha="right", fontsize=8)
    ax.grid(alpha=0.3)

    shown   = [0, n_layers // 2, n_layers - 1]
    handles = [plt.Line2D([0], [0], color=layer_colors[l],
                          linewidth=2, label=f"L{l}") for l in shown]
    ax.legend(handles=handles, fontsize=7, title="Layer",
              bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    if filename is None:
        safe = record.prompt.replace(" ", "_")[:25]
        filename = (f"entropy_vs_position_{record.hook_type}_"
                    f"{record.norm_key}_a{record.alpha}_{safe}.png")
    _save(fig, output_dir, filename)


# ============================================================================
# 2D: FULL ENTROPY SURFACE  (heatmap)
#
# Physical meaning:
#   The complete entropy landscape — rows = layers, columns = token positions.
#   Analogous to a 2D field T(x,y) in a physical domain.
# ============================================================================

def plot_2d_surface(
    record:     EntropyRecord,
    output_dir: Path,
    filename:   Optional[str] = None,
    skip_bos:   bool = True,
    cmap:       str  = "plasma",
) -> None:
    """
    2D heatmap of entropy[layer, token_position].

    Args:
        record:     single EntropyRecord
        output_dir: directory for saved figure
        filename:   output filename (auto-generated if None)
        skip_bos:   skip BOS token (default True)
        cmap:       matplotlib colormap name
    """
    surface, str_tokens = _bos_slice(record.surface, record.str_tokens, skip_bos)
    seq_len  = surface.shape[1]
    n_layers = surface.shape[0]

    fig, ax = plt.subplots(figsize=(max(6, seq_len * 1.2), 5))
    title = (f"Entropy surface — {_alpha_label(record.alpha)}\n"
             f"'{record.prompt}'  |  {_hook_label(record.hook_type)}"
             f"{_dim_annotation(record.d_model)}  |  {record.norm_key} norm")
    fig.suptitle(title, fontsize=10)

    im = ax.imshow(surface, aspect="auto", origin="upper",
                   cmap=cmap, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Entropy (bits)", fontsize=9)

    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels([repr(t) for t in str_tokens],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{l}" for l in range(n_layers)], fontsize=7)
    ax.grid(False)
    plt.tight_layout()

    if filename is None:
        safe = record.prompt.replace(" ", "_")[:25]
        filename = (f"entropy_surface_{record.hook_type}_"
                    f"{record.norm_key}_a{record.alpha}_{safe}.png")
    _save(fig, output_dir, filename)


# ============================================================================
# HOOK COMPARISON  (overlay multiple hook types on one axes)
#
# Scientific purpose:
#   Direct comparison of resid_post vs attn_out vs mlp_out entropy curves.
#   Primary tool for the attention vs MLP decomposition question.
# ============================================================================

def plot_hook_comparison(
    records:        list,
    output_dir:     Path,
    token_position: int  = -1,
    filename:       Optional[str] = None,
) -> None:
    """
    Overlay entropy-vs-layer curves for multiple hook types on one axes.

    Args:
        records:        list of EntropyRecord, one per hook type.
                        Must all share the same prompt, norm_key, alpha.
        output_dir:     directory for saved figure
        token_position: which token position to plot (-1 = final token)
        filename:       output filename (auto-generated if None)

    Typical usage:
        records = [resid_post_rec, attn_out_rec, mlp_out_rec]
        plot_hook_comparison(records, output_dir, token_position=-1)
    """
    if not records:
        return

    prompts   = set(r.prompt   for r in records)
    norm_keys = set(r.norm_key for r in records)
    alphas    = set(r.alpha    for r in records)
    if len(prompts) > 1 or len(norm_keys) > 1 or len(alphas) > 1:
        raise ValueError(
            "plot_hook_comparison: all records must share prompt, norm_key, alpha.\n"
            f"  prompts={prompts}, norm_keys={norm_keys}, alphas={alphas}"
        )

    ref      = records[0]
    n_layers = ref.n_layers
    layers   = list(range(n_layers))
    t_idx    = token_position  # -1 handled naturally by numpy indexing
    t_label  = repr(ref.str_tokens[token_position])

    fig, ax = plt.subplots(figsize=(9, 5))
    title = (f"Hook comparison — entropy vs layer\n"
             f"'{ref.prompt}'  |  token {t_label}  |  "
             f"{ref.norm_key} norm  |  {_alpha_label(ref.alpha)}")
    fig.suptitle(title, fontsize=10)

    for record in records:
        curve = record.surface[:, t_idx]
        color = HOOK_COLORS.get(record.hook_type, "gray")
        label = _hook_label(record.hook_type) + _dim_annotation(record.d_model)
        ax.plot(layers, curve,
                color=color, linewidth=2.0, marker="o", markersize=4,
                label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Entropy (bits)")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    if filename is None:
        hooks = "_vs_".join(r.hook_type for r in records)
        safe  = ref.prompt.replace(" ", "_")[:20]
        filename = f"hook_comparison_{hooks}_{safe}.png"
    _save(fig, output_dir, filename)


# ============================================================================
# CORPUS: MEAN ± 1σ ACROSS ALL PAIRS
# ============================================================================

def plot_overall_mean(
    entropy_records: list,
    alphas:          list,
    output_dir:      Path,
    model_name:      str = "",
    hook_type:       str = "resid_post",
) -> None:
    """
    One row per normalization method, one column per alpha.
    Each subplot: mean ± 1σ entropy at final token, base vs contrast.

    Args:
        entropy_records: flat list of EntropyRecord
        alphas:          list of alpha values to plot
        output_dir:      directory for saved figure
        model_name:      used in title and filename
        hook_type:       which hook type to plot (default "resid_post")
    """
    norm_keys   = [key for key, _, _ in NORM_METHODS]
    norm_labels = {key: label for key, label, _ in NORM_METHODS}

    sample = filter_records(entropy_records, hook_type=hook_type,
                            norm_key=norm_keys[0], alpha=alphas[0])
    if not sample:
        print(f"  ⚠ No records found for hook_type='{hook_type}'")
        return

    n_layers = sample[0].n_layers
    layers   = list(range(n_layers))
    n_rows   = len(norm_keys)
    n_cols   = len(alphas)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharey=False, squeeze=False)

    title = (f"Residual stream entropy — all categories (mean ± 1σ, final token)\n"
             f"{_hook_label(hook_type)}")
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=11)

    for row, nk in enumerate(norm_keys):
        for col, alpha in enumerate(alphas):
            ax = axes[row][col]

            for role, color, ls in [("base",     "#1565C0", "-"),
                                     ("contrast", "#B71C1C", "--")]:
                subset = filter_records(entropy_records, hook_type=hook_type,
                                        norm_key=nk, alpha=alpha, role=role)
                curves = [r.final_token_curve() for r in subset]
                if not curves:
                    continue
                arr  = np.array(curves)
                mean = arr.mean(axis=0)
                std  = arr.std(axis=0)
                ax.plot(layers, mean, color=color, linestyle=ls,
                        linewidth=2.0, label=f"{role} mean")
                ax.fill_between(layers, mean - std, mean + std,
                                color=color, alpha=0.15)

            ax.set_title(f"{norm_labels[nk]}\n{_alpha_label(alpha)}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Entropy (bits)")
            ax.set_xticks(layers)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    _save(fig, output_dir, f"residual_entropy_overall_{hook_type}{suffix}.png")


# ============================================================================
# CORPUS: PER-CATEGORY BASE VS CONTRAST
# ============================================================================

def plot_category(
    entropy_records: list,
    category:        str,
    alphas:          list,
    output_dir:      Path,
    model_name:      str = "",
    hook_type:       str = "resid_post",
) -> None:
    """
    Per-category plot. One subplot per normalization × alpha.
    Base=solid, contrast=dashed. Bold lines=mean across pairs.

    Args:
        entropy_records: flat list of EntropyRecord
        category:        corpus category string to filter to
        alphas:          list of alpha values
        output_dir:      directory for saved figure
        model_name:      used in title and filename
        hook_type:       which hook type to plot
    """
    norm_keys   = [key for key, _, _ in NORM_METHODS]
    norm_labels = {key: label for key, label, _ in NORM_METHODS}

    cat_records = filter_records(entropy_records, hook_type=hook_type,
                                 category=category)
    if not cat_records:
        print(f"  ⚠ No records for category='{category}', hook='{hook_type}'")
        return

    by_pair = defaultdict(lambda: defaultdict(list))
    for r in cat_records:
        if r.pair_id and r.role:
            by_pair[r.pair_id][r.role].append(r)

    n_layers = cat_records[0].n_layers
    layers   = list(range(n_layers))
    n_rows   = len(norm_keys)
    n_cols   = len(alphas)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharey=False, squeeze=False)

    title = (f"Residual stream entropy — {category} (final token)\n"
             f"{_hook_label(hook_type)}")
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=11)

    for row, nk in enumerate(norm_keys):
        for col, alpha in enumerate(alphas):
            ax = axes[row][col]
            all_base, all_contrast = [], []

            for pair_idx, (pair_id, roles) in enumerate(sorted(by_pair.items())):
                color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
                for role, ls in [("base", "-"), ("contrast", "--")]:
                    if role not in roles:
                        continue
                    matching = filter_records(roles[role], norm_key=nk, alpha=alpha)
                    if not matching:
                        continue
                    curve = matching[0].final_token_curve()
                    ax.plot(layers, curve, color=color, linestyle=ls,
                            linewidth=1.0, alpha=0.5)
                    (all_base if role == "base" else all_contrast).append(curve)

            if all_base:
                ax.plot(layers, np.mean(all_base, axis=0),
                        color="black", linestyle="-", linewidth=2.2,
                        label="mean base", zorder=5)
            if all_contrast:
                ax.plot(layers, np.mean(all_contrast, axis=0),
                        color="dimgray", linestyle="--", linewidth=2.2,
                        label="mean contrast", zorder=5)

            ax.set_title(f"{norm_labels[nk]}\n{_alpha_label(alpha)}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Entropy (bits)")
            ax.set_xticks(layers)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    _save(fig, output_dir,
          f"residual_entropy_{category}_{hook_type}{suffix}.png")
