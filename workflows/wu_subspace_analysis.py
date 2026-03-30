"""workflows/wu_subspace_analysis.py — W_U subspace entropy analysis.

Decomposes residual stream activations into the W_U prediction subspace (r‖)
and its orthogonal complement (r⊥) via SVD, then measures entropy in each
subspace across a corpus of base/contrast prompt pairs.

Tests the subspace segregation hypothesis: base prompts invest more in r⊥
(structured working memory invisible to token prediction) while concentrating
r‖ (yielding lower logit lens entropy). The k-sweep reveals at what spectral
rank the prompt-differentiating signal lives in W_U's singular value spectrum.

Pipeline:
    extraction.extract_corpus()                     -> dict[hook, list[ActivationRecord]]
    computation.compute_wu_svd(W_U)                 -> Vh (one-time)
    computation.wu_explained_variance(W_U, k_values)-> diagnostic table
    _run_wu_subspace_corpus()                       -> list[EntropyRecord]
    _run_residual_stream_corpus()  (optional)       -> list[EntropyRecord]
    _run_logit_lens_corpus()       (optional)       -> list[EntropyRecord]
    plot_explained_variance()                       -> explained variance bar chart
    plot_k_sweep_difference()                       -> k-sweep summary figures
    plot_subspace_comparison()                      -> r‖ vs r⊥ vs full comparison

Usage:
    # Basic W_U subspace analysis
    python workflows/wu_subspace_analysis.py --corpus corpus.json

    # Specify model and k values
    python workflows/wu_subspace_analysis.py --corpus corpus.json --model pythia-1b --k 10 50 100 200

    # Include residual stream and logit lens for direct comparison
    python workflows/wu_subspace_analysis.py --corpus corpus.json --also-residual --also-logit-lens

    # Just compute and save, no plots
    python workflows/wu_subspace_analysis.py --corpus corpus.json --save-data --no-plots

    # Filter to a single corpus category
    python workflows/wu_subspace_analysis.py --corpus corpus.json --category pattern
"""

import sys
import json
import argparse
import warnings
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import load_model_and_sae, MODEL_CONFIGS
from extraction import extract_corpus, HOOK_TYPES
from computation import (
    compute_wu_svd,
    wu_explained_variance,
    compute_wu_subspace_entropy,
    compute_residual_stream_entropy,
    compute_logit_lens_entropy,
    filter_records,
    print_summary,
    save_entropy_records,
    RESIDUAL_NORM_KEYS,
    NORM_LABELS,
)
from entropy_plots import (
    _alpha_label,
    _hook_label,
    _save,
)

DEFAULT_HOOKS    = ["resid_post"]
DEFAULT_K_VALUES = [10, 50, 100, 200, 400, 600]


# ============================================================================
# CORPUS ITERATION HELPERS
# Same pattern as corpus_analysis.py — orchestration loops live here.
# ============================================================================

def _run_wu_subspace_corpus(
    activation_records: list,
    alphas:             list,
    Vh:                 torch.Tensor,
    k_values:           list,
) -> list:
    """
    Iterate compute_wu_subspace_entropy() over a list of ActivationRecords.

    Vh is precomputed once from W_U via compute_wu_svd() in main().

    Args:
        activation_records: list of ActivationRecord (same hook type)
        alphas:             list of Renyi alpha values
        Vh:                 right singular vectors of W_U, from compute_wu_svd()
        k_values:           list of subspace ranks to sweep

    Returns:
        flat list of EntropyRecord with norm_keys "wu_parallel_k{k}",
        "wu_orthogonal_k{k}"
    """
    all_entropy = []
    n = len(activation_records)
    for i, record in enumerate(activation_records):
        records = compute_wu_subspace_entropy(record, alphas, Vh, k_values)
        all_entropy.extend(records)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    W_U subspace:    {i+1}/{n} prompts...")
    return all_entropy


def _run_residual_stream_corpus(
    activation_records: list,
    alphas:             list,
    norm_keys:          list,
) -> list:
    """Iterate compute_residual_stream_entropy() over a list of ActivationRecords."""
    all_entropy = []
    n = len(activation_records)
    for i, record in enumerate(activation_records):
        records = compute_residual_stream_entropy(record, alphas, norm_keys)
        all_entropy.extend(records)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    Residual stream: {i+1}/{n} prompts...")
    return all_entropy


def _run_logit_lens_corpus(
    activation_records: list,
    alphas:             list,
    W_U:                torch.Tensor,
    ln_final,
) -> list:
    """Iterate compute_logit_lens_entropy() over a list of ActivationRecords."""
    all_entropy = []
    n = len(activation_records)
    for i, record in enumerate(activation_records):
        records = compute_logit_lens_entropy(record, alphas, W_U, ln_final)
        all_entropy.extend(records)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    Logit lens:      {i+1}/{n} prompts...")
    return all_entropy


# ============================================================================
# PLOTTING: K-SWEEP SUMMARY
# Shows how the base-contrast entropy difference in r⊥ (and r‖) varies
# as a function of subspace rank k, at each layer.
# This is the central diagnostic for the subspace segregation hypothesis.
# ============================================================================

def plot_k_sweep_difference(
    entropy_records: list,
    k_values:        list,
    alphas:          list,
    output_dir:      Path,
    model_name:      str   = "",
    hook_type:       str   = "resid_post",
    subspace:        str   = "orthogonal",
) -> None:
    """
    K-sweep summary: paired difference (base - contrast) at final token
    as a function of layer, with one curve per k value.

    If the prompt-differentiating signal in r⊥ persists across all k values,
    the orthogonal content is robustly prompt-sensitive regardless of where
    the subspace boundary is drawn — the strong form of the hypothesis.

    Args:
        entropy_records: flat list of EntropyRecord
        k_values:        list of subspace ranks that were swept
        alphas:          list of alpha values to plot (one subplot column each)
        output_dir:      directory for saved figure
        model_name:      used in title and filename
        hook_type:       which hook type (default "resid_post")
        subspace:        "orthogonal" or "parallel"
    """
    from scipy import stats as sp_stats

    n_cols = len(alphas)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                             sharey=False, squeeze=False)

    sub_label = "r⊥" if subspace == "orthogonal" else "r‖"
    title = f"K-sweep: paired Δ(base − contrast) in {sub_label}, final token"
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=11)

    # Color ramp across k values — darker = larger k
    k_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(k_values)))

    for col, alpha in enumerate(alphas):
        ax = axes[0][col]
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        for ki, k in enumerate(k_values):
            nk = f"wu_{subspace}_k{k}"

            base_recs = filter_records(entropy_records, hook_type=hook_type,
                                       norm_key=nk, alpha=alpha, role="base")
            cont_recs = filter_records(entropy_records, hook_type=hook_type,
                                       norm_key=nk, alpha=alpha, role="contrast")

            base_by_pair = {r.pair_id: r for r in base_recs if r.pair_id}
            cont_by_pair = {r.pair_id: r for r in cont_recs if r.pair_id}
            common_ids   = sorted(set(base_by_pair) & set(cont_by_pair))

            if len(common_ids) < 2:
                continue

            deltas = np.array([
                base_by_pair[pid].final_token_curve()
                - cont_by_pair[pid].final_token_curve()
                for pid in common_ids
            ])  # [n_pairs, n_layers]

            n_layers = deltas.shape[1]
            layers   = np.arange(n_layers)
            mean_d   = deltas.mean(axis=0)

            ax.plot(layers, mean_d, color=k_colors[ki], linewidth=1.8,
                    marker="o", markersize=3, label=f"k={k}")

        ax.set_title(f"{_alpha_label(alpha)}", fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Δ Entropy in {sub_label} (bits)")
        if col == 0:
            ax.legend(fontsize=7, title="Subspace rank k")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    _save(fig, output_dir,
          f"k_sweep_{subspace}_{hook_type}{suffix}.png")


# ============================================================================
# PLOTTING: SUBSPACE COMPARISON
# Side-by-side comparison of entropy in r‖, r⊥, full residual stream,
# and (optionally) logit lens, for a single k value.
# This is the "anti-correlation triangle" visualization.
# ============================================================================

def plot_subspace_comparison(
    entropy_records: list,
    k:               int,
    alphas:          list,
    output_dir:      Path,
    model_name:      str   = "",
    hook_type:       str   = "resid_post",
) -> None:
    """
    Paired difference (base - contrast) for r‖, r⊥, and optionally the
    full residual stream ("energy") and logit lens, on a single figure.

    This directly visualizes whether the anti-correlation is explained by
    subspace segregation: r‖ should track with logit lens (base < contrast),
    r⊥ should track with full residual stream (base > contrast).

    Args:
        entropy_records: flat list of EntropyRecord (may include energy,
                         logit_lens, and wu_* records)
        k:               which subspace rank to plot
        alphas:          alpha values (one subplot per alpha)
        output_dir:      directory for saved figure
        model_name:      used in title and filename
        hook_type:       which hook type (default "resid_post")
    """
    from scipy import stats as sp_stats

    # Which norm_keys to plot, in order, with colors and labels
    plot_specs = [
        (f"wu_parallel_k{k}",    f"r‖ (k={k})",                "#E65100"),
        (f"wu_orthogonal_k{k}",  f"r⊥ (k={k})",                "#1B5E20"),
        ("energy",               "Full residual (energy)",      "#1565C0"),
        ("logit_lens",           "Logit lens (token space)",    "#B71C1C"),
    ]

    # Filter to norm_keys that actually exist in the records
    available_nks = set(r.norm_key for r in entropy_records
                        if r.hook_type == hook_type)
    plot_specs = [(nk, lab, col) for nk, lab, col in plot_specs
                  if nk in available_nks]

    if len(plot_specs) < 2:
        print(f"  Subspace comparison: need at least 2 norm_keys, "
              f"found {len(plot_specs)}")
        return

    n_cols = len(alphas)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                             sharey=False, squeeze=False)

    title = f"Subspace comparison: paired Δ(base − contrast), k={k}, final token"
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=11)

    for col, alpha in enumerate(alphas):
        ax = axes[0][col]
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        for nk, label, color in plot_specs:
            base_recs = filter_records(entropy_records, hook_type=hook_type,
                                       norm_key=nk, alpha=alpha, role="base")
            cont_recs = filter_records(entropy_records, hook_type=hook_type,
                                       norm_key=nk, alpha=alpha, role="contrast")

            base_by_pair = {r.pair_id: r for r in base_recs if r.pair_id}
            cont_by_pair = {r.pair_id: r for r in cont_recs if r.pair_id}
            common_ids   = sorted(set(base_by_pair) & set(cont_by_pair))

            if len(common_ids) < 2:
                continue

            deltas = np.array([
                base_by_pair[pid].final_token_curve()
                - cont_by_pair[pid].final_token_curve()
                for pid in common_ids
            ])  # [n_pairs, n_layers]

            n_layers = deltas.shape[1]
            layers   = np.arange(n_layers)
            mean_d   = deltas.mean(axis=0)
            std_d    = deltas.std(axis=0, ddof=1)
            sem_d    = std_d / np.sqrt(len(common_ids))

            ax.plot(layers, mean_d, color=color, linewidth=2.0,
                    marker="o", markersize=3, label=label)
            ax.fill_between(layers, mean_d - sem_d, mean_d + sem_d,
                            color=color, alpha=0.12)

        ax.set_title(f"{_alpha_label(alpha)}", fontsize=9)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δ Entropy (bits)")
        if col == 0:
            ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    _save(fig, output_dir,
          f"subspace_comparison_k{k}_{hook_type}{suffix}.png")


# ============================================================================
# PLOTTING: EXPLAINED VARIANCE SPECTRUM
# ============================================================================

def plot_explained_variance(
    ev:          dict,
    d_model:     int,
    output_dir:  Path,
    model_name:  str = "",
) -> None:
    """
    Bar chart of cumulative explained variance at each k value.
    Annotates how much of W_U's structure the top-k directions capture.

    Args:
        ev:         dict from wu_explained_variance(), mapping k -> fraction
        d_model:    model dimension (for axis labeling)
        output_dir: directory for saved figure
        model_name: used in title and filename
    """
    ks   = sorted(ev.keys())
    vals = [ev[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(ks)), vals, color="#1565C0", alpha=0.8)

    # Annotate each bar with the percentage
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.1%}", ha="center", va="bottom", fontsize=8)

    title = f"W_U explained variance by top-k singular directions (d_model={d_model})"
    if model_name:
        title += f"  [{model_name}]"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Subspace rank k")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.1)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()

    suffix = f"_{model_name}" if model_name else ""
    _save(fig, output_dir, f"wu_explained_variance{suffix}.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="W_U subspace entropy analysis (r‖/r⊥ decomposition)"
    )
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to corpus JSON from corpus_gen.py")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--hooks", type=str, nargs="+", default=DEFAULT_HOOKS,
                        help=f"Hook types to extract. Choices: {sorted(HOOK_TYPES.keys())}")
    parser.add_argument("--alpha", type=float, nargs="+", default=[1.0, 2.0],
                        help="Renyi alpha values (default: Shannon + alpha=2)")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help=f"Subspace ranks to sweep (default: auto-scaled to d_model)")
    parser.add_argument("--also-residual", action="store_true",
                        help="Also compute full residual stream entropy for comparison")
    parser.add_argument("--also-logit-lens", action="store_true",
                        help="Also compute logit-lens entropy for comparison")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")
    parser.add_argument("--output-dir", type=str, default="figures/wu_subspace",
                        help="Directory for plots and saved data")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--save-data", action="store_true",
                        help="Save EntropyRecords to .npz for later analysis")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    alphas     = sorted(set(args.alpha))
    hook_types = args.hooks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ht in hook_types:
        if ht not in HOOK_TYPES:
            print(f"Unknown hook type '{ht}'. Supported: {sorted(HOOK_TYPES.keys())}")
            return 1

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        print("  Run: python corpus_gen.py")
        return 1

    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"\nLoaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model '{args.model}'...")
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    d_model = model.cfg.d_model
    print(f"  Model ready on {cfg['device']}")
    print(f"  Layers:  {model.cfg.n_layers}")
    print(f"  d_model: {d_model}")
    print(f"  Hooks:   {hook_types}")
    print(f"  Alphas:  {alphas}")

    # ── Determine k values ────────────────────────────────────────────────────
    # If not specified, scale default k_values to the model's d_model.
    # Filter out any k >= d_model (required by the decomposition).
    if args.k is not None:
        k_values = sorted(set(args.k))
    else:
        k_values = [k for k in DEFAULT_K_VALUES if k < d_model]
        if not k_values:
            # Very small model: use fractions of d_model
            k_values = sorted(set(
                max(1, int(f * d_model))
                for f in [0.05, 0.1, 0.25, 0.5, 0.75]
            ))

    # Final validation
    k_values = [k for k in k_values if 1 <= k < d_model]
    if not k_values:
        print(f"No valid k values for d_model={d_model}. Aborting.")
        return 1

    print(f"  k values: {k_values}")

    # ── W_U SVD precomputation (one-time cost) ────────────────────────────────
    W_U      = model.W_U.detach()       # [d_model, vocab_size]
    ln_final = model.ln_final           # callable (for optional logit lens)

    print(f"\nComputing SVD of W_U ({W_U.shape[0]}×{W_U.shape[1]})...")
    Vh = compute_wu_svd(W_U)
    print(f"  Vh shape: {list(Vh.shape)}")

    ev = wu_explained_variance(W_U, k_values)
    print(f"\n  Explained variance by top-k singular directions:")
    for k in k_values:
        print(f"    k={k:4d}: {ev[k]:.4f} ({ev[k]:.1%})")

    # ── Extraction ────────────────────────────────────────────────────────────
    print(f"\nExtracting activations across corpus...")
    activation_dict = extract_corpus(
        model, corpus, hook_types,
        model_name=args.model,
        device=cfg["device"],
        category_filter=args.category,
    )

    # ── Computation ───────────────────────────────────────────────────────────
    print(f"\nComputing W_U subspace entropy...")
    all_entropy_records = []

    for ht in hook_types:
        act_records = activation_dict[ht]
        print(f"  Hook '{ht}' ({len(act_records)} prompts):")

        # Primary: W_U subspace decomposition
        wu_records = _run_wu_subspace_corpus(act_records, alphas, Vh, k_values)
        all_entropy_records.extend(wu_records)

        # Optional: full residual stream entropy for comparison
        if args.also_residual:
            rs_records = _run_residual_stream_corpus(
                act_records, alphas, ["energy"]
            )
            all_entropy_records.extend(rs_records)

        # Optional: logit lens entropy for comparison
        if args.also_logit_lens:
            ll_records = _run_logit_lens_corpus(
                act_records, alphas, W_U, ln_final
            )
            all_entropy_records.extend(ll_records)

    print(f"\n  Total EntropyRecords: {len(all_entropy_records)}")

    # Print summary for representative k value (middle of sweep)
    mid_k = k_values[len(k_values) // 2]
    print_summary(all_entropy_records, alphas,
                  norm_key=f"wu_orthogonal_k{mid_k}")
    print_summary(all_entropy_records, alphas,
                  norm_key=f"wu_parallel_k{mid_k}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save_data:
        data_path = output_dir / f"wu_subspace_records_{args.model}.npz"
        save_entropy_records(all_entropy_records, data_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\nGenerating plots in {output_dir}/...")

        for ht in hook_types:

            # 1. Explained variance spectrum
            plot_explained_variance(ev, d_model, output_dir,
                                    model_name=args.model)

            # 2. K-sweep: how does base-contrast difference vary with k?
            #    One plot for r⊥, one for r‖
            plot_k_sweep_difference(
                all_entropy_records, k_values, alphas, output_dir,
                model_name=args.model, hook_type=ht,
                subspace="orthogonal",
            )
            plot_k_sweep_difference(
                all_entropy_records, k_values, alphas, output_dir,
                model_name=args.model, hook_type=ht,
                subspace="parallel",
            )

            # 3. Subspace comparison at each k value:
            #    r‖ vs r⊥ (and full + logit lens if available)
            for k in k_values:
                plot_subspace_comparison(
                    all_entropy_records, k, alphas, output_dir,
                    model_name=args.model, hook_type=ht,
                )

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
