"""workflows/ablation_analysis.py — Interventional ablation of the r⊥ complement.

Removes the orthogonal complement r⊥ from the residual stream and measures
how token predictions degrade, testing whether base prompts invest more
computationally meaningful content in r⊥ than contrast prompts.

Two stages, run independently or together:

    Stage 1 (posthoc): Post-hoc projection ablation on stored activations.
        No additional forward passes. For each (prompt, layer, k), projects
        resid_post onto top-k W_U singular directions, re-applies ln_final
        and W_U, and compares to the full prediction. This is the primary
        experiment and runs by default.

    Stage 2 (intervention): Live forward pass ablation via TransformerLens
        hooks. Zeroes r⊥ at a specific layer during the forward pass and
        measures the downstream effect at the final layer. Requires a live
        model and one forward pass per (prompt, layer, k) combination.
        Enabled with --stage2. Layers are specified with --intervention-layers
        or auto-selected from the entropy crossover peak.

Pipeline:
    extraction.extract_corpus()                    -> dict[hook, list[ActivationRecord]]
    ablation_compute.compute_wu_svd(W_U)                   -> Vh (one-time)
    ablation_compute.wu_explained_variance(Vh, W_U, ks)    -> diagnostic table
    ablation_compute.validate_ablation(W_U, Vh, d_model)   -> sanity checks
    _run_posthoc_corpus()                          -> list[AblationRecord]
    _run_intervention_corpus()  (optional)         -> list[AblationRecord]
    ablation_plots.*                               -> figures

Usage:
    # Basic posthoc ablation (Stage 1 only, the primary experiment)
    python workflows/ablation_analysis.py --corpus corpus.json

    # Specify model, k values, and output directory
    python workflows/ablation_analysis.py --corpus corpus.json \\
        --model gpt2-small --k 10 50 100 200 400 600 \\
        --output-dir figures/ablation

    # Include Stage 2 intervention at specific layers
    python workflows/ablation_analysis.py --corpus corpus.json \\
        --stage2 --intervention-layers 4 8 10

    # Just Stage 1 with k values chosen by explained variance thresholds
    python workflows/ablation_analysis.py --corpus corpus.json \\
        --ev-thresholds 0.50 0.75 0.90 0.95 0.99

    # Filter to a single corpus category
    python workflows/ablation_analysis.py --corpus corpus.json \\
        --category pattern

    # Compute without plotting (save AblationRecords for later)
    python workflows/ablation_analysis.py --corpus corpus.json \\
        --save-data --no-plots
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
from ablation_compute import (
    AblationRecord,
    compute_wu_svd,
    wu_explained_variance,
    validate_ablation,
    compute_posthoc_ablation,
    compute_intervention_ablation,
)
from ablation_plots import (
    plot_kl_vs_layer,
    plot_kl_vs_k,
    plot_top1_preservation,
    plot_entropy_change_vs_layer,
    plot_entropy_vs_layer,
    plot_intervention_heatmap,
    _safe_model_name,
)


DEFAULT_HOOKS    = ["resid_post"]
DEFAULT_K_VALUES = [10, 50, 100, 200, 400, 600]

# Default intervention layers per model family.
# Layer 8 is the entropy crossover peak for GPT-2 small (12 layers).
# For deeper models, scale proportionally: peak ≈ 2/3 depth.
DEFAULT_INTERVENTION_LAYERS = {
    "gpt2-small":  [4, 8, 10],      # 12 layers: peak at 8
    "gpt2-medium": [8, 16, 22],     # 24 layers: peak ~16
    "gpt2-large":  [12, 24, 34],    # 36 layers: peak ~24
    "gpt2-xl":     [16, 32, 46],    # 48 layers: peak ~32
}


# ============================================================================
# CORPUS ITERATION HELPERS
# Same orchestration pattern as wu_subspace_analysis.py — loops live here,
# computation functions live in ablation_compute.py.
# ============================================================================

def _run_posthoc_corpus(
    activation_records: list,
    W_U:                torch.Tensor,
    ln_final,
    Vh:                 torch.Tensor,
    k_values:           list[int],
    alpha:              float = 1.0,
) -> list:
    """
    Iterate compute_posthoc_ablation() over a list of ActivationRecords.

    This is the primary experiment: no forward pass required, operates
    purely on stored activations. Each prompt yields len(k_values)
    AblationRecords.

    Args:
        activation_records: list of ActivationRecord (same hook type)
        W_U:      unembedding matrix from model.W_U
        ln_final: model.ln_final callable
        Vh:       right singular vectors from compute_wu_svd()
        k_values: list of subspace ranks to sweep
        alpha:    Renyi alpha for entropy change (default 1.0 = Shannon)

    Returns:
        flat list of AblationRecord
    """
    all_records = []
    n = len(activation_records)
    for i, record in enumerate(activation_records):
        results = compute_posthoc_ablation(
            record, W_U, ln_final, Vh, k_values, alpha
        )
        all_records.extend(results)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    Posthoc ablation: {i+1}/{n} prompts...")
    return all_records


def _run_intervention_corpus(
    activation_records:  list,
    model,
    W_U:                 torch.Tensor,
    ln_final,
    Vh:                  torch.Tensor,
    k_values:            list[int],
    intervention_layers: list[int],
    alpha:               float = 1.0,
) -> list:
    """
    Iterate compute_intervention_ablation() over a list of ActivationRecords.

    Stage 2: requires a live model. Each prompt yields
    len(k_values) × len(intervention_layers) AblationRecords.

    Args:
        activation_records:  list of ActivationRecord
        model:               live HookedTransformer
        W_U:                 unembedding matrix
        ln_final:            model.ln_final callable
        Vh:                  right singular vectors
        k_values:            subspace ranks to sweep
        intervention_layers: which layers to intervene at
        alpha:               Renyi alpha (default 1.0)

    Returns:
        flat list of AblationRecord
    """
    all_records = []
    n = len(activation_records)
    n_combos = len(k_values) * len(intervention_layers)
    for i, record in enumerate(activation_records):
        results = compute_intervention_ablation(
            record, model, W_U, ln_final, Vh,
            k_values, intervention_layers, alpha
        )
        all_records.extend(results)
        if (i + 1) % 5 == 0 or (i + 1) == n:
            print(f"    Intervention ablation: {i+1}/{n} prompts "
                  f"({n_combos} (layer,k) combos each)...")
    return all_records


# ============================================================================
# SERIALIZATION
# Save/load AblationRecords to .npz for reproducibility and later plotting.
# Follows the same pattern as save_entropy_records in entropy_compute.py.
# ============================================================================

def save_ablation_records(records: list, path) -> None:
    """Save a list of AblationRecords to a single .npz file.

    Handles variable array lengths across posthoc (n_layers) and
    intervention (length 1) records by padding to a common shape.
    """
    n = len(records)
    if n == 0:
        print(f"  No records to save.")
        return

    # Find max array length for padding
    max_len = max(len(r.kl_divergence) for r in records)

    # Pad each array to max_len with NaN
    kl_padded   = np.full((n, max_len), np.nan, dtype=np.float64)
    ent_padded  = np.full((n, max_len), np.nan, dtype=np.float64)
    top1_padded = np.full((n, max_len), False)
    arr_lens    = np.zeros(n, dtype=np.int32)

    for i, r in enumerate(records):
        length = len(r.kl_divergence)
        kl_padded[i, :length]   = r.kl_divergence
        ent_padded[i, :length]  = r.entropy_change
        top1_padded[i, :length] = r.top1_preserved
        arr_lens[i]             = length

    ks                = np.array([r.k                  for r in records], dtype=np.int32)
    ablation_types    = np.array([r.ablation_type       for r in records], dtype=object)
    intervention_lyrs = np.array([r.intervention_layer if r.intervention_layer is not None
                                  else -1              for r in records], dtype=np.int32)
    model_names       = np.array([r.model_name         for r in records], dtype=object)
    prompts           = np.array([r.prompt              for r in records], dtype=object)
    roles             = np.array([r.role                for r in records], dtype=object)
    categories        = np.array([r.category or ""      for r in records], dtype=object)
    hook_types        = np.array([r.hook_type            for r in records], dtype=object)

    np.savez(
        path,
        kl_divergence      = kl_padded,
        entropy_change     = ent_padded,
        top1_preserved     = top1_padded,
        arr_lens           = arr_lens,
        ks                 = ks,
        ablation_types     = ablation_types,
        intervention_lyrs  = intervention_lyrs,
        model_names        = model_names,
        prompts            = prompts,
        roles              = roles,
        categories         = categories,
        hook_types         = hook_types,
    )
    print(f"  Saved {n} AblationRecords to {path}")


def load_ablation_records(path) -> list:
    """Load a list of AblationRecords from a .npz file."""
    d = np.load(path, allow_pickle=True)
    n = len(d["prompts"])
    records = []
    for i in range(n):
        length = int(d["arr_lens"][i])
        int_lyr = int(d["intervention_lyrs"][i])

        records.append(AblationRecord(
            kl_divergence      = d["kl_divergence"][i, :length],
            entropy_change     = d["entropy_change"][i, :length],
            top1_preserved     = d["top1_preserved"][i, :length],
            k                  = int(d["ks"][i]),
            ablation_type      = str(d["ablation_types"][i]),
            intervention_layer = int_lyr if int_lyr >= 0 else None,
            model_name         = str(d["model_names"][i]),
            prompt             = str(d["prompts"][i]),
            role               = str(d["roles"][i]),
            category           = str(d["categories"][i]) or None,
            hook_type          = str(d["hook_types"][i]),
        ))
    return records


# ============================================================================
# SUMMARY PRINTING
# Quick diagnostic output showing mean KL, entropy change, and top-1
# preservation across prompt categories for each k value.
# ============================================================================

def print_ablation_summary(
    records:       list,
    k_values:      list[int],
    ablation_type: str = "posthoc",
) -> None:
    """
    Print mean ablation metrics per category and k value at the final layer.

    For posthoc records, "final layer" is index [-1] of the per-layer array.
    For intervention records, there is only one value (the final layer effect).

    Args:
        records:       list of AblationRecord
        k_values:      k values to summarize
        ablation_type: "posthoc" or "intervention"
    """
    filtered = [r for r in records if r.ablation_type == ablation_type]
    if not filtered:
        print(f"\n  No {ablation_type} records to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY ({ablation_type}, final layer)")
    print(f"{'='*70}")

    # Group by (role, k)
    grouped = defaultdict(list)
    for r in filtered:
        grouped[(r.role, r.k)].append(r)

    for k in k_values:
        print(f"\n  k = {k}:")
        for cat in ["base", "contrast"]:
            recs = grouped.get((cat, k), [])
            if not recs:
                continue
            # Use final layer value (index -1)
            kl_vals   = [r.kl_divergence[-1]            for r in recs]
            ent_vals  = [r.entropy_change[-1]            for r in recs]
            top1_vals = [r.top1_preserved[-1].item()     for r in recs]

            n = len(recs)
            print(f"    {cat.upper():>8s} ({n:2d} prompts): "
                  f"KL={np.mean(kl_vals):7.4f} ± {np.std(kl_vals, ddof=1)/np.sqrt(n):.4f}  "
                  f"ΔH={np.mean(ent_vals):+7.4f} ± {np.std(ent_vals, ddof=1)/np.sqrt(n):.4f}  "
                  f"top1={np.mean(top1_vals):.2%}")

        # Print base-contrast difference at this k
        base_kl = [r.kl_divergence[-1] for r in grouped.get(("base", k), [])]
        cont_kl = [r.kl_divergence[-1] for r in grouped.get(("contrast", k), [])]
        if base_kl and cont_kl:
            diff = np.mean(base_kl) - np.mean(cont_kl)
            direction = "base > contrast" if diff > 0 else "contrast > base"
            print(f"    {'DIFF':>8s}:            "
                  f"ΔKL(base−contrast) = {diff:+.4f}  ({direction})")

    print()


# ============================================================================
# K-VALUE SELECTION FROM EXPLAINED VARIANCE THRESHOLDS
# ============================================================================

def k_values_from_ev_thresholds(
    W_U:        torch.Tensor,
    thresholds: list[float],
    d_model:    int,
) -> list[int]:
    """
    Find the smallest k that achieves each explained variance threshold.

    Scans the full singular value spectrum and returns one k per threshold.
    Useful for choosing k values in principled terms rather than arbitrary
    integer ranks.

    Args:
        W_U:        unembedding matrix [d_model, vocab_size]
        thresholds: list of target explained variance fractions, e.g. [0.5, 0.9, 0.99]
        d_model:    model dimension

    Returns:
        sorted list of unique k values (one per threshold, deduplicated)
    """
    _, S, _ = torch.linalg.svd(W_U.T.float().cpu(), full_matrices=False)
    cumvar = (S**2).cumsum(dim=0) / (S**2).sum()

    k_values = set()
    for thresh in thresholds:
        # Find first k where cumulative variance >= threshold
        mask = cumvar >= thresh
        if mask.any():
            k = int(mask.nonzero(as_tuple=True)[0][0].item()) + 1  # 1-indexed
            k = min(k, d_model - 1)  # k must be < d_model
            k_values.add(k)
        else:
            k_values.add(d_model - 1)

    return sorted(k_values)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ablation analysis: interventional test of r⊥ content"
    )

    # ── Required ──
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to corpus JSON from corpus_gen.py")

    # ── Model and hooks ──
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--hooks", type=str, nargs="+", default=DEFAULT_HOOKS,
                        help=f"Hook types to extract. Choices: {sorted(HOOK_TYPES.keys())}")

    # ── K values ──
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="Subspace ranks to sweep (default: auto-scaled to d_model)")
    parser.add_argument("--ev-thresholds", type=float, nargs="+", default=None,
                        help="Choose k values by explained variance thresholds "
                             "(e.g. 0.50 0.75 0.90 0.95 0.99). "
                             "Overrides --k if both specified.")

    # ── Alpha ──
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Renyi alpha for entropy change (default: 1.0 = Shannon)")

    # ── Stage 2 options ──
    parser.add_argument("--stage2", action="store_true",
                        help="Run Stage 2 intervention ablation (requires live model, "
                             "slow: one forward pass per (prompt, layer, k))")
    parser.add_argument("--intervention-layers", type=int, nargs="+", default=None,
                        help="Layers to intervene at for Stage 2 "
                             "(default: auto from entropy crossover peak)")

    # ── Filtering ──
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")

    # ── Output ──
    parser.add_argument("--output-dir", type=str, default="figures/ablation",
                        help="Directory for plots and saved data")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--save-data", action="store_true",
                        help="Save AblationRecords to .npz for later analysis")
    parser.add_argument("--load-data", type=str, default=None,
                        help="Load precomputed AblationRecords from .npz "
                             "(skips extraction and computation, goes straight to plots)")

    # ── Device ──
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hook_types = args.hooks
    for ht in hook_types:
        if ht not in HOOK_TYPES:
            print(f"Unknown hook type '{ht}'. Supported: {sorted(HOOK_TYPES.keys())}")
            return 1

    # ==================================================================
    # FAST PATH: load precomputed records and skip to plotting
    # ==================================================================
    if args.load_data is not None:
        print(f"\nLoading precomputed AblationRecords from {args.load_data}...")
        all_ablation_records = load_ablation_records(args.load_data)
        print(f"  Loaded {len(all_ablation_records)} records.")

        # Infer k_values and model_name from loaded records
        k_values   = sorted(set(r.k for r in all_ablation_records))
        model_name = all_ablation_records[0].model_name if all_ablation_records else args.model

        # Recompute ev_dict — need W_U for this, but if we're in load-only
        # mode we may not have the model. Set to None and skip EV annotations.
        ev_dict = None
        print(f"  k values: {k_values}")
        print(f"  (Explained variance annotations unavailable in load-data mode)")

        # Summary
        posthoc_recs = [r for r in all_ablation_records if r.ablation_type == "posthoc"]
        interv_recs  = [r for r in all_ablation_records if r.ablation_type == "intervention"]
        if posthoc_recs:
            print_ablation_summary(all_ablation_records, k_values, "posthoc")
        if interv_recs:
            print_ablation_summary(all_ablation_records, k_values, "intervention")

        # Skip to plotting
        if not args.no_plots:
            _generate_plots(all_ablation_records, k_values, ev_dict,
                            model_name, output_dir, args)

        print(f"\nDone. Results in {output_dir}/\n")
        return 0

    # ==================================================================
    # FULL PATH: load model, extract, compute, plot
    # ==================================================================

    # ── Load corpus ──
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        print("  Run: python corpus_gen.py")
        return 1

    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"\nLoaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")

    # ── Load model ──
    print(f"\nLoading model '{args.model}'...")
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    d_model  = model.cfg.d_model
    n_layers = model.cfg.n_layers
    print(f"  Model ready on {cfg['device']}")
    print(f"  Layers:  {n_layers}")
    print(f"  d_model: {d_model}")
    print(f"  Hooks:   {hook_types}")
    print(f"  Alpha:   {args.alpha}")

    # ── Determine k values ──
    if args.ev_thresholds is not None:
        # Choose k from explained variance thresholds
        W_U_detached = model.W_U.detach()
        k_values = k_values_from_ev_thresholds(
            W_U_detached, args.ev_thresholds, d_model
        )
        print(f"  k values from EV thresholds {args.ev_thresholds}:")
        print(f"    {k_values}")
    elif args.k is not None:
        k_values = sorted(set(args.k))
    else:
        # Auto-scale defaults to model's d_model
        k_values = [k for k in DEFAULT_K_VALUES if k < d_model]
        if not k_values:
            k_values = sorted(set(
                max(1, int(f * d_model))
                for f in [0.05, 0.1, 0.25, 0.5, 0.75]
            ))

    # Final validation: k must satisfy 1 <= k < d_model
    k_values = [k for k in k_values if 1 <= k < d_model]
    if not k_values:
        print(f"No valid k values for d_model={d_model}. Aborting.")
        return 1
    print(f"  k values: {k_values}")

    # ── W_U SVD precomputation (one-time cost) ──
    W_U      = model.W_U.detach().cpu()
    ln_final = model.ln_final

    print(f"\nComputing SVD of W_U ({W_U.shape[0]}×{W_U.shape[1]})...")
    Vh = compute_wu_svd(W_U)
    print(f"  Vh shape: {list(Vh.shape)}")

    ev_dict = wu_explained_variance(Vh, W_U, k_values)
    print(f"\n  Explained variance by top-k singular directions:")
    for k in k_values:
        print(f"    k={k:4d}: {ev_dict[k]:.4f} ({ev_dict[k]:.1%})")

    # ── Validation ──
    print(f"\nRunning ablation validation checks...")
    validate_ablation(W_U, Vh, d_model)

    # ── Quick KL sanity check: k=d_model should give KL ≈ 0 ──
    # We'll verify this after computing the first posthoc record below.

    # ── Extraction ──
    print(f"\nExtracting activations across corpus...")
    activation_dict = extract_corpus(
        model, corpus, hook_types,
        model_name=args.model,
        device=cfg["device"],
        category_filter=args.category,
    )

    # ── Computation ──
    all_ablation_records = []

    for ht in hook_types:
        act_records = activation_dict[ht]
        print(f"\n  Hook '{ht}' ({len(act_records)} prompts):")

        # ── Stage 1: posthoc ablation ──
        print(f"\n  Stage 1: posthoc ablation...")
        posthoc_records = _run_posthoc_corpus(
            act_records, W_U, ln_final, Vh, k_values, alpha=args.alpha
        )
        all_ablation_records.extend(posthoc_records)

        # ── KL sanity check at k ≈ d_model ──
        # Verify that full-rank "ablation" gives near-zero KL
        print(f"\n  Sanity check: running k={d_model-1} on first prompt...")
        sanity_records = compute_posthoc_ablation(
            act_records[0], W_U, ln_final, Vh,
            k_values=[d_model - 1], alpha=args.alpha
        )
        max_kl = sanity_records[0].kl_divergence.max()
        print(f"    Max KL at k={d_model-1}: {max_kl:.2e} "
              f"({'PASS' if max_kl < 1e-3 else 'WARNING: expected ~0'})")

        # ── Stage 2: intervention ablation (optional) ──
        if args.stage2:
            # Determine intervention layers
            if args.intervention_layers is not None:
                int_layers = sorted(args.intervention_layers)
            elif args.model in DEFAULT_INTERVENTION_LAYERS:
                int_layers = DEFAULT_INTERVENTION_LAYERS[args.model]
            else:
                # Auto-select: ~1/3, ~2/3, and n-1 (final) of depth
                int_layers = sorted(set([
                    n_layers // 3,
                    2 * n_layers // 3,
                    n_layers - 1,
                ]))

            # Validate layer indices
            int_layers = [L for L in int_layers if 0 <= L < n_layers]
            if not int_layers:
                print(f"  No valid intervention layers for n_layers={n_layers}. "
                      f"Skipping Stage 2.")
            else:
                print(f"\n  Stage 2: intervention ablation at layers {int_layers}...")
                n_combos = len(k_values) * len(int_layers)
                n_total  = len(act_records) * n_combos
                print(f"    ({len(act_records)} prompts × {n_combos} combos "
                      f"= {n_total} forward passes)")

                intervention_records = _run_intervention_corpus(
                    act_records, model, W_U, ln_final, Vh,
                    k_values, int_layers, alpha=args.alpha
                )
                all_ablation_records.extend(intervention_records)

    print(f"\n  Total AblationRecords: {len(all_ablation_records)}")

    # ── Summary ──
    posthoc_recs = [r for r in all_ablation_records if r.ablation_type == "posthoc"]
    interv_recs  = [r for r in all_ablation_records if r.ablation_type == "intervention"]

    if posthoc_recs:
        print_ablation_summary(all_ablation_records, k_values, "posthoc")
    if interv_recs:
        print_ablation_summary(all_ablation_records, k_values, "intervention")

    # ── Save ──
    if args.save_data:
        data_path = output_dir / f"ablation_records_{args.model}.npz"
        save_ablation_records(all_ablation_records, data_path)

    # ── Plots ──
    if not args.no_plots:
        _generate_plots(all_ablation_records, k_values, ev_dict,
                        args.model, output_dir, args)

    print(f"\nDone. Results in {output_dir}/\n")
    return 0


# ============================================================================
# PLOT GENERATION
# Separated from main() so it can be called from both the full path
# and the load-data fast path.
# ============================================================================

def _generate_plots(
    all_records: list,
    k_values:    list[int],
    ev_dict:     dict | None,
    model_name:  str,
    output_dir:  Path,
    args,
) -> None:
    """Generate all ablation figures."""

    safe_name = _safe_model_name(model_name)

    posthoc_recs = [r for r in all_records if r.ablation_type == "posthoc"]
    interv_recs  = [r for r in all_records if r.ablation_type == "intervention"]

    if posthoc_recs:
        print(f"\nGenerating posthoc ablation plots in {output_dir}/...")

        # Figure 1: KL divergence vs layer
        plot_kl_vs_layer(
            posthoc_recs, model_name, k_values, ev_dict=ev_dict,
            save_path=str(output_dir / f"ablation_kl_vs_layer_{safe_name}.png"),
        )

        # Figure 2: KL divergence vs k at fixed layers
        # Choose representative layers: early, middle, late
        if posthoc_recs:
            n_layers = len(posthoc_recs[0].kl_divergence)
            # Default fixed layers: 1/4, 1/2, 3/4 of depth, plus final
            fixed_layers = sorted(set([
                n_layers // 4,
                n_layers // 2,
                3 * n_layers // 4,
                n_layers - 1,
            ]))
            layer_str = '_'.join(str(l) for l in fixed_layers)
            plot_kl_vs_k(
                posthoc_recs, model_name, fixed_layers, k_values,
                ev_dict=ev_dict,
                save_path=str(output_dir / f"ablation_kl_vs_k_layers{layer_str}_{safe_name}.png"),
            )

        # Figure 3: Top-1 preservation
        plot_top1_preservation(
            posthoc_recs, model_name, k_values, ev_dict=ev_dict,
            save_path=str(output_dir / f"ablation_top1_preservation_{safe_name}.png"),
        )

        # Figure 4: Entropy change vs layer
        plot_entropy_change_vs_layer(
            posthoc_recs, model_name, k_values, ev_dict=ev_dict,
            save_path=str(output_dir / f"ablation_entropy_change_{safe_name}.png"),
        )

        # Figure 5: Entropy vs layer
        plot_entropy_vs_layer(
            posthoc_recs, model_name, k_values, ev_dict=ev_dict,
            save_path=str(output_dir / f"ablation_entropy_{safe_name}.png"),
        )

    if interv_recs:
        print(f"\nGenerating intervention ablation plots in {output_dir}/...")

        # Figure 5: Intervention heatmaps for each metric
        for metric in ["kl_divergence", "entropy_change", "top1_preserved"]:
            plot_intervention_heatmap(
                interv_recs, model_name, metric=metric,
                save_path=str(output_dir / f"ablation_intervention_heatmap_{metric}_{safe_name}.png"),
            )


if __name__ == "__main__":
    sys.exit(main() or 0)
