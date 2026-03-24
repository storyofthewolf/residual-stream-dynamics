"""Entropy of raw residual stream activations (geometric diffuseness).

Architecture:
    normalization function  →  probability distribution  →  renyi_entropy(probs, alpha)

Normalization choices:
    normalize_energy    — v²/Σv²      energy weighting (effective rank)
    normalize_abs       — |v|/Σ|v|    linear magnitude weighting
    normalize_softmax   — softmax(v)   exponential weighting

Entropy formula:
    renyi_entropy(probs, alpha)  — generalized entropy
        alpha → 1.0  recovers Shannon entropy exactly
        alpha  < 1   sensitive to small/rare activations
        alpha  > 1   dominated by large activations

Data structure (NEW — 2D entropy surfaces):
    Each result entry stores entropy as a 2D numpy array of shape [n_layers, seq_len].
    Key format: "{norm_key}_alpha_{alpha}"  e.g. "energy_alpha_1.0"
    Token labels stored as "str_tokens": ['<|endoftext|>', 'The', ' wolf', ...]

    Backward-compatible slice for old 1D behavior:
        result["energy_alpha_1.0"][:, -1]   → entropy vs layer at final token
        result["energy_alpha_1.0"][layer, :] → entropy vs token position at fixed layer

Trajectory geometry (PCA, cosine similarity, velocity, speed, acceleration)
lives in residual_stream_dynamics.py.

Plotting lives in entropy_plots.py:
    plot_fixed_position()  — entropy vs layer at one token position (1D)
    plot_fixed_layer()     — entropy vs token position at one layer (1D)
    plot_2d_surface()      — full entropy surface as heatmap/contour (2D)
    plot_overall_mean()    — mean ± 1σ across corpus pairs
    plot_category()        — per-category base vs contrast curves

Usage:
    python residual_stream_entropy.py --corpus corpus.json
    python residual_stream_entropy.py --corpus corpus.json --model gpt2-small
    python residual_stream_entropy.py --corpus corpus.json --alpha 0.5 1.0 2.0 3.0
    python residual_stream_entropy.py --corpus corpus.json --category pattern
    python residual_stream_entropy.py --corpus corpus.json --no-plots
    python residual_stream_entropy.py   # runs on default prompts (no corpus)
"""

import json
import argparse
import warnings
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

# Fallback prompts for quick standalone testing (no corpus required)
DEFAULT_PROMPTS = [
    "After this error adjustment and freezing of attention and normalization nonlinearities"
]


# ============================================================================
# NORMALIZATION FUNCTIONS
# Each takes a raw activation vector and returns a valid probability
# distribution (non-negative, sums to 1) over the d_model dimensions.
# ============================================================================

def normalize_energy(activations: torch.Tensor) -> torch.Tensor:
    """v²/Σv² — energy weighting by squared magnitude.
    Natural physics analog: treats activation dimensions as energy density.
    Primary normalization for effective rank entropy."""
    v = activations.flatten().float()
    v2 = v ** 2
    return (v2 / v2.sum()).clamp(min=1e-12)


def normalize_abs(activations: torch.Tensor) -> torch.Tensor:
    """|v|/Σ|v| — linear magnitude weighting.
    Less aggressive than energy weighting.
    More sensitive to small activations than normalize_energy."""
    v = activations.flatten().float().abs()
    return (v / v.sum()).clamp(min=1e-12)


def normalize_softmax(activations: torch.Tensor) -> torch.Tensor:
    """softmax(v) — exponential weighting.
    Heavily concentrates probability on the largest activations.
    Least informative about geometric structure — included for comparison."""
    v = activations.flatten().float()
    return torch.softmax(v, dim=0).clamp(min=1e-12)


# ============================================================================
# ENTROPY FORMULA
# One generalized function. Shannon entropy is the special case alpha=1.
# ============================================================================

def renyi_entropy(probs: torch.Tensor, alpha: float) -> float:
    """Rényi entropy at parameter alpha, applied to a probability distribution.

    H_alpha = (1 / (1 - alpha)) * log2(Σ p_i^alpha)

    Special case: alpha → 1.0 recovers Shannon entropy H = -Σ p_i log2(p_i)

    Args:
        probs:  valid probability distribution (non-negative, sums to 1)
        alpha:  order parameter
                alpha < 1  — sensitive to rare/small probability mass
                alpha = 1  — Shannon entropy (use this as primary measure)
                alpha > 1  — dominated by large probability mass

    Returns:
        entropy in bits
    """
    if abs(alpha - 1.0) < 1e-6:
        return -(probs * probs.log2()).sum().item()
    return (1.0 / (1.0 - alpha)) * (probs.pow(alpha).sum().log2().item())


# ============================================================================
# CONVENIENCE WRAPPERS
# Named combinations of normalization + entropy for common use cases.
# These are imported by unified_analysis.py.
# ============================================================================

def effective_rank(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Rényi entropy of energy-weighted distribution (v²/Σv²).
    At alpha=1.0 this is the classical effective rank measure.
    Primary measure throughout this codebase."""
    return renyi_entropy(normalize_energy(activations), alpha)


def effective_rank_abs(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Rényi entropy of absolute-value-weighted distribution (|v|/Σ|v|)."""
    return renyi_entropy(normalize_abs(activations), alpha)


def softmax_entropy(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Rényi entropy of softmax-weighted distribution.
    Included for comparison — see normalize_softmax docstring."""
    return renyi_entropy(normalize_softmax(activations), alpha)


# ============================================================================
# NORMALIZATION METHODS TABLE
# ============================================================================

# Normalization methods available for analysis.
# Each entry: (key_prefix, display_label, normalization_function)
NORM_METHODS = [
    ("energy",  "Energy v²/Σv²",    normalize_energy),
    ("abs",     "Absolute |v|/Σ|v|", normalize_abs),
    ("softmax", "Softmax",           normalize_softmax),
]


# ============================================================================
# CORE COMPUTATION: 2D ENTROPY SURFACE
# For a single prompt, computes entropy at every (layer, token_position).
# Returns a dict of 2D numpy arrays, shape [n_layers, seq_len].
# ============================================================================

def compute_entropy_surface(cache, n_layers: int, hook_pattern: str,
                             seq_len: int, alphas: list) -> dict:
    """
    Compute entropy at every (layer, token_position) for a single prompt.

    Args:
        cache:        TransformerLens activation cache from run_with_cache()
        n_layers:     number of transformer layers
        hook_pattern: hook name template e.g. "blocks.{layer}.hook_resid_pre"
        seq_len:      number of token positions (including BOS)
        alphas:       list of Rényi alpha values

    Returns:
        dict mapping key strings to 2D numpy arrays of shape [n_layers, seq_len].
        Key format: "{norm_key}_alpha_{alpha}"  e.g. "energy_alpha_1.0"

    FORTRAN analogy:
        This is equivalent to a doubly-nested loop over layer index and token
        position index, writing results into a 2D array ENTROPY(layer, pos).
        In FORTRAN you would declare:
            REAL entropy(n_layers, seq_len)
        Here we build that as a numpy array indexed [layer, token_position].
    """
    # Initialize 2D accumulator arrays: one per (norm_method, alpha) combination.
    # Shape: [n_layers, seq_len] — analogous to FORTRAN REAL arr(n_layers, seq_len)
    surfaces = {
        f"{norm_key}_alpha_{alpha}": np.zeros((n_layers, seq_len), dtype=np.float32)
        for norm_key, _, _ in NORM_METHODS
        for alpha in alphas
    }

    for layer in range(n_layers):
        hook_name = hook_pattern.format(layer=layer)
        activations = cache[hook_name]          # shape: [batch, seq_len, d_model]

        for t in range(seq_len):
            vec = activations[0, t, :].float().cpu()   # shape: [d_model]

            for norm_key, _, norm_fn in NORM_METHODS:
                probs = norm_fn(vec)
                for alpha in alphas:
                    key = f"{norm_key}_alpha_{alpha}"
                    surfaces[key][layer, t] = renyi_entropy(probs, alpha)

    return surfaces


# ============================================================================
# ANALYSIS OVER CORPUS
# ============================================================================

def run_entropy_analysis(model, corpus: list, n_layers: int,
                         hook_pattern: str, device: str,
                         alphas: list,
                         category_filter: Optional[str] = None,
                         return_vectors: bool = False):
    """
    For each prompt in corpus, compute the 2D entropy surface [n_layers, seq_len]
    at every token position and layer.

    Each result entry contains:
        - "str_tokens":  list of string token labels e.g. ['<|endoftext|>', 'The', ...]
        - "seq_len":     number of token positions
        - "{norm_key}_alpha_{alpha}": 2D numpy array [n_layers, seq_len]

    Backward-compatible slice for old 1D behavior:
        result["energy_alpha_1.0"][:, -1]   → entropy vs layer, final token only

    Args:
        return_vectors: if True, also return raw layer vectors as a second value.
                        Dict { prompt_str: tensor (n_layers, d_model) } using
                        final token position only (for dynamics compatibility).

    Returns:
        results list, or (results list, vectors dict) if return_vectors=True
    """
    filtered = corpus
    if category_filter:
        filtered = [e for e in corpus if e["category"] == category_filter]
        print(f"  Filtered to category '{category_filter}': {len(filtered)} prompts")

    results = []
    vectors = {}  # only populated if return_vectors=True

    for i, entry in enumerate(filtered):
        prompt = entry["prompt"]
        tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
        seq_len = tokens.shape[1]                          # number of token positions
        str_tokens = model.to_str_tokens(prompt)           # human-readable token labels

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Compute full 2D entropy surface for this prompt
        surfaces = compute_entropy_surface(
            cache, n_layers, hook_pattern, seq_len, alphas
        )

        # Collect final-position vectors for dynamics compatibility
        layer_vecs = []
        if return_vectors:
            for layer in range(n_layers):
                hook_name = hook_pattern.format(layer=layer)
                vec = cache[hook_name][0, -1, :].float().cpu()
                layer_vecs.append(vec)

        results.append({
            "pair_id":     entry["pair_id"],
            "role":        entry["role"],
            "category":    entry["category"],
            "description": entry["description"],
            "prompt":      prompt,
            "str_tokens":  str_tokens,
            "seq_len":     seq_len,
            **surfaces,
        })

        if return_vectors:
            vectors[prompt] = torch.stack(layer_vecs)

        if (i + 1) % 10 == 0 or (i + 1) == len(filtered):
            print(f"  Processed {i+1}/{len(filtered)} prompts...")

    if return_vectors:
        return results, vectors
    return results


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results: list, alphas: list):
    """Print mean trajectory for energy normalization at each alpha.
    Uses final token position only ([:, -1]) for backward compatibility."""
    print("\n" + "=" * 60)
    print("ENTROPY TRAJECTORY SUMMARY (energy normalization v²/Σv²)")
    print("  (final token position)")
    print("=" * 60)

    for alpha in alphas:
        key = f"energy_alpha_{alpha}"
        alpha_label = "Shannon" if abs(alpha - 1.0) < 1e-6 else f"Rényi α={alpha}"
        print(f"\n  {alpha_label}:")

        by_role = defaultdict(list)
        for r in results:
            by_role[r["role"]].append(r[key][:, -1])   # slice to final token → 1D

        for role in ["base", "contrast"]:
            curves = by_role[role]
            if not curves:
                continue
            mean = np.mean(curves, axis=0)
            min_layer = int(np.argmin(mean))
            n_layers = len(mean)
            print(f"    {role.upper()} ({len(curves)} prompts): "
                  f"L0={mean[0]:.2f}  "
                  f"L{n_layers-1}={mean[-1]:.2f}  "
                  f"min=L{min_layer}({mean[min_layer]:.2f})")
            print(f"      {' '.join(f'{v:.1f}' for v in mean)}")
    print()


# ============================================================================
# RESULTS SERIALIZATION
# numpy arrays are not JSON-serializable; convert to nested lists for saving.
# ============================================================================

def results_to_json(results: list) -> list:
    """Convert 2D numpy arrays in results to nested lists for JSON serialization."""
    serializable = []
    for r in results:
        entry = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                entry[k] = v.tolist()   # 2D array → list of lists
            else:
                entry[k] = v
        serializable.append(entry)
    return serializable


def results_from_json(raw: list) -> list:
    """Convert nested lists back to 2D numpy arrays after JSON load."""
    results = []
    for r in raw:
        entry = {}
        for k, v in r.items():
            # Identify entropy surface keys by shape: list of lists of floats
            if (isinstance(v, list) and len(v) > 0
                    and isinstance(v[0], list)):
                entry[k] = np.array(v, dtype=np.float32)
            else:
                entry[k] = v
        results.append(entry)
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise residual stream entropy (geometric diffuseness)"
    )
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to corpus JSON from corpus_gen.py "
                             "(if omitted, runs on default prompts)")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0],
                        help="Rényi alpha values (default: 0.5 1.0 2.0 3.0). "
                             "alpha=1.0 is Shannon entropy.")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory for plots and results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    alphas = sorted(set(args.alpha))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"\nLoading model '{args.model}'...")
    from setup import load_model_and_sae, MODEL_CONFIGS
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")

    model_cfg = MODEL_CONFIGS[args.model]
    hook_pattern = model_cfg.get("hook_pattern", "blocks.{layer}.hook_resid_pre")
    n_layers = model.cfg.n_layers
    print(f"  Layers: {n_layers}  Hook: {hook_pattern}  α={alphas}")

    # ── Corpus mode ───────────────────────────────────────────────────────────
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"✗ Corpus not found: {corpus_path}")
            print("  Run: python corpus_gen.py")
            return 1

        with open(corpus_path) as f:
            corpus = json.load(f)
        print(f"\n✓ Loaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")

        print(f"\nComputing residual stream entropy surfaces across {n_layers} layers...")
        results = run_entropy_analysis(
            model, corpus, n_layers, hook_pattern,
            cfg["device"], alphas, args.category
        )

        results_path = output_dir / f"residual_entropy_results_{args.model}.json"
        with open(results_path, "w") as f:
            json.dump(results_to_json(results), f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        print_summary(results, alphas)

        if not args.no_plots:
            from entropy_plots import plot_overall_mean, plot_category
            print(f"Generating plots in {output_dir}/...")
            categories = sorted(set(r["category"] for r in results))
            for cat in categories:
                plot_category(results, cat, alphas, output_dir, args.model)
            plot_overall_mean(results, alphas, output_dir, args.model)

    # ── Default prompts mode (quick standalone test) ──────────────────────────
    else:
        print(f"\nNo corpus provided — running on default prompts...")

        all_results = []
        for prompt in DEFAULT_PROMPTS:
            tokens = model.to_tokens(prompt).to(cfg["device"])
            seq_len = tokens.shape[1]
            str_tokens = model.to_str_tokens(prompt)

            print(f"\nPrompt: '{prompt}'")
            print(f"  Tokens ({seq_len}): {str_tokens}")

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            surfaces = compute_entropy_surface(
                cache, n_layers, hook_pattern, seq_len, alphas
            )

            # Print final-token Shannon entropy trajectory for quick inspection
            shannon_final = surfaces["energy_alpha_1.0"][:, -1]
            print(f"  Shannon entropy at final token ('{str_tokens[-1]}') vs layer:")
            print(f"  {[f'{v:.3f}' for v in shannon_final]}")

            all_results.append({
                "prompt":     prompt,
                "str_tokens": str_tokens,
                "seq_len":    seq_len,
                **surfaces,
            })

        if not args.no_plots:
            from entropy_plots import (plot_fixed_position, plot_fixed_layer,
                                       plot_2d_surface)
            print(f"\nGenerating plots in {output_dir}/...")
            for result in all_results:
                prompt = result["prompt"]
                safe_name = prompt.replace(" ", "_")[:30]

                # 1D: entropy vs layer, every token position overlaid
                plot_fixed_position(
                    result, alphas, output_dir,
                    filename=f"entropy_vs_layer_{safe_name}.png"
                )

                # 1D: entropy vs token position, every layer overlaid
                plot_fixed_layer(
                    result, alphas, output_dir,
                    filename=f"entropy_vs_position_{safe_name}.png"
                )

                # 2D: full entropy surface heatmap
                plot_2d_surface(
                    result, alpha=1.0, output_dir=output_dir,
                    filename=f"entropy_surface_{safe_name}.png"
                )

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
