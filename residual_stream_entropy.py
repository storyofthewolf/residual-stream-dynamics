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

Trajectory geometry (PCA, cosine similarity, velocity, speed, acceleration)
lives in residual_stream_dynamics.py.

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

# Fallback prompts for quick standalone testing (no corpus required)
DEFAULT_PROMPTS = [
    "wolf",
    "The wolf ran",
    "The wolf ran through the forest",
    "wolf wolf wolf",
    "asdfgh",
    "wol",
]

COLORS = plt.cm.tab10.colors


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
# ANALYSIS OVER CORPUS
# ============================================================================

# Normalization methods available for analysis.
# Each entry: (key_prefix, display_label, normalization_function)
NORM_METHODS = [
    ("energy",  "Energy v²/Σv²",    normalize_energy),
    ("abs",     "Absolute |v|/Σ|v|", normalize_abs),
    ("softmax", "Softmax",           normalize_softmax),
]


def run_entropy_analysis(model, corpus: list, n_layers: int,
                         hook_pattern: str, device: str,
                         alphas: list,
                         category_filter: Optional[str] = None,
                         return_vectors: bool = False):
    """
    For each prompt in corpus, compute per-layer Rényi entropy of the
    residual stream activation vector at the final token position.

    Computes all normalization methods × all alpha values.

    Result dict keys follow the pattern: "{norm_key}_alpha_{alpha}"
    e.g. "energy_alpha_1.0", "abs_alpha_2.0", "softmax_alpha_0.5"

    Args:
        return_vectors: if True, also return raw layer vectors as a second
                        value — a dict { prompt_str: tensor (n_layers, d_model) }
                        Used by residual_stream_dynamics.py to avoid a second
                        forward pass.

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

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        # Build entropy curves: {key: [layer0, layer1, ...]}
        entropy_curves = {
            f"{norm_key}_alpha_{alpha}": []
            for norm_key, _, _ in NORM_METHODS
            for alpha in alphas
        }

        layer_vecs = []

        for layer in range(n_layers):
            hook_name = hook_pattern.format(layer=layer)
            activations = cache[hook_name]
            vec = activations[0, -1, :].float().cpu()

            for norm_key, _, norm_fn in NORM_METHODS:
                probs = norm_fn(vec)
                for alpha in alphas:
                    key = f"{norm_key}_alpha_{alpha}"
                    entropy_curves[key].append(renyi_entropy(probs, alpha))

            if return_vectors:
                layer_vecs.append(vec)

        results.append({
            "pair_id":     entry["pair_id"],
            "role":        entry["role"],
            "category":    entry["category"],
            "description": entry["description"],
            "prompt":      prompt,
            **entropy_curves,
        })

        if return_vectors:
            vectors[prompt] = torch.stack(layer_vecs)

        if (i + 1) % 10 == 0 or (i + 1) == len(filtered):
            print(f"  Processed {i+1}/{len(filtered)} prompts...")

    if return_vectors:
        return results, vectors
    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_overall_mean(results: list, alphas: list, output_dir: Path,
                      model_name: str = ""):
    """
    One row of subplots per normalization method.
    One column per alpha value.
    Each subplot: mean ± 1σ entropy across all pairs, base vs contrast.
    """
    n_rows = len(NORM_METHODS)
    n_cols = len(alphas)
    n_layers = len(results[0][f"energy_alpha_{alphas[0]}"])
    layers = list(range(n_layers))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharey=False, squeeze=False)

    title = "Residual stream entropy — all categories (mean ± 1σ)"
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=12)

    for row, (norm_key, norm_label, _) in enumerate(NORM_METHODS):
        for col, alpha in enumerate(alphas):
            ax = axes[row][col]
            key = f"{norm_key}_alpha_{alpha}"
            alpha_label = "Shannon" if abs(alpha - 1.0) < 1e-6 else f"Rényi α={alpha}"

            for role, color, ls in [("base",     "#1565C0", "-"),
                                     ("contrast", "#B71C1C", "--")]:
                curves = [r[key] for r in results if r["role"] == role]
                if not curves:
                    continue
                arr = np.array(curves)
                mean, std = arr.mean(axis=0), arr.std(axis=0)
                ax.plot(layers, mean, color=color, linestyle=ls,
                        linewidth=2.0, label=f"{role} mean")
                ax.fill_between(layers, mean - std, mean + std,
                                color=color, alpha=0.15)

            ax.set_title(f"{norm_label}\n{alpha_label}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Entropy (bits)")
            ax.set_xticks(layers)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    out_path = output_dir / f"residual_entropy_overall{suffix}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_category(results: list, category: str, alphas: list,
                  output_dir: Path, model_name: str = ""):
    """
    Per-category plot. One subplot per normalization × alpha combination.
    Base=solid, contrast=dashed. Bold lines=mean across pairs.
    """
    cat_results = [r for r in results if r["category"] == category]
    if not cat_results:
        return

    by_pair = defaultdict(dict)
    for r in cat_results:
        by_pair[r["pair_id"]][r["role"]] = r

    n_layers = len(next(iter(cat_results))[f"energy_alpha_{alphas[0]}"])
    layers = list(range(n_layers))
    PAIR_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors[:5])

    n_rows = len(NORM_METHODS)
    n_cols = len(alphas)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             sharey=False, squeeze=False)

    title = f"Residual stream entropy — {category}"
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=12)

    for row, (norm_key, norm_label, _) in enumerate(NORM_METHODS):
        for col, alpha in enumerate(alphas):
            ax = axes[row][col]
            key = f"{norm_key}_alpha_{alpha}"
            alpha_label = "Shannon" if abs(alpha - 1.0) < 1e-6 else f"Rényi α={alpha}"
            all_base, all_contrast = [], []

            for pair_idx, (pair_id, roles) in enumerate(sorted(by_pair.items())):
                color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
                for role, ls in [("base", "-"), ("contrast", "--")]:
                    if role not in roles:
                        continue
                    curve = roles[role][key]
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

            ax.set_title(f"{norm_label}\n{alpha_label}", fontsize=9)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Entropy (bits)")
            ax.set_xticks(layers)
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    out_path = output_dir / f"residual_entropy_{category}{suffix}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results: list, alphas: list):
    """Print mean trajectory for energy normalization at each alpha."""
    print("\n" + "=" * 60)
    print("ENTROPY TRAJECTORY SUMMARY (energy normalization v²/Σv²)")
    print("=" * 60)

    for alpha in alphas:
        key = f"energy_alpha_{alpha}"
        alpha_label = "Shannon" if abs(alpha - 1.0) < 1e-6 else f"Rényi α={alpha}"
        print(f"\n  {alpha_label}:")

        by_role = defaultdict(list)
        for r in results:
            by_role[r["role"]].append(r[key])

        for role in ["base", "contrast"]:
            curves = by_role[role]
            if not curves:
                continue
            mean = np.mean(curves, axis=0)
            min_layer = int(np.argmin(mean))
            print(f"    {role.upper()} ({len(curves)} prompts): "
                  f"L0={mean[0]:.2f}  "
                  f"L{len(mean)-1}={mean[-1]:.2f}  "
                  f"min=L{min_layer}({mean[min_layer]:.2f})")
            print(f"      {' '.join(f'{v:.1f}' for v in mean)}")
    print()


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

        print(f"\nComputing residual stream entropy across {n_layers} layers...")
        results = run_entropy_analysis(
            model, corpus, n_layers, hook_pattern,
            cfg["device"], alphas, args.category
        )

        results_path = output_dir / f"residual_entropy_results_{args.model}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        print_summary(results, alphas)

        if not args.no_plots:
            print(f"Generating plots in {output_dir}/...")
            categories = sorted(set(r["category"] for r in results))
            for cat in categories:
                plot_category(results, cat, alphas, output_dir, args.model)
            plot_overall_mean(results, alphas, output_dir, args.model)

    # ── Default prompts mode (quick standalone test) ──────────────────────────
    else:
        print(f"\nNo corpus provided — running on default prompts...")
        curves_by_prompt = {p: {a: [] for a in alphas} for p in DEFAULT_PROMPTS}

        for prompt in DEFAULT_PROMPTS:
            tokens = model.to_tokens(prompt).to(cfg["device"])
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            for layer in range(n_layers):
                hook_name = hook_pattern.format(layer=layer)
                vec = cache[hook_name][0, -1, :].float().cpu()
                probs = normalize_energy(vec)
                for alpha in alphas:
                    curves_by_prompt[prompt][alpha].append(renyi_entropy(probs, alpha))

            print(f"  '{prompt}': "
                  f"{[f'{v:.3f}' for v in curves_by_prompt[prompt][1.0]]}")

        if not args.no_plots:
            print(f"\nGenerating plots in {output_dir}/...")
            shannon_curves = {p: curves_by_prompt[p][1.0] for p in DEFAULT_PROMPTS}
            layers = list(range(len(next(iter(shannon_curves.values())))))
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, (prompt, curve) in enumerate(shannon_curves.items()):
                ax.plot(layers, curve, marker="o", markersize=4,
                        linewidth=1.8, color=COLORS[i % len(COLORS)],
                        label=repr(prompt))
            ax.set_xlabel("Layer")
            ax.set_ylabel("Entropy (bits)")
            ax.set_title("Residual stream entropy — energy norm, Shannon (α=1)")
            ax.set_xticks(layers)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.tight_layout()
            out = str(output_dir / "residual_entropy_default.png")
            plt.savefig(out, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"✓ {out}")

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
