"""
entropy_logit_projections.py — Layer-wise entropy via logit lens over a prompt corpus.

Entropy of logit-lens projection through W_U (token-space uncertainty).

For each prompt, projects the residual stream at each layer through the
unembedding matrix (logit lens) to obtain a vocabulary distribution, then
computes Shannon and Rényi entropy at each layer × token position.

This reveals how entropy evolves through the network — whether it decreases
monotonically, forms a U-shape, or behaves differently for structured
(base) vs. incoherent (contrast) prompts.

Key idea:
    At layer L, residual stream h_L is projected as:
        logits_L = LayerNorm(h_L) @ W_U
        p_L = softmax(logits_L)
        H_L = entropy(p_L)

    Plotting H_L across L=0..11 gives the entropy trajectory.

Usage:
    python entropy_analysis.py --corpus corpus.json
    python entropy_analysis.py --corpus corpus.json --model gpt2-small
    python entropy_analysis.py --corpus corpus.json --alpha 0.5 2.0 3.0
    python entropy_analysis.py --corpus corpus.json --category pattern
    python entropy_analysis.py --corpus corpus.json --no-plots
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

N_LAYERS = 12  # GPT-2 small

# ============================================================================
# ENTROPY FUNCTIONS
# ============================================================================

def shannon_entropy(probs: torch.Tensor) -> float:
    probs = probs.clamp(min=1e-12)
    return -(probs * probs.log2()).sum().item()


def renyi_entropy(probs: torch.Tensor, alpha: float) -> float:
    if abs(alpha - 1.0) < 1e-6:
        return shannon_entropy(probs)
    probs = probs.clamp(min=1e-12)
    return (1.0 / (1.0 - alpha)) * (probs.pow(alpha).sum().log2().item())


# ============================================================================
# LOGIT LENS
# ============================================================================

def logit_lens_entropy(model, prompt: str, alphas: list, device: str) -> dict:
    """
    Run forward pass, then at each layer project residual stream through
    the unembedding matrix to get a vocab distribution and compute entropy
    at every token position.

    Returns:
        {
            "tokens": [str, ...],
            "shannon":    ndarray (n_layers, seq_len),
            "renyi_{a}":  ndarray (n_layers, seq_len),  for each alpha
        }
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    tokens = tokens.to(device)
    seq_len = tokens.shape[1]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    W_U = model.unembed.W_U  # (d_model, vocab)

    token_ids = tokens[0].tolist()
    token_strs = [model.tokenizer.decode([t]) for t in token_ids]

    shannon_matrix = np.zeros((N_LAYERS, seq_len))
    renyi_matrices = {a: np.zeros((N_LAYERS, seq_len)) for a in alphas}

    for layer in range(N_LAYERS):
        resid = cache[f"blocks.{layer}.hook_resid_post"][0]  # (seq_len, d_model)
        with torch.no_grad():
            normed = model.ln_final(resid)      # (seq_len, d_model)
            logits = normed @ W_U               # (seq_len, vocab)
        probs = torch.softmax(logits, dim=-1)

        for pos in range(seq_len):
            p = probs[pos]
            shannon_matrix[layer, pos] = shannon_entropy(p)
            for a in alphas:
                renyi_matrices[a][layer, pos] = renyi_entropy(p, a)

    result = {"tokens": token_strs, "shannon": shannon_matrix}
    for a in alphas:
        result[f"renyi_{a}"] = renyi_matrices[a]
    return result


# ============================================================================
# ANALYSIS OVER CORPUS
# ============================================================================

def run_entropy_analysis(model, corpus: list, alphas: list,
                         device: str,
                         category_filter: Optional[str] = None) -> list:
    filtered = corpus
    if category_filter:
        filtered = [e for e in corpus if e["category"] == category_filter]
        print(f"  Filtered to category '{category_filter}': {len(filtered)} prompts")

    results = []
    for i, entry in enumerate(filtered):
        profile = logit_lens_entropy(model, entry["prompt"], alphas, device)
        results.append({
            "pair_id": entry["pair_id"],
            "role": entry["role"],
            "category": entry["category"],
            "description": entry["description"],
            "prompt": entry["prompt"],
            "tokens": profile["tokens"],
            "shannon": profile["shannon"].tolist(),
            **{f"renyi_{a}": profile[f"renyi_{a}"].tolist() for a in alphas},
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(filtered):
            print(f"  Processed {i+1}/{len(filtered)} prompts...")

    return results


# ============================================================================
# PLOTTING
# ============================================================================

PAIR_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors[:5])


def plot_category(results: list, category: str, alphas: list, output_dir: Path):
    """
    One figure per category. Each subplot is one entropy metric (Shannon +
    each Rényi α). X=layer, Y=entropy averaged across token positions.
    Base = solid, contrast = dashed. Pairs share color.
    Bold black/gray lines = mean across all base/contrast.
    """
    cat_results = [r for r in results if r["category"] == category]
    if not cat_results:
        return

    by_pair = defaultdict(dict)
    for r in cat_results:
        by_pair[r["pair_id"]][r["role"]] = r

    metrics = [("shannon", "Shannon Entropy (bits)")] + \
              [(f"renyi_{a}", f"Rényi α={a} (bits)") for a in alphas]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(f"Layer-wise entropy — {category}", fontsize=12)
    layers = list(range(N_LAYERS))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        all_base, all_contrast = [], []

        for pair_idx, (pair_id, roles) in enumerate(sorted(by_pair.items())):
            color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
            for role, ls in [("base", "-"), ("contrast", "--")]:
                if role not in roles:
                    continue
                curve = np.array(roles[role][metric_key]).mean(axis=1)  # (n_layers,)
                ax.plot(layers, curve, color=color, linestyle=ls,
                        linewidth=1.2, alpha=0.55,
                        label=f"P{pair_id} {role}" if pair_idx < 6 else None)
                (all_base if role == "base" else all_contrast).append(curve)

        if all_base:
            ax.plot(layers, np.mean(all_base, axis=0),
                    color="black", linestyle="-", linewidth=2.5,
                    label="mean base", zorder=5)
        if all_contrast:
            ax.plot(layers, np.mean(all_contrast, axis=0),
                    color="dimgray", linestyle="--", linewidth=2.5,
                    label="mean contrast", zorder=5)

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_label)
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    out_path = output_dir / f"entropy_layers_{category}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_overall_mean(results: list, alphas: list, output_dir: Path):
    """Mean ± std entropy trajectory across all categories."""
    metrics = [("shannon", "Shannon Entropy (bits)")] + \
              [(f"renyi_{a}", f"Rényi α={a} (bits)") for a in alphas]

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle("Layer-wise entropy — all categories (mean ± 1σ)", fontsize=12)
    layers = list(range(N_LAYERS))

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for role, color, ls in [("base", "#1565C0", "-"),
                                  ("contrast", "#B71C1C", "--")]:
            curves = [np.array(r[metric_key]).mean(axis=1)
                      for r in results if r["role"] == role]
            if not curves:
                continue
            arr = np.array(curves)
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(layers, mean, color=color, linestyle=ls,
                    linewidth=2.5, label=f"{role} mean")
            ax.fill_between(layers, mean - std, mean + std,
                            color=color, alpha=0.15, label=f"{role} ±1σ")

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_label)
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "entropy_layers_overall.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_per_token_heatmap(results: list, output_dir: Path, metric: str = "shannon"):
    """
    Heatmap of entropy at (layer × token position) for one base/contrast
    sample per category. Shows whether entropy changes are uniform across
    positions or concentrated at specific tokens.
    """
    by_cat_pair = defaultdict(dict)
    for r in results:
        key = (r["category"], r["pair_id"])
        by_cat_pair[key][r["role"]] = r

    samples = []
    seen_cats = set()
    for (cat, pid), roles in sorted(by_cat_pair.items()):
        if cat in seen_cats:
            continue
        if "base" in roles and "contrast" in roles:
            samples += [roles["base"], roles["contrast"]]
            seen_cats.add(cat)

    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Entropy heatmap (layer × token) — {metric}", fontsize=10)

    for ax, r in zip(axes, samples):
        matrix = np.array(r[metric])
        im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                       interpolation="nearest", origin="upper")
        ax.set_title(f"[{r['role']}]\n{r['prompt'][:28]}", fontsize=7)
        ax.set_xlabel("Token")
        if ax is axes[0]:
            ax.set_ylabel("Layer")
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([str(l) for l in range(N_LAYERS)], fontsize=7)
        toks = r["tokens"]
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=45, ha="right", fontsize=7)
        plt.colorbar(im, ax=ax, label="bits")

    plt.tight_layout()
    out_path = output_dir / f"heatmap_layer_token_{metric}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results: list):
    by_role = defaultdict(list)
    for r in results:
        curve = np.array(r["shannon"]).mean(axis=1)
        by_role[r["role"]].append(curve)

    print("\n" + "=" * 60)
    print("ENTROPY TRAJECTORY SUMMARY (Shannon, avg across token positions)")
    print("=" * 60)
    for role in ["base", "contrast"]:
        curves = by_role[role]
        if not curves:
            continue
        mean = np.mean(curves, axis=0)
        min_layer = int(np.argmin(mean))
        print(f"\n  {role.upper()} ({len(curves)} prompts):")
        print(f"    Layer  0: {mean[0]:.3f} bits")
        print(f"    Layer 11: {mean[-1]:.3f} bits")
        print(f"    Minimum:  layer {min_layer} → {mean[min_layer]:.3f} bits")
        print(f"    Trajectory: {' '.join(f'{v:.1f}' for v in mean)}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer-wise entropy analysis via logit lens"
    )
    parser.add_argument("--corpus", type=str, default="corpus.json")
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5, 2.0, 3.0])
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="entropy_results")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    alphas = sorted(set(args.alpha))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"✗ Corpus not found: {corpus_path}")
        print("  Run: python corpus_gen.py")
        return 1

    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"\n✓ Loaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")

    print(f"\nLoading model '{args.model}'...")
    from setup import load_model_and_sae
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")

    print(f"\nComputing logit-lens entropy across {N_LAYERS} layers (α={alphas})...")
    results = run_entropy_analysis(model, corpus, alphas, cfg["device"], args.category)

    results_path = output_dir / "entropy_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    print_summary(results)

    if not args.no_plots:
        print(f"Generating plots in {output_dir}/...")
        categories = sorted(set(r["category"] for r in results))
        for cat in categories:
            plot_category(results, cat, alphas, output_dir)
        plot_overall_mean(results, alphas, output_dir)
        plot_per_token_heatmap(results, output_dir, "shannon")

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
