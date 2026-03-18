"""Entropy of raw residual stream activations (geometric diffuseness)."""

import json
import argparse
import warnings
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import numpy as np
from sklearn.decomposition import PCA
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
# ENTROPY FUNCTIONS
# ============================================================================

def effective_rank(activations):
    """Entropy using v²/Σv² — energy weighting by squared magnitude."""
    v = activations.flatten().float()
    v2 = v ** 2
    probs = v2 / v2.sum()
    probs = probs[probs > 1e-10]
    return -torch.sum(probs * torch.log2(probs)).item()


def effective_rank_abs(activations):
    """Entropy using |v|/Σ|v| — linear weighting by magnitude."""
    v = activations.flatten().float().abs()
    probs = v / v.sum()
    probs = probs[probs > 1e-10]
    return -torch.sum(probs * torch.log2(probs)).item()


# ============================================================================
# ANALYSIS OVER CORPUS
# ============================================================================

def run_entropy_analysis(model, corpus: list, n_layers: int,
                         hook_pattern: str, device: str,
                         category_filter: Optional[str] = None) -> list:
    """
    For each prompt in corpus, compute per-layer effective_rank entropy
    of the residual stream activation vector at the final token position.

    Returns list of result dicts, one per prompt, mirroring the structure
    of entropy_logit_projection.py for consistency.
    """
    filtered = corpus
    if category_filter:
        filtered = [e for e in corpus if e["category"] == category_filter]
        print(f"  Filtered to category '{category_filter}': {len(filtered)} prompts")

    results = []
    for i, entry in enumerate(filtered):
        prompt = entry["prompt"]
        tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        entropy_by_layer = []
        entropy_abs_by_layer = []
        layer_vecs = []

        for layer in range(n_layers):
            hook_name = hook_pattern.format(layer=layer)
            activations = cache[hook_name]
            vec = activations[0, -1, :].float().cpu()
            entropy_by_layer.append(effective_rank(vec))
            entropy_abs_by_layer.append(effective_rank_abs(vec))
            layer_vecs.append(vec)

        results.append({
            "pair_id": entry["pair_id"],
            "role": entry["role"],
            "category": entry["category"],
            "description": entry["description"],
            "prompt": prompt,
            "effective_rank": entropy_by_layer,
            "effective_rank_abs": entropy_abs_by_layer,
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(filtered):
            print(f"  Processed {i+1}/{len(filtered)} prompts...")

    return results


# ============================================================================
# PLOTTING
# ============================================================================

def plot_entropy(results: dict, output_path: str = "entropy_activation.png"):
    """
    results: { prompt_str: [entropy_layer0, ..., entropy_layerN] }
    One line per prompt, X=layer, Y=activation entropy (bits).
    """
    n_layers = len(next(iter(results.values())))
    layers = list(range(n_layers))
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (prompt, curve) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(layers, curve, marker="o", markersize=4,
                linewidth=1.8, color=color, label=repr(prompt))

    ax.set_xlabel("Layer")
    ax.set_ylabel("Activation entropy (bits)")
    ax.set_title("Residual stream activation entropy by layer")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✓ Plot saved to {output_path}")


def plot_entropy_comparison(results_sq: dict, results_abs: dict,
                            output_path: str = "entropy_comparison.png"):
    """
    Side-by-side: effective rank (v²) vs absolute value (|v|) normalization.
    """
    n_layers = len(next(iter(results_sq.values())))
    layers = list(range(n_layers))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    titles = ["Effective rank  v²/Σv²", "Absolute value  |v|/Σ|v|"]
    datasets = [results_sq, results_abs]

    for ax, data, title in zip(axes, datasets, titles):
        for i, (prompt, curve) in enumerate(data.items()):
            color = COLORS[i % len(COLORS)]
            ax.plot(layers, curve, marker="o", markersize=4,
                    linewidth=1.8, color=color, label=repr(prompt))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title(title)
        ax.set_xticks(layers)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.suptitle("Residual stream entropy: v² vs |v| normalization", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✓ Comparison plot saved to {output_path}")


def plot_category(results: list, category: str, output_dir: Path):
    """
    One figure per category. X=layer, Y=mean effective_rank entropy.
    Base = solid, contrast = dashed. Bold lines = mean across pairs.
    """
    cat_results = [r for r in results if r["category"] == category]
    if not cat_results:
        return

    by_pair = defaultdict(dict)
    for r in cat_results:
        by_pair[r["pair_id"]][r["role"]] = r

    n_layers = len(next(iter(cat_results))["effective_rank"])
    layers = list(range(n_layers))
    PAIR_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors[:5])

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(f"Residual stream entropy — {category}", fontsize=12)

    all_base, all_contrast = [], []

    for pair_idx, (pair_id, roles) in enumerate(sorted(by_pair.items())):
        color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
        for role, ls in [("base", "-"), ("contrast", "--")]:
            if role not in roles:
                continue
            curve = roles[role]["effective_rank"]
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
    ax.set_ylabel("Effective rank entropy (bits)")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    out_path = output_dir / f"residual_entropy_{category}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_overall_mean(results: list, output_dir: Path):
    """Mean ± std effective_rank entropy across all categories."""
    n_layers = len(results[0]["effective_rank"])
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Residual stream entropy — all categories (mean ± 1σ)", fontsize=12)

    for role, color, ls in [("base", "#1565C0", "-"),
                              ("contrast", "#B71C1C", "--")]:
        curves = [r["effective_rank"] for r in results if r["role"] == role]
        if not curves:
            continue
        arr = np.array(curves)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(layers, mean, color=color, linestyle=ls,
                linewidth=2.5, label=f"{role} mean")
        ax.fill_between(layers, mean - std, mean + std,
                        color=color, alpha=0.15, label=f"{role} ±1σ")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Effective rank entropy (bits)")
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "residual_entropy_overall.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_pca_trajectory(vectors: dict, n_layers: int,
                        output_path: str = "pca_trajectory.png"):
    """
    vectors: { prompt_str: tensor of shape (n_layers, d_model) }
    Fits PCA on all prompts x layers, plots each prompt as a trajectory
    through 2D PCA space colored by layer index.
    """
    prompt_list = list(vectors.keys())
    all_vecs = [vectors[p].numpy() for p in prompt_list]
    stacked = np.vstack(all_vecs)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(stacked)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, prompt in enumerate(prompt_list):
        color = COLORS[i % len(COLORS)]
        traj = projected[i * n_layers: (i + 1) * n_layers]
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.7)
        sc = ax.scatter(traj[:, 0], traj[:, 1],
                        c=list(range(n_layers)), cmap="plasma",
                        s=40, zorder=3, edgecolors=color, linewidths=0.8)
        ax.annotate("L0", traj[0], fontsize=6, color=color,
                    xytext=(3, 3), textcoords="offset points")
        ax.annotate(f"L{n_layers-1}", traj[-1], fontsize=6, color=color,
                    xytext=(3, 3), textcoords="offset points")
        ax.plot([], [], color=color, linewidth=2, label=repr(prompt))

    plt.colorbar(sc, ax=ax, label="Layer index")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("PCA trajectory of residual stream across layers")
    ax.legend(fontsize=8, bbox_to_anchor=(1.15, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✓ PCA trajectory saved to {output_path}")


def plot_cosine_similarity(vectors: dict, n_layers: int,
                           output_path: str = "cosine_similarity.png"):
    """
    Plots cos(h_L, h_{L+1}) for each consecutive layer pair.
    Dips indicate layers doing substantial directional work.
    """
    transitions = list(range(n_layers - 1))
    transition_labels = [f"{l}→{l+1}" for l in transitions]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (prompt, vecs) in enumerate(vectors.items()):
        color = COLORS[i % len(COLORS)]
        sims = []
        for layer in range(n_layers - 1):
            h0 = vecs[layer].float()
            h1 = vecs[layer + 1].float()
            cos_sim = torch.nn.functional.cosine_similarity(
                h0.unsqueeze(0), h1.unsqueeze(0)
            ).item()
            sims.append(cos_sim)
        ax.plot(transitions, sims, marker="o", markersize=4,
                linewidth=1.8, color=color, label=repr(prompt))

    ax.set_xlabel("Layer transition")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Cosine similarity of residual stream between consecutive layers")
    ax.set_xticks(transitions)
    ax.set_xticklabels(transition_labels, rotation=45, ha="right")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✓ Cosine similarity saved to {output_path}")


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results: list):
    by_role = defaultdict(list)
    for r in results:
        by_role[r["role"]].append(r["effective_rank"])

    print("\n" + "=" * 60)
    print("ENTROPY TRAJECTORY SUMMARY (effective_rank, final token position)")
    print("=" * 60)
    for role in ["base", "contrast"]:
        curves = by_role[role]
        if not curves:
            continue
        mean = np.mean(curves, axis=0)
        min_layer = int(np.argmin(mean))
        print(f"\n  {role.upper()} ({len(curves)} prompts):")
        print(f"    Layer  0: {mean[0]:.3f} bits")
        print(f"    Layer {len(mean)-1:2d}: {mean[-1]:.3f} bits")
        print(f"    Minimum:  layer {min_layer} → {mean[min_layer]:.3f} bits")
        print(f"    Trajectory: {' '.join(f'{v:.1f}' for v in mean)}")
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
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory for plots and results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"\nLoading model '{args.model}'...")
    from setup import load_model_and_sae, MODEL_CONFIGS
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")

    model_cfg = MODEL_CONFIGS[args.model]
    hook_pattern = model_cfg.get("hook_pattern", "blocks.{layer}.hook_resid_pre")

    # Infer number of layers from model
    n_layers = model.cfg.n_layers
    print(f"  Layers: {n_layers}  Hook: {hook_pattern}")

    # ── Corpus mode ──────────────────────────────────────────────────────────
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
            cfg["device"], args.category
        )

        results_path = output_dir / "residual_entropy_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        print_summary(results)

        if not args.no_plots:
            print(f"Generating plots in {output_dir}/...")
            categories = sorted(set(r["category"] for r in results))
            for cat in categories:
                plot_category(results, cat, output_dir)
            plot_overall_mean(results, output_dir)

    # ── Default prompts mode (quick standalone test) ──────────────────────────
    else:
        print(f"\nNo corpus provided — running on default prompts...")
        results_sq = {}
        results_abs = {}
        vectors = {}

        for prompt in DEFAULT_PROMPTS:
            tokens = model.to_tokens(prompt).to(cfg["device"])
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)

            entropy_by_layer = []
            entropy_by_layer_abs = []
            layer_vecs = []

            for layer in range(n_layers):
                hook_name = hook_pattern.format(layer=layer)
                activations = cache[hook_name]
                vec = activations[0, -1, :].float().cpu()
                entropy_by_layer.append(effective_rank(vec))
                entropy_by_layer_abs.append(effective_rank_abs(vec))
                layer_vecs.append(vec)

            results_sq[prompt] = entropy_by_layer
            results_abs[prompt] = entropy_by_layer_abs
            vectors[prompt] = torch.stack(layer_vecs)
            print(f"  '{prompt}': {[f'{v:.3f}' for v in entropy_by_layer]}")

        if not args.no_plots:
            print(f"\nGenerating plots in {output_dir}/...")
            plot_entropy(results_sq,
                         str(output_dir / "residual_entropy_default.png"))
            plot_entropy_comparison(results_sq, results_abs,
                                    str(output_dir / "residual_entropy_comparison.png"))
            plot_pca_trajectory(vectors, n_layers,
                                str(output_dir / "pca_trajectory.png"))
            plot_cosine_similarity(vectors, n_layers,
                                   str(output_dir / "cosine_similarity.png"))

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
