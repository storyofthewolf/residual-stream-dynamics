"""Entropy of raw residual stream activations (geometric diffuseness)."""

import torch
import numpy as np
from sklearn.decomposition import PCA
import argparse
from pathlib import Path
from transformer_lens import HookedTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Instead of softmax entropy, use the fraction of variance explained
def effective_rank(activations):
    v = activations.flatten().float()
    v2 = v ** 2
    probs = v2 / v2.sum()          # weight by squared magnitude, not softmax
    probs = probs[probs > 1e-10]
    return -torch.sum(probs * torch.log2(probs)).item()

def effective_rank_abs(activations):
    """Entropy using |v|/sum(|v|) — linear weighting by magnitude."""
    v = activations.flatten().float().abs()
    probs = v / v.sum()
    probs = probs[probs > 1e-10]
    return -torch.sum(probs * torch.log2(probs)).item()


def entropy(activations):
    """Compute entropy of activation distribution"""
    # Normalize to probability distribution
    probs = torch.softmax(activations.flatten(), dim=0)
    # Remove zeros (log(0) = undefined)
    probs = probs[probs > 1e-10]
    # Entropy = -Σ p log p
    return -torch.sum(probs * torch.log2(probs)).item()

prompts = [
    "wolf",                                    # Clear, single noun
    "The wolf ran",                            # Clear, short sentence
    "The wolf ran through the forest",         # Clear, longer
    "wolf wolf wolf",                          # Repetitive
    "asdfgh",                                  # Noise
    "wol",                                     # Typo/degraded
]



COLORS = plt.cm.tab10.colors

def plot_entropy(results: dict, output_path: str = "entropy_activation.png"):
    """
    results: { prompt_str: [entropy_layer0, ..., entropy_layer11] }
    One line per prompt, X=layer, Y=activation entropy (bits).
    """
    layers = list(range(12))
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
    ax.set_yscale("linear")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Plot saved to {output_path}")


def plot_entropy_comparison(results_sq: dict, results_abs: dict,
                            output_path: str = "entropy_comparison.png"):
    """
    Side-by-side: effective rank (v²) vs absolute value (|v|) normalization.
    One subplot per measure, same prompts, same colors.
    """
    layers = list(range(12))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    titles = [
        "Effective rank  v²/Σv²",
        "Absolute value  |v|/Σ|v|",
    ]
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



def plot_pca_trajectory(vectors: dict, output_path: str = "pca_trajectory.png"):
    """
    vectors: { prompt_str: tensor of shape (n_layers, d_model) }
    Fits PCA on all prompts x layers together, plots each prompt
    as a trajectory through 2D PCA space, colored by layer index.
    """
    n_layers = 12
    prompt_list = list(vectors.keys())
    all_vecs = [vectors[p].numpy() for p in prompt_list]
    stacked = np.vstack(all_vecs)  # (n_prompts * n_layers, d_model)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(stacked)  # (n_prompts * n_layers, 2)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, prompt in enumerate(prompt_list):
        color = COLORS[i % len(COLORS)]
        traj = projected[i * n_layers : (i + 1) * n_layers]

        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.7)
        sc = ax.scatter(traj[:, 0], traj[:, 1],
                        c=list(range(n_layers)), cmap="plasma",
                        s=40, zorder=3, edgecolors=color, linewidths=0.8)
        ax.annotate("L0",  traj[0],  fontsize=6, color=color,
                    xytext=(3, 3), textcoords="offset points")
        ax.annotate("L11", traj[-1], fontsize=6, color=color,
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


def plot_cosine_similarity(vectors: dict, output_path: str = "cosine_similarity.png"):
    """
    vectors: { prompt_str: tensor of shape (n_layers, d_model) }
    Plots cos(h_L, h_{L+1}) for each consecutive layer pair.
    Dips indicate layers doing substantial directional work.
    """
    n_layers = 12
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

def main():
    parser = argparse.ArgumentParser(description="Layer-wise entropy analysis via logit lens")
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()


    print(f"\nLoading model '{args.model}'...")
    from setup import load_model_and_sae
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")



    results = {}
    results_abs = {}
    vectors = {}
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        logits, cache = model.run_with_cache(tokens)

        entropy_by_layer = []
        entropy_by_layer_abs = []
        layer_vecs = []
        for layer in range(12):
            activations = cache[f"blocks.{layer}.hook_resid_pre"]
            vec = activations[0, -1, :].float().cpu()
            ent = effective_rank(vec)
            ent_abs = effective_rank_abs(vec)
            entropy_by_layer.append(ent)
            entropy_by_layer_abs.append(ent_abs)
            layer_vecs.append(vec)

        results[prompt] = entropy_by_layer
        results_abs[prompt] = entropy_by_layer_abs
        vectors[prompt] = torch.stack(layer_vecs)
        print(f"'{prompt}': {[f'{v:.4f}' for v in entropy_by_layer]}")

    plot_entropy(results)
    plot_entropy_comparison(results, results_abs)
    plot_pca_trajectory(vectors)
    plot_cosine_similarity(vectors)


if __name__ == "__main__":
    main()

"""

This would show you the entropy profile. **My prediction:** You'll see a U-shape (high early, low middle, higher late).

---

## Why This Matters Foundationally

If entropy *does* follow information bottleneck theory:

1. **Predictive:** You can predict what each layer should compute
2. **General:** Should apply across models, not just GPT-2
3. **Explanatory:** Explains *why* layers organize information this way
4. **Connects to physics:** Information theory bridges neuroscience, ML, and physics

---

## Your Skepticism About SAEs Resolves Here

SAEs are useful for describing features. But **entropy analysis explains the structure**.
```
SAE approach: "Feature 4484 appears in Layers 3-6"
Entropy approach: "Layers 3-6 are in the compression phase, 
                   so features should cluster and become interpretable"
"""
