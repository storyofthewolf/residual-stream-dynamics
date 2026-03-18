"""
sae_analysis.py — SAE feature analysis over a structured prompt corpus.

For each layer and each contrast pair, extracts top-k SAE features and
computes:
  - Features active on base prompt only
  - Features active on contrast prompt only
  - Features shared by both

Then aggregates across pairs within each category to find features that
consistently appear (or consistently differentiate) across the corpus.

Builds on the layer-scanning approach in sae_layers.py but driven by
the contrast pair corpus from corpus_gen.py rather than hardcoded prompts.

Usage:
    python sae_analysis.py --corpus corpus.json
    python sae_analysis.py --corpus corpus.json --layers 4 5 6 7
    python sae_analysis.py --corpus corpus.json --layer 6
    python sae_analysis.py --corpus corpus.json --top-k 20
    python sae_analysis.py --corpus corpus.json --category pattern
    python sae_analysis.py --corpus corpus.json --no-plots
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

warnings.filterwarnings("ignore", category=UserWarning, module="sae_lens")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sae_lens")
warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_top_features(model, sae, prompt: str, layer: int,
                          top_k: int, device: str) -> dict:
    """
    Run a forward pass and return top-k SAE features at the final token
    position for the given layer.

    Returns:
        {
            "indices": [int, ...],       # top-k feature indices
            "values":  [float, ...],     # corresponding activation values
            "all_activations": tensor,   # full feature vector (for later use)
            "tokens": [str, ...],        # token strings
        }
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    tokens = tokens.to(device)

    hook_name = f"blocks.{layer}.hook_resid_pre"

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)

    resid = cache[hook_name]           # (1, seq_len, d_model)
    features = sae.encode(resid)       # (1, seq_len, d_sae)
    final_features = features[0, -1]   # (d_sae,) — final token position

    top = torch.topk(final_features, k=top_k)
    token_ids = tokens[0].tolist()
    token_strs = [model.tokenizer.decode([t]) for t in token_ids]

    return {
        "indices": top.indices.tolist(),
        "values": top.values.tolist(),
        "all_activations": final_features.cpu(),
        "tokens": token_strs,
    }


# ============================================================================
# PAIR COMPARISON
# ============================================================================

def compare_pair_features(base_result: dict, contrast_result: dict) -> dict:
    """
    Given feature extraction results for a base and contrast prompt,
    return the set-level comparison.
    """
    base_set = set(base_result["indices"])
    contrast_set = set(contrast_result["indices"])

    return {
        "base_only": sorted(base_set - contrast_set),
        "contrast_only": sorted(contrast_set - base_set),
        "shared": sorted(base_set & contrast_set),
        "base_indices": base_result["indices"],
        "contrast_indices": contrast_result["indices"],
        "base_values": base_result["values"],
        "contrast_values": contrast_result["values"],
        "base_tokens": base_result["tokens"],
        "contrast_tokens": contrast_result["tokens"],
    }


# ============================================================================
# CROSS-PAIR AGGREGATION
# ============================================================================

def aggregate_across_pairs(pair_results: list, role: str = "base_only") -> dict:
    """
    Count how often each feature index appears across pairs for a given role
    (base_only, contrast_only, or shared).

    Returns dict: {feature_idx: count}
    """
    counts = defaultdict(int)
    for pr in pair_results:
        for feat in pr["comparison"][role]:
            counts[feat] += 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def find_consistent_features(pair_results: list, min_count: int = 2) -> dict:
    """
    Find features that consistently appear across multiple pairs.
    Returns features appearing >= min_count times for each role.
    """
    roles = ["base_only", "contrast_only", "shared"]
    consistent = {}
    for role in roles:
        counts = aggregate_across_pairs(pair_results, role)
        consistent[role] = {k: v for k, v in counts.items() if v >= min_count}
    return consistent


def jaccard(features_a, features_b):
    a = set(features_a)
    b = set(features_b)
    return len(a & b) / len(a | b)


# ============================================================================
# MAIN ANALYSIS LOOP
# ============================================================================

def run_sae_analysis(model, corpus: list, layers: list[int], top_k: int,
                     device: str, model_name: str,
                     category_filter: Optional[str] = None) -> dict:
    """
    For each layer, for each contrast pair, extract features and compare.
    Returns nested dict: results[layer][pair_id] = {...}
    """
    from sae_lens import SAE

    # Build pairs from corpus
    by_pair = defaultdict(dict)
    for entry in corpus:
        by_pair[entry["pair_id"]][entry["role"]] = entry

    pairs = [(pid, roles) for pid, roles in sorted(by_pair.items())
             if "base" in roles and "contrast" in roles]

    if category_filter:
        pairs = [(pid, roles) for pid, roles in pairs
                 if roles["base"]["category"] == category_filter]
        print(f"  Filtered to category '{category_filter}': {len(pairs)} pairs")

    # Determine SAE release from setup.py registry
    from setup import MODEL_CONFIGS
    cfg = MODEL_CONFIGS.get(model_name, {})
    sae_release = cfg.get("sae_release")
    hook_pattern = cfg.get("hook_pattern", "blocks.{layer}.hook_resid_pre")

    if not sae_release:
        print(f"✗ No SAE release configured for '{model_name}' in setup.py")
        return {}

    all_results = {}

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        hook_name = hook_pattern.format(layer=layer)

        # Load SAE for this layer
        try:
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=hook_name,
                device=device,
            )
        except Exception as e:
            print(f"  ✗ No SAE for layer {layer}: {e}")
            continue

        print(f"  ✓ SAE loaded ({hook_name})")

        layer_results = []

        for pair_id, roles in pairs:
            base_entry = roles["base"]
            contrast_entry = roles["contrast"]

            base_feats = extract_top_features(
                model, sae, base_entry["prompt"], layer, top_k, device
            )
            contrast_feats = extract_top_features(
                model, sae, contrast_entry["prompt"], layer, top_k, device
            )

            comparison = compare_pair_features(base_feats, contrast_feats)
                
            layer_results.append({
                "pair_id": pair_id,
                "category": base_entry["category"],
                "description": base_entry["description"],
                "base_prompt": base_entry["prompt"],
                "contrast_prompt": contrast_entry["prompt"],
                "comparison": comparison,
            })

        # Aggregate across pairs
        consistent = find_consistent_features(layer_results, min_count=2)

        all_results[layer] = {
            "pairs": layer_results,
            "consistent_features": consistent,
            "hook_name": hook_name,
        }

        # Print layer summary
        n_base_only = len(consistent["base_only"])
        n_contrast_only = len(consistent["contrast_only"])
        n_shared = len(consistent["shared"])
        print(f"  Consistent base-only features:     {n_base_only}")
        print(f"  Consistent contrast-only features: {n_contrast_only}")
        print(f"  Consistent shared features:        {n_shared}")

        if consistent["base_only"]:
            top = list(consistent["base_only"].items())[:5]
            print(f"  Top base-only: {top}")
        if consistent["contrast_only"]:
            top = list(consistent["contrast_only"].items())[:5]
            print(f"  Top contrast-only: {top}")

    return all_results


# ============================================================================
# PRINTING
# ============================================================================

def print_summary(all_results: dict, top_n: int = 5):
    print("\n" + "=" * 70)
    print("SAE FEATURE SUMMARY — Consistent features across contrast pairs")
    print("=" * 70)

    for layer, data in sorted(all_results.items()):
        consistent = data["consistent_features"]
        print(f"\nLayer {layer} ({data['hook_name']}):")

        for role, label in [
            ("base_only",     "Base-only  (structured/predictable prompts)"),
            ("contrast_only", "Contrast-only (incoherent/surprising prompts)"),
            ("shared",        "Shared     (appear in both)"),
        ]:
            feats = list(consistent[role].items())[:top_n]
            if feats:
                feat_str = ", ".join(f"#{f}(×{c})" for f, c in feats)
                print(f"  {label}:")
                print(f"    {feat_str}")

    print()


# ============================================================================
# PLOTTING
# ============================================================================

def plot_feature_consistency(all_results: dict, output_dir: Path, top_n: int = 15):
    """
    For each layer, bar chart of most consistent features by role.
    """
    for layer, data in all_results.items():
        consistent = data["consistent_features"]
        hook = data["hook_name"]

        roles = {
            "base_only": "#2196F3",
            "contrast_only": "#F44336",
            "shared": "#4CAF50",
        }

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Layer {layer} — Consistent SAE features ({hook})", fontsize=10)

        for ax, (role, color) in zip(axes, roles.items()):
            items = list(consistent[role].items())[:top_n]
            if not items:
                ax.set_title(f"{role}\n(none)")
                ax.axis("off")
                continue
            feat_ids = [f"#{f}" for f, _ in items]
            counts = [c for _, c in items]
            ax.barh(feat_ids[::-1], counts[::-1], color=color, alpha=0.8)
            ax.set_xlabel("Pairs where feature appears")
            ax.set_title(role.replace("_", " "))
            ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"layer_{layer:02d}_consistent_features.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  ✓ {out_path}")


def plot_feature_heatmap(all_results: dict, output_dir: Path, role: str = "base_only"):
    """
    Heatmap: layers (y) × top features (x), value = consistency count.
    Shows how features emerge and disappear across layers.
    """
    # Collect all features that appear in any layer for this role
    all_features = set()
    for data in all_results.values():
        all_features.update(data["consistent_features"][role].keys())

    if not all_features:
        return

    layers = sorted(all_results.keys())
    features = sorted(all_features)

    matrix = np.zeros((len(layers), len(features)))
    for i, layer in enumerate(layers):
        counts = all_results[layer]["consistent_features"][role]
        for j, feat in enumerate(features):
            matrix[i, j] = counts.get(feat, 0)

    fig, ax = plt.subplots(figsize=(max(10, len(features) * 0.5), max(4, len(layers) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"L{l}" for l in layers])
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([f"#{f}" for f in features], rotation=90, fontsize=7)
    ax.set_title(f"Feature consistency across layers — {role.replace('_', ' ')}")
    plt.colorbar(im, ax=ax, label="# pairs")
    plt.tight_layout()

    out_path = output_dir / f"heatmap_{role}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAE feature analysis over a prompt corpus")
    parser.add_argument("--corpus", type=str, default="corpus.json",
                        help="Path to corpus JSON from corpus_gen.py")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(12)),
                        help="Layers to scan (default: 0-11)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Single layer shorthand (overrides --layers)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k features per prompt (default: 10)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single category")
    parser.add_argument("--output-dir", type=str, default="sae_results",
                        help="Directory for results and plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (default: cpu for stability)")
    args = parser.parse_args()

    layers = [args.layer] if args.layer is not None else args.layers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"✗ Corpus not found: {corpus_path}")
        print("  Run: python corpus_gen.py")
        return 1

    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"\n✓ Loaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")

    # Load model
    print(f"\nLoading model '{args.model}'...")
    from setup import load_model_and_sae
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")

    # Run analysis
    print(f"\nScanning layers {layers}, top-k={args.top_k}...")
    all_results = run_sae_analysis(
        model, corpus, layers, args.top_k,
        cfg["device"], args.model, args.category
    )

    if not all_results:
        print("✗ No results — check SAE release names in setup.py")
        return 1

    # Save results (exclude non-serializable tensors)
    def serialize(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    results_path = output_dir / "sae_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=serialize)
    print(f"\n✓ Results saved to {results_path}")

    # Print summary
    print_summary(all_results)

    # Plots
    if not args.no_plots:
        print(f"Generating plots in {output_dir}/...")
        plot_feature_consistency(all_results, output_dir)
        for role in ["base_only", "contrast_only", "shared"]:
            plot_feature_heatmap(all_results, output_dir, role)

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
