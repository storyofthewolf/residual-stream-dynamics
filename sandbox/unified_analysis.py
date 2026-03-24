"""
unified_analysis.py — Joint residual stream entropy and SAE feature divergence.

Hypothesis: the layer at which residual stream entropy drops sharply
is the same layer at which SAE feature sets diverge between semantically
distinct prompt pairs (Jaccard similarity drops).

For each contrast pair, at each layer:
  - Entropy:  effective_rank() on hook_resid_pre activation vector
  - Jaccard:  overlap of top-k SAE features between base and contrast prompt

Both signals are computed from the same forward pass and plotted together.

Usage:
    python unified_analysis.py
    python unified_analysis.py --model gpt2-small --top-k 10
    python unified_analysis.py --category pattern
    python unified_analysis.py --no-plots
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
warnings.filterwarnings("ignore", category=UserWarning, module="sae_lens")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sae_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

# Import from sibling scripts — each remains independently executable
from entropy_residual_stream import effective_rank
from sae_analysis import jaccard


# ============================================================================
# CORE: single forward pass, both signals
# ============================================================================

def analyze_pair_at_layer(model, sae, base_prompt: str, contrast_prompt: str,
                           layer: int, top_k: int, device: str) -> dict:
    """
    Run forward pass for both prompts at a single layer.
    Returns entropy and top-k SAE features for each, plus Jaccard similarity.

    Both signals extracted from the same hook_resid_pre activation.
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    results = {}

    for role, prompt in [("base", base_prompt), ("contrast", contrast_prompt)]:
        tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        resid = cache[hook_name]            # (1, seq_len, d_model)
        vec = resid[0, -1, :].float().cpu() # final token position

        # Entropy from raw residual stream
        ent = effective_rank(vec)

        # SAE feature extraction
        features = sae.encode(resid)        # (1, seq_len, d_sae)
        final_features = features[0, -1]    # (d_sae,)
        top = torch.topk(final_features, k=top_k)

        results[role] = {
            "entropy": ent,
            "feature_indices": top.indices.tolist(),
            "feature_values": top.values.tolist(),
        }

    # Jaccard similarity between base and contrast feature sets
    j = jaccard(
        results["base"]["feature_indices"],
        results["contrast"]["feature_indices"]
    )

    return {
        "base_entropy": results["base"]["entropy"],
        "contrast_entropy": results["contrast"]["entropy"],
        "jaccard": j,
        "base_features": results["base"]["feature_indices"],
        "contrast_features": results["contrast"]["feature_indices"],
    }


# ============================================================================
# MAIN ANALYSIS LOOP
# ============================================================================

def run_unified_analysis(model, corpus: list, layers: list,
                         top_k: int, device: str, model_name: str,
                         category_filter: Optional[str] = None) -> dict:
    """
    For each layer, for each contrast pair:
      - compute entropy for base and contrast prompt
      - compute Jaccard similarity of top-k SAE features

    Returns:
        {
            layer: {
                pairs: [
                    {
                        pair_id, category, description,
                        base_prompt, contrast_prompt,
                        base_entropy, contrast_entropy, jaccard
                    },
                    ...
                ]
            }
        }
    """
    from sae_lens import SAE
    from setup import MODEL_CONFIGS

    # Build contrast pairs from corpus
    by_pair = defaultdict(dict)
    for entry in corpus:
        by_pair[entry["pair_id"]][entry["role"]] = entry

    pairs = [(pid, roles) for pid, roles in sorted(by_pair.items())
             if "base" in roles and "contrast" in roles]

    if category_filter:
        pairs = [(pid, roles) for pid, roles in pairs
                 if roles["base"]["category"] == category_filter]
        print(f"  Filtered to category '{category_filter}': {len(pairs)} pairs")

    # SAE release from setup.py registry
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

        try:
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=hook_name,
                device=device,
            )
        except Exception as e:
            print(f"  ✗ No SAE for layer {layer}: {e}")
            continue

        print(f"  ✓ SAE loaded")

        layer_results = []

        for pair_id, roles in pairs:
            base_entry = roles["base"]
            contrast_entry = roles["contrast"]

            pair_data = analyze_pair_at_layer(
                model, sae,
                base_entry["prompt"],
                contrast_entry["prompt"],
                layer, top_k, device
            )

            layer_results.append({
                "pair_id": pair_id,
                "category": base_entry["category"],
                "description": base_entry["description"],
                "base_prompt": base_entry["prompt"],
                "contrast_prompt": contrast_entry["prompt"],
                **pair_data,
            })

        all_results[layer] = {"pairs": layer_results}

        # Quick layer summary
        mean_j = np.mean([r["jaccard"] for r in layer_results])
        mean_ent_base = np.mean([r["base_entropy"] for r in layer_results])
        mean_ent_contrast = np.mean([r["contrast_entropy"] for r in layer_results])
        print(f"  Mean Jaccard:          {mean_j:.3f}")
        print(f"  Mean entropy (base):   {mean_ent_base:.3f}")
        print(f"  Mean entropy (contrast): {mean_ent_contrast:.3f}")

    return all_results


# ============================================================================
# AGGREGATE CURVES
# ============================================================================

def extract_aggregate_curves(all_results: dict) -> dict:
    """
    Compute mean curves across all pairs for each layer.
    Returns dict of lists indexed by layer.
    """
    layers = sorted(all_results.keys())
    curves = {
        "layers": layers,
        "mean_jaccard": [],
        "mean_entropy_base": [],
        "mean_entropy_contrast": [],
        "std_jaccard": [],
        "std_entropy_base": [],
        "std_entropy_contrast": [],
    }

    for layer in layers:
        pairs = all_results[layer]["pairs"]
        j = [r["jaccard"] for r in pairs]
        eb = [r["base_entropy"] for r in pairs]
        ec = [r["contrast_entropy"] for r in pairs]

        curves["mean_jaccard"].append(np.mean(j))
        curves["mean_entropy_base"].append(np.mean(eb))
        curves["mean_entropy_contrast"].append(np.mean(ec))
        curves["std_jaccard"].append(np.std(j))
        curves["std_entropy_base"].append(np.std(eb))
        curves["std_entropy_contrast"].append(np.std(ec))

    return curves


# ============================================================================
# PLOTTING
# ============================================================================

def plot_aggregate(curves: dict, output_path: str = "unified_aggregate.png"):
    """
    Dual y-axis plot: entropy (left) and Jaccard (right) vs layer.
    The visual test of the hypothesis — do they co-vary?
    """
    layers = curves["layers"]
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Entropy on left axis
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Residual stream entropy (bits)", color="#1565C0")
    ax1.tick_params(axis="y", labelcolor="#1565C0")

    ax1.plot(layers, curves["mean_entropy_base"], color="#1565C0",
             linewidth=2.5, marker="o", markersize=5, label="entropy (base)")
    ax1.fill_between(layers,
                     np.array(curves["mean_entropy_base"]) - np.array(curves["std_entropy_base"]),
                     np.array(curves["mean_entropy_base"]) + np.array(curves["std_entropy_base"]),
                     color="#1565C0", alpha=0.12)

    ax1.plot(layers, curves["mean_entropy_contrast"], color="#42A5F5",
             linewidth=2.5, marker="o", markersize=5, linestyle="--",
             label="entropy (contrast)")
    ax1.fill_between(layers,
                     np.array(curves["mean_entropy_contrast"]) - np.array(curves["std_entropy_contrast"]),
                     np.array(curves["mean_entropy_contrast"]) + np.array(curves["std_entropy_contrast"]),
                     color="#42A5F5", alpha=0.12)

    # Jaccard on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Jaccard similarity (feature overlap)", color="#B71C1C")
    ax2.tick_params(axis="y", labelcolor="#B71C1C")
    ax2.set_ylim(0, 1)

    ax2.plot(layers, curves["mean_jaccard"], color="#B71C1C",
             linewidth=2.5, marker="s", markersize=5, label="Jaccard")
    ax2.fill_between(layers,
                     np.array(curves["mean_jaccard"]) - np.array(curves["std_jaccard"]),
                     np.array(curves["mean_jaccard"]) + np.array(curves["std_jaccard"]),
                     color="#B71C1C", alpha=0.12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    ax1.set_xticks(layers)
    ax1.grid(alpha=0.3)
    fig.suptitle("Residual stream entropy vs SAE feature divergence across layers\n"
                 "(mean ± 1σ across all contrast pairs)", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✓ Aggregate plot saved to {output_path}")


def plot_by_category(all_results: dict, output_dir: Path):
    """
    One aggregate plot per category — same dual y-axis design.
    Lets you see whether the hypothesis holds differently across categories.
    """
    # Group pairs by category across all layers
    categories = set()
    for layer_data in all_results.values():
        for r in layer_data["pairs"]:
            categories.add(r["category"])

    for category in sorted(categories):
        layers = sorted(all_results.keys())
        mean_j, mean_eb, mean_ec = [], [], []

        for layer in layers:
            pairs = [r for r in all_results[layer]["pairs"]
                     if r["category"] == category]
            if not pairs:
                continue
            mean_j.append(np.mean([r["jaccard"] for r in pairs]))
            mean_eb.append(np.mean([r["base_entropy"] for r in pairs]))
            mean_ec.append(np.mean([r["contrast_entropy"] for r in pairs]))

        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Entropy (bits)", color="#1565C0")
        ax1.tick_params(axis="y", labelcolor="#1565C0")
        ax1.plot(layers, mean_eb, color="#1565C0", linewidth=2,
                 marker="o", markersize=4, label="entropy (base)")
        ax1.plot(layers, mean_ec, color="#42A5F5", linewidth=2,
                 marker="o", markersize=4, linestyle="--", label="entropy (contrast)")

        ax2 = ax1.twinx()
        ax2.set_ylabel("Jaccard similarity", color="#B71C1C")
        ax2.tick_params(axis="y", labelcolor="#B71C1C")
        ax2.set_ylim(0, 1)
        ax2.plot(layers, mean_j, color="#B71C1C", linewidth=2,
                 marker="s", markersize=4, label="Jaccard")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        ax1.set_xticks(layers)
        ax1.grid(alpha=0.3)
        fig.suptitle(f"Entropy vs feature divergence — {category}", fontsize=10)
        plt.tight_layout()

        out_path = output_dir / f"unified_{category}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"✓ {out_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Joint entropy and SAE feature divergence analysis"
    )
    parser.add_argument("--corpus", type=str, default="corpus.json")
    parser.add_argument("--model", type=str, default="gpt2-small")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

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
    print(f"\nScanning layers {args.layers}, top-k={args.top_k}...")
    all_results = run_unified_analysis(
        model, corpus, args.layers, args.top_k,
        cfg["device"], args.model, args.category
    )

    if not all_results:
        print("✗ No results — check SAE release names in setup.py")
        return 1

    # Save results
    results_path = output_dir / "unified_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Extract and print aggregate curves
    curves = extract_aggregate_curves(all_results)
    print("\nAggregate curves (mean across all pairs):")
    print(f"  {'Layer':<8} {'Entropy(base)':<16} {'Entropy(contrast)':<20} {'Jaccard':<10}")
    print(f"  {'-'*54}")
    for i, layer in enumerate(curves["layers"]):
        print(f"  {layer:<8} "
              f"{curves['mean_entropy_base'][i]:<16.3f} "
              f"{curves['mean_entropy_contrast'][i]:<20.3f} "
              f"{curves['mean_jaccard'][i]:<10.3f}")

    # Plots
    if not args.no_plots:
        print(f"\nGenerating plots in {output_dir}/...")
        plot_aggregate(curves, str(output_dir / "unified_aggregate.png"))
        plot_by_category(all_results, output_dir)

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
