"""workflows/corpus_analysis.py — Full corpus base/contrast entropy analysis.

Runs entropy analysis over a corpus JSON (produced by corpus_gen.py),
computing entropy surfaces for all base/contrast pairs, saving results,
and generating statistical summary plots.

Pipeline:
    extraction.extract_corpus()          → dict[hook_type, list[ActivationRecord]]
    computation.compute_entropy_corpus() → list[EntropyRecord]
    entropy_plots.plot_overall_mean()    → figures
    entropy_plots.plot_category()        → figures

Usage:
    python workflows/corpus_analysis.py --corpus corpus.json
    python workflows/corpus_analysis.py --corpus corpus.json --model pythia-1b
    python workflows/corpus_analysis.py --corpus corpus.json --hooks resid_post attn_out mlp_out
    python workflows/corpus_analysis.py --corpus corpus.json --category pattern
    python workflows/corpus_analysis.py --corpus corpus.json --no-plots
    python workflows/corpus_analysis.py --corpus corpus.json --save-data
"""

import sys
import json
import argparse
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import load_model_and_sae, MODEL_CONFIGS
from extraction import extract_corpus, HOOK_TYPES
from computation import (compute_entropy_corpus, print_summary,
                          save_entropy_records, load_entropy_records)
from entropy_plots import plot_overall_mean, plot_category

DEFAULT_HOOKS = ["resid_post"]


def main():
    parser = argparse.ArgumentParser(
        description="Corpus-level residual stream entropy analysis"
    )
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to corpus JSON from corpus_gen.py")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--hooks", type=str, nargs="+", default=DEFAULT_HOOKS,
                        help=f"Hook types to extract. Choices: {sorted(HOOK_TYPES.keys())}")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0],
                        help="Rényi alpha values")
    parser.add_argument("--norm", type=str, nargs="+", default=["energy", "abs", "softmax"],
                        help="Normalization methods: energy, abs, softmax")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")
    parser.add_argument("--output-dir", type=str, default="figures/corpus",
                        help="Directory for plots and saved data")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--save-data", action="store_true",
                        help="Save EntropyRecords to .npz for later multi-model plots")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    alphas     = sorted(set(args.alpha))
    hook_types = args.hooks
    norm_keys  = args.norm
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate
    for ht in hook_types:
        if ht not in HOOK_TYPES:
            print(f"✗ Unknown hook type '{ht}'. Supported: {sorted(HOOK_TYPES.keys())}")
            return 1

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
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hooks:  {hook_types}")
    print(f"  α:      {alphas}")

    # ── Extraction ────────────────────────────────────────────────────────────
    print(f"\nExtracting activations across corpus...")
    activation_dict = extract_corpus(
        model, corpus, hook_types,
        model_name=args.model,
        device=cfg["device"],
        category_filter=args.category,
    )

    # ── Computation ───────────────────────────────────────────────────────────
    print(f"\nComputing entropy surfaces...")
    all_entropy_records = []

    for ht in hook_types:
        act_records = activation_dict[ht]
        print(f"  Computing entropy for hook '{ht}' "
              f"({len(act_records)} prompts)...")
        entropy_records = compute_entropy_corpus(act_records, alphas, norm_keys)
        all_entropy_records.extend(entropy_records)

    print(f"\n✓ Total EntropyRecords: {len(all_entropy_records)}")
    print_summary(all_entropy_records, alphas)

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save_data:
        data_path = output_dir / f"entropy_records_{args.model}.npz"
        save_entropy_records(all_entropy_records, data_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\nGenerating plots in {output_dir}/...")
        categories = sorted(set(r.category for r in all_entropy_records
                                if r.category))

        for ht in hook_types:
            plot_overall_mean(
                all_entropy_records, alphas, output_dir,
                model_name=args.model, hook_type=ht
            )
            for cat in categories:
                plot_category(
                    all_entropy_records, cat, alphas, output_dir,
                    model_name=args.model, hook_type=ht
                )

    print(f"\n✓ Done. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
