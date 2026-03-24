"""workflows/explore_prompt.py — Single-prompt exploratory analysis.

Runs one forward pass on one or more default prompts, extracting multiple
hook types simultaneously, computing entropy surfaces, and generating plots.

This is the primary workflow for exploratory science — testing hypotheses
on specific prompts before committing to full corpus analysis.

Pipeline:
    extraction.extract_activations()     → dict[hook_type, ActivationRecord]
    computation.compute_entropy_surface() → list[EntropyRecord]
    entropy_plots.*                       → figures

Usage:
    python workflows/explore_prompt.py
    python workflows/explore_prompt.py --model gpt2-small
    python workflows/explore_prompt.py --model pythia-1b --hooks resid_post attn_out mlp_out
    python workflows/explore_prompt.py --alpha 0.5 1.0 2.0
    python workflows/explore_prompt.py --output-dir figures/explore
    python workflows/explore_prompt.py --no-plots
    python workflows/explore_prompt.py --save-data
"""

import sys
import argparse
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

# Allow running from project root or from workflows/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import load_model_and_sae, MODEL_CONFIGS
from extraction import extract_activations, HOOK_TYPES, save_activation_record
from computation import compute_entropy_surface, filter_records, save_entropy_records
from entropy_plots import (plot_fixed_position, plot_fixed_layer,
                            plot_2d_surface, plot_hook_comparison)

# ============================================================================
# DEFAULT PROMPTS
# Edit here for quick experiments. Corpus mode uses --corpus flag.
# ============================================================================

DEFAULT_PROMPTS = [
    "The wolf ran through",
    "Ran through wolf the",
]

# Default hooks to extract in one forward pass
DEFAULT_HOOKS = ["resid_post", "attn_out", "mlp_out"]


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-prompt exploratory entropy analysis"
    )
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--hooks", type=str, nargs="+", default=DEFAULT_HOOKS,
                        help=f"Hook types to extract. Choices: {sorted(HOOK_TYPES.keys())}")
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0],
                        help="Rényi alpha values (alpha=1.0 is Shannon entropy)")
    parser.add_argument("--norm", type=str, nargs="+", default=["energy", "abs", "softmax"],
                        help="Normalization methods: energy, abs, softmax")
    parser.add_argument("--output-dir", type=str, default="figures/explore",
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

    # Validate hook types before loading model
    for ht in hook_types:
        if ht not in HOOK_TYPES:
            print(f"✗ Unknown hook type '{ht}'. Supported: {sorted(HOOK_TYPES.keys())}")
            return 1

    # Load model
    print(f"\nLoading model '{args.model}'...")
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hooks:  {hook_types}")
    print(f"  α:      {alphas}")
    print(f"  Norms:  {norm_keys}")

    # ── Process each prompt ───────────────────────────────────────────────────
    all_entropy_records = []

    for prompt in DEFAULT_PROMPTS:
        print(f"\n{'─'*60}")
        print(f"Prompt: '{prompt}'")

        # One forward pass → multiple ActivationRecords
        try:
            activation_records = extract_activations(
                model, prompt, hook_types,
                model_name=args.model,
                device=cfg["device"],
            )
        except KeyError as e:
            print(f"  ⚠ {e} — skipping unavailable hook")
            continue

        for ht, record in activation_records.items():
            print(f"  {ht}: activations shape {record.activations.shape}  "
                  f"d_model={record.d_model}")

        # Compute entropy surfaces for all hooks
        prompt_entropy = []
        for ht, act_record in activation_records.items():
            records = compute_entropy_surface(act_record, alphas, norm_keys)
            prompt_entropy.extend(records)
            print(f"  ✓ Entropy computed for {ht} "
                  f"({len(records)} norm×alpha combinations)")

        all_entropy_records.extend(prompt_entropy)

        # Print quick summary: Shannon energy entropy at final token
        for ht in hook_types:
            shannon = filter_records(prompt_entropy, hook_type=ht,
                                     norm_key="energy", alpha=1.0)
            if shannon:
                curve = shannon[0].final_token_curve()
                print(f"  Shannon (energy, final token) [{ht}]: "
                      f"{[f'{v:.3f}' for v in curve]}")

        # ── Plots per prompt ──────────────────────────────────────────────────
        if not args.no_plots:
            safe = prompt.replace(" ", "_")[:30]
            print(f"\n  Generating plots...")

            for ht in hook_types:
                # 2D entropy surface — Shannon, energy norm
                shannon_energy = filter_records(prompt_entropy, hook_type=ht,
                                               norm_key="energy", alpha=1.0)
                if shannon_energy:
                    plot_2d_surface(
                        shannon_energy[0], output_dir,
                        filename=f"entropy_surface_{ht}_{safe}.png"
                    )

                # 1D: entropy vs layer (all token positions)
                shannon_energy = filter_records(prompt_entropy, hook_type=ht,
                                               norm_key="energy", alpha=1.0)
                if shannon_energy:
                    plot_fixed_position(
                        shannon_energy[0], output_dir,
                        filename=f"entropy_vs_layer_{ht}_{safe}.png"
                    )

                # 1D: entropy vs token position (all layers)
                if shannon_energy:
                    plot_fixed_layer(
                        shannon_energy[0], output_dir,
                        filename=f"entropy_vs_position_{ht}_{safe}.png"
                    )

            # Hook comparison: overlay resid_post, attn_out, mlp_out
            # at final token position, Shannon entropy, energy norm
            comparison_hooks = [ht for ht in ["resid_post", "attn_out", "mlp_out"]
                                 if ht in hook_types]
            if len(comparison_hooks) > 1:
                comp_records = []
                for ht in comparison_hooks:
                    matched = filter_records(prompt_entropy, hook_type=ht,
                                            norm_key="energy", alpha=1.0)
                    if matched:
                        comp_records.append(matched[0])

                if len(comp_records) > 1:
                    plot_hook_comparison(
                        comp_records, output_dir,
                        token_position=-1,
                        filename=f"hook_comparison_{safe}.png"
                    )

    # ── Save data for multi-model comparison ─────────────────────────────────
    if args.save_data and all_entropy_records:
        data_path = output_dir / f"entropy_records_{args.model}.npz"
        save_entropy_records(all_entropy_records, data_path)

    print(f"\n✓ Done. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
