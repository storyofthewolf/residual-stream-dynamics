"""workflows/explore_prompt.py — Single-prompt exploratory analysis.

Runs one forward pass on one or more default prompts, extracting multiple
hook types simultaneously, computing entropy surfaces, and generating plots.

This is the primary workflow for exploratory science — testing hypotheses
on specific prompts before committing to full corpus analysis.

Pipeline:
    extraction.extract_activations()               -> dict[hook_type, ActivationRecord]
    _run_residual_stream_prompt()                  -> list[EntropyRecord]
    _run_logit_lens_prompt()       (--logit-lens)  -> list[EntropyRecord]
    entropy_plots.*                                -> figures

Usage:
    python workflows/explore_prompt.py
    python workflows/explore_prompt.py --model gpt2-small
    python workflows/explore_prompt.py --model pythia-1b --hooks resid_post attn_out mlp_out
    python workflows/explore_prompt.py --logit-lens
    python workflows/explore_prompt.py --logit-lens --no-residual
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

import torch

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import load_model_and_sae, MODEL_CONFIGS
from extraction import extract_activations, HOOK_TYPES, save_activation_record
from entropy_compute import (
    compute_residual_stream_entropy,
    compute_logit_lens_entropy,
    filter_records,
    save_entropy_records,
    RESIDUAL_NORM_KEYS,
)
from entropy_plots import (plot_fixed_position, plot_fixed_layer,
                            plot_2d_surface, plot_hook_comparison)

# ============================================================================
# DEFAULT PROMPTS
# Edit here for quick experiments. Corpus mode uses --corpus flag.
# ============================================================================

DEFAULT_PROMPTS = [
    "The wolf ran through",
    "Ran through wolf the",
    "mucho gusto el lobo"
]

DEFAULT_HOOKS = ["resid_post", "attn_out", "mlp_out"]


# ============================================================================
# SINGLE-PROMPT ITERATION HELPERS
# Orchestration lives here in the workflow layer, not in entropy_compute.py.
# ============================================================================

def _run_residual_stream_prompt(
    activation_records: dict,
    alphas:             list,
    norm_keys:          list,
) -> list:
    """
    Run compute_residual_stream_entropy() for all hook types from one prompt.

    Args:
        activation_records: dict mapping hook_type -> ActivationRecord
        alphas:             list of Renyi alpha values
        norm_keys:          normalization methods to compute

    Returns:
        flat list of EntropyRecord across all hook types and (norm, alpha) combos
    """
    all_entropy = []
    for ht, record in activation_records.items():
        records = compute_residual_stream_entropy(record, alphas, norm_keys)
        all_entropy.extend(records)
        print(f"    Residual stream [{ht}]: "
              f"{len(records)} records ({len(norm_keys)} norms x {len(alphas)} alphas)")
    return all_entropy


def _run_logit_lens_prompt(
    activation_records: dict,
    alphas:             list,
    W_U:                torch.Tensor,
    ln_final,
) -> list:
    """
    Run compute_logit_lens_entropy() for all hook types from one prompt.

    W_U and ln_final are extracted from the model in main() and passed here,
    keeping model components confined to the workflow layer.

    Args:
        activation_records: dict mapping hook_type -> ActivationRecord
        alphas:             list of Renyi alpha values
        W_U:                unembedding matrix [d_model, vocab_size]
        ln_final:           final layer norm callable

    Returns:
        flat list of EntropyRecord with norm_key="logit_lens"
    """
    all_entropy = []
    for ht, record in activation_records.items():
        records = compute_logit_lens_entropy(record, alphas, W_U, ln_final)
        all_entropy.extend(records)
        print(f"    Logit lens      [{ht}]: "
              f"{len(records)} records ({len(alphas)} alphas)")
    return all_entropy


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
                        help="Renyi alpha values (alpha=1.0 is Shannon entropy)")
    parser.add_argument("--norm", type=str, nargs="+", default=RESIDUAL_NORM_KEYS,
                        help="Normalization methods for residual stream: energy, abs, softmax")
    parser.add_argument("--logit-lens", action="store_true",
                        help="Also compute logit-lens entropy (token prediction space)")
    parser.add_argument("--no-residual", action="store_true",
                        help="Skip residual stream entropy (use with --logit-lens)")
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

    for ht in hook_types:
        if ht not in HOOK_TYPES:
            print(f"Unknown hook type '{ht}'. Supported: {sorted(HOOK_TYPES.keys())}")
            return 1

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model '{args.model}'...")
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"  Model ready on {cfg['device']}")
    print(f"  Layers: {model.cfg.n_layers}")
    print(f"  Hooks:  {hook_types}")
    print(f"  alpha:  {alphas}")
    print(f"  Norms:  {norm_keys}")

    # Extract model components for logit lens before processing any prompts.
    # These are tensors, not the model — safe to pass into computation layer.
    W_U      = model.W_U.detach()       # [d_model, vocab_size]
    ln_final = model.ln_final           # callable, stateless layer norm

    # ── Process each prompt ───────────────────────────────────────────────────
    all_entropy_records = []

    for prompt in DEFAULT_PROMPTS:
        print(f"\n{'─'*60}")
        print(f"Prompt: '{prompt}'")

        try:
            activation_records = extract_activations(
                model, prompt, hook_types,
                model_name=args.model,
                device=cfg["device"],
            )
        except KeyError as e:
            print(f"  {e} — skipping unavailable hook")
            continue

        for ht, record in activation_records.items():
            print(f"  {ht}: activations shape {record.activations.shape}  "
                  f"d_model={record.d_model}")

        prompt_entropy = []

        if not args.no_residual:
            rs_records = _run_residual_stream_prompt(
                activation_records, alphas, norm_keys
            )
            prompt_entropy.extend(rs_records)

        if args.logit_lens:
            ll_records = _run_logit_lens_prompt(
                activation_records, alphas, W_U, ln_final
            )
            prompt_entropy.extend(ll_records)

        all_entropy_records.extend(prompt_entropy)

        # Quick summary: Shannon energy entropy at final token
        for ht in hook_types:
            shannon = filter_records(prompt_entropy, hook_type=ht,
                                     norm_key="energy", alpha=1.0)
            if shannon:
                curve = shannon[0].final_token_curve()
                print(f"  Shannon energy [{ht}] final token: "
                      f"{[f'{v:.3f}' for v in curve]}")

        # Quick summary: logit lens Shannon at final token
        if args.logit_lens:
            for ht in hook_types:
                ll_shannon = filter_records(prompt_entropy, hook_type=ht,
                                            norm_key="logit_lens", alpha=1.0)
                if ll_shannon:
                    curve = ll_shannon[0].final_token_curve()
                    print(f"  Shannon logit-lens [{ht}] final token: "
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
                    plot_fixed_position(
                        shannon_energy[0], output_dir,
                        filename=f"entropy_vs_layer_{ht}_{safe}.png"
                    )
                    plot_fixed_layer(
                        shannon_energy[0], output_dir,
                        filename=f"entropy_vs_position_{ht}_{safe}.png"
                    )

                # 2D entropy surface — Shannon, logit lens
                if args.logit_lens:
                    ll_shannon = filter_records(prompt_entropy, hook_type=ht,
                                               norm_key="logit_lens", alpha=1.0)
                    if ll_shannon:
                        plot_2d_surface(
                            ll_shannon[0], output_dir,
                            filename=f"entropy_surface_logit_lens_{ht}_{safe}.png"
                        )
                        plot_fixed_position(
                            ll_shannon[0], output_dir,
                            filename=f"entropy_vs_layer_logit_lens_{ht}_{safe}.png"
                        )

            # Hook comparison at final token, Shannon energy norm
            comparison_hooks = [ht for ht in ["resid_post", "attn_out", "mlp_out"]
                                 if ht in hook_types]
            if len(comparison_hooks) > 1 and not args.no_residual:
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

    # ── Save data ─────────────────────────────────────────────────────────────
    if args.save_data and all_entropy_records:
        data_path = output_dir / f"entropy_records_{args.model}.npz"
        save_entropy_records(all_entropy_records, data_path)

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
