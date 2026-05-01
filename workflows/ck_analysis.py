"""workflows/ck_analysis.py — c_k spectrum analysis of the residual stream.

Orchestrates extraction, computation, and plotting for the c_k spectrum.

Pipeline:
    extraction.extract_corpus()                      -> dict[hook_type, list[ActivationRecord]]
    ck_spectrum_compute.compute_wu_svd_full()        -> (S, Vh)  one-time per model
    ck_spectrum_compute.compute_ck_spectrum()        -> CkRecord  per prompt
    ck_spectrum_plots.plot_single_prompt_diagnostic()
    ck_spectrum_plots.plot_heatmap_lasttoken()
    ck_spectrum_plots.plot_heatmap_alltokens()
    ck_spectrum_plots.plot_com_vs_layer()
    ck_spectrum_plots.plot_cumpower_vs_k()
    ck_spectrum_plots.plot_delta_ck_heatmap()
    ck_spectrum_plots.plot_heatmap_variance_lasttoken()
    ck_spectrum_plots.plot_variance_ratio_vs_k()
    ck_spectrum_plots.plot_com_variance_vs_layer()

Usage:
    python workflows/ck_analysis.py --corpus corpus.json
    python workflows/ck_analysis.py --corpus corpus.json --model gpt2-small
    python workflows/ck_analysis.py --corpus corpus.json --layer 8 --token -1
    python workflows/ck_analysis.py --corpus corpus.json --save-data
    python workflows/ck_analysis.py --corpus corpus.json --no-plots
    python workflows/ck_analysis.py --corpus corpus.json --category pattern
"""

import sys
import json
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from setup import load_model_and_sae, MODEL_CONFIGS
from extraction import extract_corpus, HOOK_TYPES, save_activation_records
from ck_spectrum_compute import (
    compute_wu_svd_full,
    compute_ck_spectrum,
    save_ck_records,
    load_ck_records,
)
from ck_spectrum_plots import (
    plot_single_prompt_diagnostic,
    plot_heatmap_lasttoken,
    plot_heatmap_alltokens,
    plot_com_vs_layer,
    plot_cumpower_vs_k,
    plot_delta_ck_heatmap,
    plot_heatmap_variance_lasttoken,
    plot_variance_ratio_vs_k,
    plot_com_variance_vs_layer,
)


DEFAULT_HOOKS = ["resid_post"]


def _safe_model_name(model_name: str) -> str:
    return model_name.replace(" ", "_").replace("-", "_")


def _run_ck_corpus(
    activation_records: list,
    S:                  "torch.Tensor",
    Vh:                 "torch.Tensor",
) -> list:
    """
    Iterate compute_ck_spectrum() over a list of ActivationRecords.

    Args:
        activation_records: list of ActivationRecord (same hook type)
        S:   singular values from compute_wu_svd_full()
        Vh:  right singular vectors from compute_wu_svd_full()

    Returns:
        list of CkRecord, one per prompt
    """
    all_records = []
    n = len(activation_records)
    for i, record in enumerate(activation_records):
        ck_rec = compute_ck_spectrum(record, S, Vh)
        all_records.append(ck_rec)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    c_k spectrum: {i+1}/{n} prompts...")
    return all_records


def main():
    parser = argparse.ArgumentParser(
        description="c_k spectrum analysis of the residual stream"
    )

    parser.add_argument("--corpus", type=str,
                        default=str(_PROJECT_ROOT / "corpus" / "base_vs_contrast_n50.json"),
                        help="Path to corpus JSON from corpus_gen.py")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--hooks", type=str, nargs="+", default=DEFAULT_HOOKS,
                        help=f"Hook types to extract. Choices: {sorted(HOOK_TYPES.keys())}")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index for single-prompt 2×2 diagnostic "
                             "(default: middle layer)")
    parser.add_argument("--token", type=int, default=-1,
                        help="Token position for single-prompt 2×2 diagnostic "
                             "(default: -1 = final token)")
    parser.add_argument("--prompt-index", type=int, default=0,
                        help="Index into the corpus for the single-prompt diagnostic "
                             "(default: 0, first prompt)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")
    parser.add_argument("--output-dir-plots", type=str,
                        default=str(_PROJECT_ROOT / "figures" / "workflows" / "ck_analysis"),
                        help="Directory for saved plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--output-dir-data", type=str,
                        default=str(_PROJECT_ROOT / "data"),
                        help="Directory for saved data")
    parser.add_argument("--save-data", action="store_true",
                        help="Save CkRecords to .npz for later analysis")
    parser.add_argument("--load-data", type=str, default=None,
                        help="Load precomputed CkRecords from .npz "
                             "(skips extraction and computation, goes straight to plots)")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Optional tag appended to output filenames")
    parser.add_argument("--summary-layers", type=int, nargs="+", default=[1, 3, 6, 9, 11],
                        help="Layer indices for cumulative power fraction subplots "
                             "(default: [1, 3, 6, 9, 11])")
    parser.add_argument("--skip-layer0", action=argparse.BooleanOptionalAction, default=True,
                        help="Drop layer 0 from heatmaps and summary plots (default: True; "
                             "use --no-skip-layer0 to include layer 0)")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    output_dir_plots = Path(args.output_dir_plots)
    output_dir_plots.mkdir(parents=True, exist_ok=True)
    output_dir_data = Path(args.output_dir_data)
    output_dir_data.mkdir(parents=True, exist_ok=True)

    run_tag    = f"_{args.run_tag}" if args.run_tag else ""
    hook_types = args.hooks

    for ht in hook_types:
        if ht not in HOOK_TYPES:
            print(f"Unknown hook type '{ht}'. Supported: {sorted(HOOK_TYPES.keys())}")
            return 1

    corpus_path = Path(args.corpus)
    corpus_tag  = corpus_path.stem
    if not corpus_path.exists():
        corpus_path = Path("corpus") / corpus_path
    if not corpus_path.exists():
        print(f"Corpus not found: {args.corpus}")
        return 1

    with open(corpus_path) as f:
        corpus = json.load(f)
    print(f"\nLoaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")
    print(f"  Corpus file:  {corpus_path.name}")

    # ── Fast path: load precomputed records ──────────────────────────────────
    if args.load_data is not None:
        print(f"\nLoading precomputed CkRecords from {args.load_data}...")
        all_ck_records = load_ck_records(args.load_data)
        print(f"  Loaded {len(all_ck_records)} records.")
        model_name = all_ck_records[0].model_name if all_ck_records else args.model
        n_layers   = all_ck_records[0].n_layers   if all_ck_records else 12
        layer      = args.layer if args.layer is not None else n_layers // 2

        if not args.no_plots:
            safe      = _safe_model_name(model_name)
            base_stem = f"ck_{safe}_{corpus_tag}{run_tag}"
            if all_ck_records:
                idx = min(args.prompt_index, len(all_ck_records) - 1)
                plot_single_prompt_diagnostic(
                    all_ck_records[idx], layer=layer, token=args.token,
                    save_path=str(output_dir_plots /
                        f"ck_diagnostic_{safe}_{corpus_tag}_L{layer}{run_tag}.png"),
                )
            plot_heatmap_lasttoken(
                all_ck_records, model_name, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_heatmap_lasttoken.png"),
            )
            plot_heatmap_alltokens(
                all_ck_records, model_name, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_heatmap_alltokens_nolayer0.png"),
            )
            plot_com_vs_layer(
                all_ck_records, model_name, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_com_vs_layer.png"),
            )
            plot_cumpower_vs_k(
                all_ck_records, model_name, summary_layers=args.summary_layers,
                skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_cumpower_vs_k.png"),
            )
            plot_delta_ck_heatmap(
                all_ck_records, model_name, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_delta_ck_heatmap.png"),
            )
            plot_heatmap_variance_lasttoken(
                all_ck_records, model_name, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_heatmap_variance_lasttoken.png"),
            )
            plot_variance_ratio_vs_k(
                all_ck_records, model_name, summary_layers=args.summary_layers,
                skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_variance_ratio_vs_k.png"),
            )
            plot_com_variance_vs_layer(
                all_ck_records, model_name, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_com_variance_vs_layer.png"),
            )

        print(f"\nDone. Results in {output_dir_plots}/\n")
        return 0

    # ── Full path: load model, extract, compute, plot ─────────────────────────
    print(f"\nLoading model '{args.model}'...")
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    n_layers = model.cfg.n_layers
    d_model  = model.cfg.d_model
    print(f"  Model ready on {cfg['device']}")
    print(f"  Layers:  {n_layers}")
    print(f"  d_model: {d_model}")
    print(f"  Hooks:   {hook_types}")

    layer = args.layer if args.layer is not None else n_layers // 2
    if not (0 <= layer < n_layers):
        print(f"--layer {layer} out of range for model with {n_layers} layers.")
        return 1

    print(f"\nComputing SVD of W_U ({d_model}×{model.W_U.shape[1]})...")
    W_U = model.W_U.detach()
    S, Vh = compute_wu_svd_full(W_U)
    print(f"  S shape: {list(S.shape)}  Vh shape: {list(Vh.shape)}")

    print(f"\nExtracting activations across corpus...")
    activation_dict = extract_corpus(
        model, corpus, hook_types,
        model_name=args.model,
        device=cfg["device"],
        category_filter=args.category,
    )

    all_ck_records = []
    safe = _safe_model_name(args.model)

    for ht in hook_types:
        act_records = activation_dict[ht]
        print(f"\n  Hook '{ht}' ({len(act_records)} prompts):")
        ck_records = _run_ck_corpus(act_records, S, Vh)
        all_ck_records.extend(ck_records)

    print(f"\n  Total CkRecords: {len(all_ck_records)}")

    if args.save_data:
        for ht in hook_types:
            ht_records = [r for r in all_ck_records if r.hook_type == ht]
            data_path = output_dir_data / f"ck_records_{args.model}_{corpus_tag}_{ht}{run_tag}.npz"
            save_ck_records(ht_records, data_path)

        for ht in hook_types:
            data_path = output_dir_data / f"activation_records_{args.model}_{corpus_tag}_{ht}{run_tag}.npz"
            save_activation_records(activation_dict[ht], data_path)

    if not args.no_plots:
        print(f"\nGenerating plots in {output_dir_plots}/...")

        for ht in hook_types:
            ht_records = [r for r in all_ck_records if r.hook_type == ht]
            if not ht_records:
                continue

            base_stem = f"ck_{safe}_{corpus_tag}_{ht}{run_tag}"

            idx = min(args.prompt_index, len(ht_records) - 1)
            plot_single_prompt_diagnostic(
                ht_records[idx], layer=layer, token=args.token,
                save_path=str(output_dir_plots /
                    f"ck_diagnostic_{safe}_{corpus_tag}_{ht}_L{layer}{run_tag}.png"),
            )
            plot_heatmap_lasttoken(
                ht_records, args.model, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_heatmap_lasttoken.png"),
            )
            plot_heatmap_alltokens(
                ht_records, args.model, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_heatmap_alltokens_nolayer0.png"),
            )
            plot_com_vs_layer(
                ht_records, args.model, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_com_vs_layer.png"),
            )
            plot_cumpower_vs_k(
                ht_records, args.model, summary_layers=args.summary_layers,
                skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_cumpower_vs_k.png"),
            )
            plot_delta_ck_heatmap(
                ht_records, args.model, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_delta_ck_heatmap.png"),
            )
            plot_heatmap_variance_lasttoken(
                ht_records, args.model, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_heatmap_variance_lasttoken.png"),
            )
            plot_variance_ratio_vs_k(
                ht_records, args.model, summary_layers=args.summary_layers,
                skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_variance_ratio_vs_k.png"),
            )
            plot_com_variance_vs_layer(
                ht_records, args.model, skip_layer0=args.skip_layer0,
                save_path=str(output_dir_plots / f"{base_stem}_com_variance_vs_layer.png"),
            )

    print(f"\nDone. Results in {output_dir_plots}/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
