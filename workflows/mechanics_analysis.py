"""workflows/mechanics_analysis.py — Full corpus residual stream mechanics analysis.

Runs mechanical quantity analysis over a corpus JSON (produced by corpus_gen.py),
computing five scalar curves per prompt at the final token position, saving
results, and generating statistical summary plots.

Pipeline:
    extraction.extract_corpus()            -> dict[hook_type, list[ActivationRecord]]
    mechanics_compute.compute_mechanics()  -> MechanicsRecord per prompt
    mechanics_plots.plot_mechanics_overview()  -> figure
    mechanics_plots.plot_mechanics_category()  -> figure per category

Usage:
    python workflows/mechanics_analysis.py --corpus corpus.json
    python workflows/mechanics_analysis.py --corpus corpus.json --model pythia-1b
    python workflows/mechanics_analysis.py --corpus corpus.json --category pattern
    python workflows/mechanics_analysis.py --corpus corpus.json --no-plots
    python workflows/mechanics_analysis.py --corpus corpus.json --save-data
    python workflows/mechanics_analysis.py --corpus corpus.json --save-data --run-tag v2
"""

import sys
import json
import argparse
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "utils"))
sys.path.insert(0, str(_PROJECT_ROOT / "plotting"))

from model_loader import load_model_and_sae
from extraction import extract_corpus
from mechanics_compute import compute_mechanics, save_mechanics_records
from mechanics_plots import plot_mechanics_overview, plot_mechanics_category


# ============================================================================
# CORPUS ITERATION HELPER
# ============================================================================

def _run_mechanics_corpus(activation_records: list) -> list:
    """
    Iterate compute_mechanics() over a list of ActivationRecords.

    Args:
        activation_records: list of ActivationRecord (all from hook_type resid_post)

    Returns:
        list of MechanicsRecord, one per prompt
    """
    mech_records = []
    n = len(activation_records)
    for i, record in enumerate(activation_records):
        mech_records.append(compute_mechanics(record, token_pos=-1))
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    Mechanics: {i+1}/{n} prompts...")
    return mech_records


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Corpus-level residual stream mechanics analysis"
    )

    # ── Required ──
    parser.add_argument("--corpus", type=str,
                        default=str(_PROJECT_ROOT / "corpus" / "base_vs_contrast_n50.json"),
                        help="Path to corpus JSON from corpus_gen.py")

    # ── Model ──
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in utils/model_loader.py MODEL_CONFIGS)")

    # ── Filtering ──
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")

    # ── Output ──
    parser.add_argument("--output-dir-plots", type=str,
                        default=str(_PROJECT_ROOT / "figures" / "workflows" / "mechanics_analysis"),
                        help="Directory for saved plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--output-dir-data", type=str,
                        default=str(_PROJECT_ROOT / "data"),
                        help="Directory for saved data")
    parser.add_argument("--save-data", action="store_true",
                        help="Save MechanicsRecords and ActivationRecords to .npz")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Optional tag appended to output filenames to prevent collisions")

    # ── Device ──
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    output_dir_plots = Path(args.output_dir_plots)
    output_dir_plots.mkdir(parents=True, exist_ok=True)
    output_dir_data = Path(args.output_dir_data)
    output_dir_data.mkdir(parents=True, exist_ok=True)

    run_tag    = f"_{args.run_tag}" if args.run_tag else ""
    hook_types = ["resid_post"]

    # ── Corpus ────────────────────────────────────────────────────────────────
    corpus_path = Path(args.corpus)
    corpus_tag  = corpus_path.stem
    if not corpus_path.exists():
        corpus_path = Path("corpus") / corpus_path
    if not corpus_path.exists():
        print(f"Corpus not found: {args.corpus}")
        return 1

    try:
        with open(corpus_path) as f:
            corpus = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading corpus file '{corpus_path}': {e}")
        return 1
    print(f"\nLoaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")
    print(f"  Corpus file:  {corpus_path.name}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model '{args.model}'...")
    try:
        model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    except (ValueError, RuntimeError) as e:
        print(f"Error loading model '{args.model}': {e}")
        return 1
    print(f"  Model ready on {cfg['device']}")
    print(f"  Layers: {model.cfg.n_layers}")

    # ── Extraction ────────────────────────────────────────────────────────────
    print(f"\nExtracting activations across corpus...")
    activation_dict = extract_corpus(
        model, corpus, hook_types,
        model_name=args.model,
        device=cfg["device"],
        category_filter=args.category,
    )

    act_records = activation_dict["resid_post"]
    print(f"  resid_post: {len(act_records)} prompts")

    # ── Computation ───────────────────────────────────────────────────────────
    print(f"\nComputing mechanics...")
    mech_records = _run_mechanics_corpus(act_records)
    print(f"\n  Total MechanicsRecords: {len(mech_records)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.save_data:
        mech_data_dir = output_dir_data / "mechanics"
        mech_data_dir.mkdir(parents=True, exist_ok=True)
        data_path = (mech_data_dir /
                     f"mechanics_records_{args.model}_{corpus_tag}{run_tag}.npz")
        save_mechanics_records(mech_records, data_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\nGenerating plots in {output_dir_plots}/...")

        plot_mechanics_overview(
            mech_records, output_dir_plots,
            corpus_tag, run_tag, model_name=args.model,
        )

        categories = sorted(set(r.category for r in mech_records if r.category))
        for cat in categories:
            plot_mechanics_category(
                mech_records, cat, output_dir_plots,
                corpus_tag, run_tag, model_name=args.model,
            )

    print(f"\nDone. Results in {output_dir_plots}/\n")


if __name__ == "__main__":
    main()
