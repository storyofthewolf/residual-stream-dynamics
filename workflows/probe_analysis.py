"""workflows/probe_analysis.py — Linear probe analysis over a corpus.

Asks whether the corpus's base/contrast distinction is linearly decodable from
the residual stream at each layer. Complements the entropy workflows: entropy is
a scalar summary that discards direction, so a property carried as a direction
in activation space can be invisible to it and obvious here.

Pipeline:
    extraction.extract_corpus()             -> dict[hook_type, list[ActivationRecord]]
    probe_compute.compute_probe()           -> ProbeRecord   (grouped CV + null)
    probe_compute.compute_probe_generalization() -> ProbeRecord (leave-one-group-out)
    probe_plots.plot_*()                    -> figures

Read the generalization figure, not just the accuracy figure. Within-distribution
accuracy conflates "the model represents this" with "the classifier memorized
these words"; holding out a whole category sub-factor separates them.

Usage:
    python workflows/probe_analysis.py --corpus corpus/corpus_moral.json
    python workflows/probe_analysis.py --corpus corpus/corpus_moral.json --model gpt2-medium
    python workflows/probe_analysis.py --corpus corpus/corpus_moral.json --generalize-by foundation style
    python workflows/probe_analysis.py --load-data data/activation_records_....npz
    python workflows/probe_analysis.py --corpus corpus/corpus_moral.json --save-data
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

from extraction import extract_corpus, save_activation_records, load_activation_records
from probe_compute import (
    compute_probe,
    compute_probe_generalization,
    save_probe_records,
    CATEGORY_FACTORS,
)
from probe_plots import plot_probe_accuracy, plot_probe_generalization


def main():
    parser = argparse.ArgumentParser(
        description="Linear probe analysis: is base vs contrast linearly decodable?"
    )

    parser.add_argument("--corpus", type=str,
                        default=str(_PROJECT_ROOT / "corpus" / "corpus_moral.json"),
                        help="Path to corpus JSON")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in utils/model_loader.py MODEL_CONFIGS)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")

    parser.add_argument("--generalize-by", type=str, nargs="+", default=["foundation", "style"],
                        choices=sorted(CATEGORY_FACTORS),
                        help="Category sub-factors to hold out, one figure each. "
                             "Requires the foundation_level_style category convention.")
    parser.add_argument("--no-generalize", action="store_true",
                        help="Skip the leave-one-group-out probes")
    parser.add_argument("--C", type=float, default=0.05,
                        help="Inverse L2 regularization strength (default 0.05)")
    parser.add_argument("--n-perm", type=int, default=200,
                        help="Permutation-null resamples per layer (default 200)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Grouped CV folds (default 5)")

    parser.add_argument("--load-data", type=str, default=None,
                        help="Skip extraction; load ActivationRecords from this .npz")
    parser.add_argument("--output-dir-plots", type=str,
                        default=str(_PROJECT_ROOT / "figures" / "workflows" / "probe_analysis"),
                        help="Directory for saved plots")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--output-dir-data", type=str,
                        default=str(_PROJECT_ROOT / "data" / "probe"),
                        help="Directory for saved data")
    parser.add_argument("--save-data", action="store_true",
                        help="Save ProbeRecords and ActivationRecords to .npz")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Optional tag appended to output filenames")

    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: cuda, cpu, or mps. Default None = auto-detect.")
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16"],
                        help="Model dtype. Default None = float32.")

    args = parser.parse_args()

    run_tag = f"_{args.run_tag}" if args.run_tag else ""
    output_dir_plots = Path(args.output_dir_plots)
    output_dir_data = Path(args.output_dir_data)

    # ── Obtain ActivationRecords, either from disk or by extraction ──
    if args.load_data:
        data_path = Path(args.load_data)
        if not data_path.exists():
            print(f"Data file not found: {data_path}")
            return 1
        # ActivationRecord files are named
        #   activation_records_<model>_<corpus_tag>_<hook>[_<run_tag>].npz
        # Using the whole stem as the corpus tag produces unreadable output
        # filenames, so recover just the corpus portion when the name follows
        # that convention; otherwise fall back to the full stem.
        corpus_tag = data_path.stem
        _prefix = f"activation_records_{args.model}_"
        if corpus_tag.startswith(_prefix):
            corpus_tag = corpus_tag[len(_prefix):]
            for _hook in ("_resid_post", "_resid_mid", "_resid_pre",
                          "_attn_out", "_mlp_out"):
                if _hook in corpus_tag:
                    corpus_tag = corpus_tag.split(_hook)[0]
                    break
        print(f"\nLoading ActivationRecords from {data_path}")
        try:
            act_records = load_activation_records(data_path)
        except Exception as e:
            print(f"Error reading '{data_path}': {e}")
            return 1
        print(f"  Loaded {len(act_records)} records")
    else:
        corpus_path = Path(args.corpus)
        corpus_tag = corpus_path.stem
        if not corpus_path.exists():
            corpus_path = Path("corpus") / corpus_path
        if not corpus_path.exists():
            print(f"Corpus not found: {args.corpus}")
            return 1
        try:
            with open(corpus_path) as f:
                corpus = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error reading corpus file '{corpus_path}': {e}")
            return 1

        print(f"\nLoaded corpus: {len(corpus)} prompts ({len(corpus) // 2} pairs)")
        print(f"  Corpus file:  {corpus_path.name}")

        from model_loader import load_model_and_sae, MODEL_CONFIGS
        if args.model not in MODEL_CONFIGS:
            print(f"Unknown model '{args.model}'. "
                  f"Available: {', '.join(sorted(MODEL_CONFIGS))}")
            return 1

        model, _ = load_model_and_sae(args.model, device=args.device, dtype=args.dtype)
        print(f"\nExtracting activations across corpus...")
        activation_dict = extract_corpus(
            model, corpus, ["resid_post"],
            category_filter=args.category,
        )
        act_records = activation_dict["resid_post"]

        if args.save_data:
            output_dir_data.mkdir(parents=True, exist_ok=True)
            act_path = (output_dir_data /
                        f"activation_records_{args.model}_{corpus_tag}_resid_post{run_tag}.npz")
            save_activation_records(act_records, act_path)

    if not act_records:
        print("No activation records to probe.")
        return 1

    # ── Probe ──
    probe_records = []

    print(f"\nFitting linear probe (grouped CV by pair_id)...")
    try:
        rec = compute_probe(
            act_records, model_name=args.model, corpus_tag=corpus_tag,
            C=args.C, n_splits=args.n_splits, n_perm=args.n_perm,
        )
    except (ValueError, ImportError) as e:
        print(f"Probe failed: {e}")
        return 1
    probe_records.append(rec)
    print(f"  layer-0 baseline {rec.accuracy[0]:.3f}   "
          f"peak {rec.accuracy.max():.3f} @ L{int(rec.accuracy.argmax())}   "
          f"rise {rec.accuracy.max() - rec.accuracy[0]:+.3f}")

    gen_records = []
    if not args.no_generalize:
        for by in args.generalize_by:
            print(f"\nLeave-one-{by}-out generalization...")
            try:
                grec = compute_probe_generalization(
                    act_records, by, model_name=args.model,
                    corpus_tag=corpus_tag, C=args.C,
                )
            except ValueError as e:
                print(f"  Skipped ({e})")
                continue
            gen_records.append(grec)
            probe_records.append(grec)
            print(f"  peak mean {grec.accuracy.max():.3f} @ L{int(grec.accuracy.argmax())}")

    # ── Save ──
    if args.save_data:
        output_dir_data.mkdir(parents=True, exist_ok=True)
        data_path = output_dir_data / f"probe_records_{args.model}_{corpus_tag}{run_tag}.npz"
        save_probe_records(probe_records, data_path)

    # ── Plot ──
    if not args.no_plots:
        print(f"\nGenerating plots in {output_dir_plots}/...")
        plot_probe_accuracy(rec, output_dir_plots, corpus_tag, run_tag)
        for grec in gen_records:
            plot_probe_generalization(grec, output_dir_plots, corpus_tag, run_tag)

    print(f"\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
