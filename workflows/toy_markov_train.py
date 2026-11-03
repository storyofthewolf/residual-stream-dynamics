"""toy_markov_train.py — Phase 1: train a 2-layer attention-only transformer on
in-context Markov chains and record the staged learning (decision log P1–P7).

Outputs:
    data/toy/{run}/history.npz            metrics every --eval-every steps
    data/toy/{run}/ckpt/step*.pt          checkpoints (git-ignored)
    figures/workflows/toy_markov_train/   stage plot + final attention patterns

Usage:
    python workflows/toy_markov_train.py
    python workflows/toy_markov_train.py --steps 5000 --run-tag quick
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "plotting"):
    sys.path.insert(0, str(_PROJECT_ROOT / sub))

import torch

from toy_training import ToyConfig, train, save_history, sample_markov_batch
from toy_plots import plot_training_stages, plot_attention_patterns

FIG_DIR = _PROJECT_ROOT / "figures" / "workflows" / "toy_markov_train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(ToyConfig):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    ap.add_argument("--run-tag", default="")
    args = ap.parse_args()
    cfg = ToyConfig(**{f.name: getattr(args, f.name) for f in fields(ToyConfig)})

    run = f"k{cfg.k}_L{cfg.length}_l{cfg.n_layers}h{cfg.n_heads}d{cfg.d_model}_s{cfg.seed}"
    run += f"_{args.run_tag}" if args.run_tag else ""
    out = _PROJECT_ROOT / "data" / "toy" / run
    print(f"Run {run}\n  {cfg}")

    t0 = time.time()
    model, hist = train(cfg, ckpt_dir=out / "ckpt")
    print(f"Trained in {time.time() - t0:.0f}s")
    save_history(hist, out / "history.npz")

    plot_training_stages(hist, f"2-layer attn-only transformer, in-context Markov (k={cfg.k}, L={cfg.length})",
                         save_path=FIG_DIR / f"stages_{run}.png")
    x = sample_markov_batch(1, cfg.length, cfg.k, cfg.alpha, torch.Generator().manual_seed(7))
    with torch.no_grad():
        _, cache = model.run_with_cache(x)
    pats = [cache[f"blocks.{l}.attn.hook_pattern"][0, 0].numpy() for l in range(cfg.n_layers)]
    plot_attention_patterns(pats, x[0].numpy(), f"Final attention patterns ({run})",
                            save_path=FIG_DIR / f"attention_{run}.png")
    print(f"Saved → {out}  and  {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
