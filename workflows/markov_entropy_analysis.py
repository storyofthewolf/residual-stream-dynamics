"""markov_entropy_analysis.py — Residual / output entropy vs. Markov source entropy rate.

For each prompt length, loads corpus/markov_entropy_L{L}.json, extracts
resid_post, computes energy-norm (α=1) residual entropy and logit-lens entropy
at every (layer, position) with the existing entropy_compute functions, plus
final-layer output statistics (markov_compute), summarizes over second-half
positions, and runs the decision-log M12 statistics.

Usage:
    python workflows/markov_entropy_analysis.py --model gpt2-small --lengths 64 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "utils", "plotting", "corpus"):
    sys.path.insert(0, str(_PROJECT_ROOT / sub))

import numpy as np
import torch

from model_loader import load_model_and_sae
from extraction import extract_corpus
from entropy_compute import (compute_residual_stream_entropy, compute_logit_lens_entropy,
                             save_entropy_records)
from markov_compute import (output_distribution_stats, summarize_prompt, second_half_positions,
                            spearman_with_pair_ci, level_means_with_ci, save_markov_summary)
from markov_plots import plot_rho_vs_layer, plot_resid_by_level, plot_output_vs_rate
from markov_gen import ALPHABET

DATA_DIR = _PROJECT_ROOT / "data" / "markov"
FIG_DIR  = _PROJECT_ROOT / "figures" / "workflows" / "markov_entropy_analysis"


def run_length(model, cfg, model_name, L):
    corpus_path = _PROJECT_ROOT / "corpus" / f"markov_entropy_L{L}.json"
    corpus = json.loads(corpus_path.read_text())
    print(f"\n── L={L}: {len(corpus)} prompts ({corpus_path.name})")

    alphabet_ids = np.array([model.to_single_token(" " + w) for w in ALPHABET])
    W_U, ln_final = model.W_U.detach(), model.ln_final

    t0 = time.time()
    acts = extract_corpus(model, corpus, ["resid_post"], model_name=model_name,
                          device=cfg["device"])["resid_post"]
    print(f"   extracted in {time.time() - t0:.0f}s")

    t0 = time.time()
    rows, ent_records = [], []
    for e, rec in zip(corpus, acts):
        ids = model.to_tokens(e["prompt"], prepend_bos=True)[0].cpu().numpy()
        assert len(ids) == L + 1 and rec.seq_len == L + 1
        rs = compute_residual_stream_entropy(rec, [1.0], ["energy"])[0]
        ll = compute_logit_lens_entropy(rec, [1.0], W_U, ln_final)[0]
        out = output_distribution_stats(rec, W_U, ln_final, alphabet_ids, ids)
        s = summarize_prompt(rs.surface, out)
        s["ll"] = ll.surface[:, second_half_positions(rec.seq_len)].mean(axis=1)
        rows.append(s)
        ent_records += [rs, ll]
    print(f"   entropy computed in {time.time() - t0:.0f}s")

    H     = np.array([e["entropy_rate_target"] for e in corpus])
    roles = np.array([e["role"] for e in corpus])
    pids  = np.array([e["pair_id"] for e in corpus])
    resid = np.stack([r["resid"] for r in rows])
    llyr  = np.stack([r["ll"] for r in rows])
    outs  = {k: np.array([r[k] for r in rows]) for k in ("H_full", "H_alpha", "mass", "xent")}

    tag = f"{model_name}_markov_L{L}"
    save_entropy_records(ent_records, DATA_DIR / f"entropy_records_{tag}.npz")

    stats = {}
    for name, Y in [("resid", resid), ("loglens", llyr)] + [(k, v) for k, v in outs.items()]:
        for role in ("base", "contrast"):
            stats[(name, role)] = spearman_with_pair_ci(H, Y, roles, pids, role)
    lvl = {(k, role): level_means_with_ci(H, outs[k], roles, pids, role)
           for k in outs for role in ("base", "contrast")}
    lvl_resid = {role: level_means_with_ci(H, resid, roles, pids, role) for role in ("base", "contrast")}

    save_markov_summary(
        DATA_DIR / f"markov_summary_{tag}.npz",
        model_name=model_name, length=L, H=H, roles=roles, pair_ids=pids,
        resid=resid, loglens=llyr, **outs,
        **{f"rho_{n}_{r}": np.stack([v["rho"], v["ci_lo"], v["ci_hi"]]) for (n, r), v in stats.items()},
    )

    # ── print ──
    nl = resid.shape[1]
    print(f"\n   Spearman ρ(H, residual entropy) by layer [95% CI]")
    for j in range(nl):
        b, c = stats[("resid", "base")], stats[("resid", "contrast")]
        print(f"     L{j:2d}  Markov {b['rho'][j]:+.3f} [{b['ci_lo'][j]:+.3f},{b['ci_hi'][j]:+.3f}]"
              f"   Shuffled {c['rho'][j]:+.3f} [{c['ci_lo'][j]:+.3f},{c['ci_hi'][j]:+.3f}]")
    print(f"\n   Spearman ρ(H, logit-lens entropy) by layer, Markov / Shuffled")
    print("     " + "  ".join(f"L{j}:{stats[('loglens','base')]['rho'][j]:+.2f}/"
                               f"{stats[('loglens','contrast')]['rho'][j]:+.2f}" for j in range(nl)))
    print(f"\n   Final-layer output (second-half mean) vs H, Markov | Shuffled")
    print(f"     {'H':>5} {'H_alpha':>15} {'xent':>15} {'mass':>15} {'H_full':>15}")
    for i, h in enumerate(lvl[("H_alpha", "base")]["levels"]):
        cells = []
        for k in ("H_alpha", "xent", "mass", "H_full"):
            cells.append(f"{lvl[(k,'base')]['mean'][i]:6.2f}|{lvl[(k,'contrast')]['mean'][i]:6.2f}")
        print(f"     {h:5.2f} " + " ".join(f"{c:>15}" for c in cells))
    for k in ("H_alpha", "xent", "mass", "H_full"):
        b, c = stats[(k, "base")], stats[(k, "contrast")]
        print(f"   ρ(H, {k:7s}) Markov {b['rho'][0]:+.3f} [{b['ci_lo'][0]:+.3f},{b['ci_hi'][0]:+.3f}]"
              f"  Shuffled {c['rho'][0]:+.3f} [{c['ci_lo'][0]:+.3f},{c['ci_hi'][0]:+.3f}]")

    plot_resid_by_level(lvl_resid["base"]["levels"], lvl_resid["base"]["mean"],
                        lvl_resid["contrast"]["mean"], model_name, f"L = {L}",
                        save_path=FIG_DIR / f"markov_resid_by_level_{tag}.png")
    rho_panel = {"title": f"L = {L}", "base": stats[("resid", "base")],
                 "shuf": stats[("resid", "contrast")]}
    out_panel = {"title": f"L = {L}"}
    for k in ("H_alpha", "xent", "mass"):
        out_panel[k] = {"base": lvl[(k, "base")], "shuf": lvl[(k, "contrast")]}
    return rho_panel, out_panel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2-small")
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    except (ValueError, RuntimeError) as e:
        print(f"Error loading model '{args.model}': {e}")
        return 1

    rho_panels, out_panels = [], []
    for L in args.lengths:
        r, o = run_length(model, cfg, args.model, L)
        rho_panels.append(r)
        out_panels.append(o)

    lt = "_".join(str(L) for L in args.lengths)
    plot_rho_vs_layer(rho_panels, args.model,
                      save_path=FIG_DIR / f"markov_rho_vs_layer_{args.model}_L{lt}.png")
    plot_output_vs_rate(out_panels, args.model,
                        save_path=FIG_DIR / f"markov_output_vs_rate_{args.model}_L{lt}.png")
    print(f"\nFigures → {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
