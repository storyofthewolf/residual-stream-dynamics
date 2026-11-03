"""ablation_controls_analysis.py — Batch 1 controls: per-model compute driver.

Stages (run in order; each saves to data/ablation_controls/):

    reproduce     Saved .npz → notebook "~20% at 99% EV" quantities, for all
                  models with a saved file. No model needed.
    margins       Clean logit gap / top-1 prob, joined to saved ablation flips.
    posthoc       Recompute existing post-hoc ablation on fresh activations and
                  compare to saved records; W_U basis under preLN and postLN at
                  all layers; random-subspace null (N draws) at the layer subset;
                  k = d_model sanity checks.
    intervention  Recompute saved intervention at the layer subset; random-
                  subspace null (N draws); k = d_model sanity checks.
    seedcheck     Re-run a slice of the posthoc and intervention nulls and
                  require bitwise equality with the saved arrays.

Examples:
    python workflows/ablation_controls_analysis.py --stages reproduce
    python workflows/ablation_controls_analysis.py --model gpt2-small \\
        --stages margins posthoc intervention seedcheck
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "utils", "plotting"):
    sys.path.insert(0, str(_PROJECT_ROOT / sub))

import numpy as np
import torch

DATA_ABL  = _PROJECT_ROOT / "data" / "ablation"
DATA_OUT  = _PROJECT_ROOT / "data" / "ablation_controls"
CORPUS    = _PROJECT_ROOT / "corpus" / "base_vs_contrast_n50.json"
TAG       = "base_vs_contrast_n50"
ALL_MODELS = ["gpt2-small", "gpt2-medium", "gpt2-large", "gpt2-xl",
              "pythia-160m", "pythia-1b", "pythia-2.8b", "pythia-6.9b"]


def _saved_path(model: str) -> Path:
    return DATA_ABL / f"ablation_records_{model}_{TAG}.npz"


# ----------------------------------------------------------------------------
def stage_reproduce(corpus):
    from ablation_controls_compute import saved_flip_table, published_quantities, save_controls
    from npz_utils import load_ablation_npz, get_ablation_records

    out = {}
    for m in ALL_MODELS:
        p = _saved_path(m)
        if not p.exists():
            print(f"[skip] {p.name}")
            continue
        tab = saved_flip_table(p, corpus)
        pq  = published_quantities(tab)

        # (a) notebook code path, copied from notebook cell 15
        abl = load_ablation_npz(p)
        ks  = sorted(int(k) for k in np.unique(abl["ks"]) if k > 0)
        lyrs = sorted(int(l) for l in np.unique(abl["intervention_lyrs"]) if l >= 0)
        n_total = max(lyrs) + 1
        mid = [l for l in lyrs if 0.3 <= l / n_total <= 0.6] or lyrs
        nb_ph, nb_iv = [], []
        for k in ks:
            _, _, bt, _, _ = get_ablation_records(abl, role="base", ablation_type="posthoc", k=k)
            _, _, ct, _, _ = get_ablation_records(abl, role="contrast", ablation_type="posthoc", k=k)
            nb_ph.append(np.mean(np.concatenate([[x[-1].astype(float) for x in bt],
                                                 [x[-1].astype(float) for x in ct]])))
            vals = []
            for L in mid:
                _, _, bi, _, _ = get_ablation_records(abl, role="base", ablation_type="intervention", k=k, intervention_lyr=L)
                _, _, ci, _, _ = get_ablation_records(abl, role="contrast", ablation_type="intervention", k=k, intervention_lyr=L)
                vals.append(np.mean(np.concatenate([[x[0].astype(float) for x in bi],
                                                    [x[0].astype(float) for x in ci]])))
            nb_iv.append(np.mean(vals))
        nb_ph_flip = 1 - np.array(nb_ph)
        nb_iv_flip = 1 - np.array(nb_iv)
        same = (np.array_equal(ks, tab["ks"]) and np.allclose(nb_ph_flip, pq["posthoc_final"], atol=1e-12)
                and np.allclose(nb_iv_flip, pq["intervention_mid"], atol=1e-12))
        print(f"{m:12s} ks={ks}")
        print(f"   posthoc final flip : " + " ".join(f"{v:.3f}" for v in pq["posthoc_final"]))
        print(f"   interv. mid flip   : " + " ".join(f"{v:.3f}" for v in pq["intervention_mid"]))
        print(f"   notebook path identical: {same}")
        out[m] = dict(ks=tab["ks"], posthoc_final=pq["posthoc_final"],
                      intervention_mid=pq["intervention_mid"], notebook_identical=same)

    ph99 = np.array([v["posthoc_final"][6] for v in out.values()])
    iv99 = np.array([v["intervention_mid"][6] for v in out.values()])
    print(f"\nEV=0.99, mean over {len(out)} models: posthoc {ph99.mean():.3f}, "
          f"intervention-mid {iv99.mean():.3f}")
    ok_c = (0.15 <= ph99.mean() <= 0.25) and (0.15 <= iv99.mean() <= 0.25)
    ok_a = all(v["notebook_identical"] for v in out.values())
    print(f"Criterion (a) notebook path identical: {ok_a}")
    print(f"Criterion (c) both in [0.15, 0.25]:    {ok_c}")
    save_controls(DATA_OUT / "reproduction.npz",
                  models=np.array(list(out), dtype=object),
                  ks=np.array([v["ks"] for v in out.values()], dtype=object),
                  posthoc_final=np.array([v["posthoc_final"] for v in out.values()], dtype=object),
                  intervention_mid=np.array([v["intervention_mid"] for v in out.values()], dtype=object),
                  criterion_a=ok_a, criterion_c=ok_c,
                  mean_posthoc_99=ph99.mean(), mean_intervention_99=iv99.mean())
    return 0 if (ok_a and ok_c) else 2


# ----------------------------------------------------------------------------
def _load(model_name, device):
    from model_loader import load_model_and_sae
    model, _, cfg = load_model_and_sae(model_name, load_sae=False, device=device)
    model.eval()
    return model, cfg


def stage_margins(model, model_name, corpus, table):
    from ablation_controls_compute import clean_output_margins, save_controls
    prompts = list(table["prompts"])
    mg = clean_output_margins(model, prompts)
    path = DATA_OUT / f"margins_{model_name}_{TAG}.npz"
    save_controls(path, model_name=model_name, prompts=table["prompts"], roles=table["roles"],
                  categories=table["categories"], pair_ids=table["pair_ids"],
                  logit_gap=mg["logit_gap"], p_top1=mg["p_top1"], top1_token=mg["top1_token"],
                  ks=table["ks"], saved_posthoc_flip=table["posthoc_flip"],
                  saved_int_layers=table["int_layers"],
                  saved_intervention_flip=table["intervention_flip"],
                  saved_intervention_mask=table["intervention_mask"])
    print(f"  margins: gap median {np.median(mg['logit_gap']):.3f}; saved → {path.name}")
    return mg


def _null_bases(d, k, n, offset=0):
    from ablation_controls_compute import random_orthonormal_basis
    return torch.stack([random_orthonormal_basis(d, k, offset + i) for i in range(n)])


def stage_posthoc(model, model_name, corpus, table, R, n_null):
    from ablation_compute import compute_posthoc_ablation
    from ablation_controls_compute import posthoc_flips_batched, layer_subset, save_controls
    from math_utils import compute_wu_svd
    from extraction import ActivationRecord  # noqa: F401 (type only)

    W_U, ln_final = model.W_U.detach().float(), model.ln_final
    d, n_layers   = model.cfg.d_model, model.cfg.n_layers
    Vh  = compute_wu_svd(W_U).float().cpu()
    ks  = [int(k) for k in table["ks"]]
    subset = layer_subset(n_layers)
    Rt  = torch.from_numpy(R)                                   # [n, L, d]

    # (b) existing code on fresh activations vs saved records
    t0 = time.time()
    mism = 0
    for i, rec in enumerate(_ACT_RECORDS):
        for ar in compute_posthoc_ablation(rec, W_U, ln_final, Vh, ks):
            ki = ks.index(ar.k)
            mism += int(((~ar.top1_preserved) != table["posthoc_flip"][ki, :, i]).sum())
    print(f"  (b) posthoc recompute vs saved: {mism} mismatching (k, layer, prompt) cells "
          f"of {len(ks) * n_layers * len(_ACT_RECORDS)}  [{time.time() - t0:.1f}s]")

    convs = ["preLN", "postLN"]
    real_flip = np.zeros((2, len(ks), n_layers, R.shape[0]), dtype=bool)
    real_en   = np.zeros_like(real_flip, dtype=np.float64)
    for c, conv in enumerate(convs):
        for ki, k in enumerate(ks):
            Q = Vh[:k, :].T.contiguous()[None]
            for L in range(n_layers):
                f, e = posthoc_flips_batched(Rt[:, L, :], Q, W_U, ln_final, conv)
                real_flip[c, ki, L], real_en[c, ki, L] = f[0], e[0]
    own_vs_saved = int((real_flip[0] != table["posthoc_flip"]).sum())
    print(f"  batched preLN W_U vs saved: {own_vs_saved} mismatching cells")

    # sanity: k = d_model, real basis + every null draw
    Qd = Vh[:d, :].T.contiguous()[None]
    Qn = _null_bases(d, d, n_null)
    sanity = {}
    for conv in convs:
        sr = sum(int(posthoc_flips_batched(Rt[:, L, :], Qd, W_U, ln_final, conv)[0].sum()) for L in range(n_layers))
        sn = sum(int(posthoc_flips_batched(Rt[:, L, :], Qn, W_U, ln_final, conv)[0].sum()) for L in subset)
        sanity[conv] = (sr, sn)
        print(f"  sanity k=d_model {conv}: real flips={sr}, null flips={sn} over {n_null} draws")

    # null
    t0 = time.time()
    null_flip = np.zeros((2, len(ks), n_null, len(subset), R.shape[0]), dtype=bool)
    null_en   = np.zeros((2, len(ks), n_null, len(subset), R.shape[0]), dtype=np.float32)
    for ki, k in enumerate(ks):
        Qs = _null_bases(d, k, n_null)
        for c, conv in enumerate(convs):
            for li, L in enumerate(subset):
                f, e = posthoc_flips_batched(Rt[:, L, :], Qs, W_U, ln_final, conv)
                null_flip[c, ki, :, li], null_en[c, ki, :, li] = f, e
        print(f"    k={k:5d} done [{time.time() - t0:.0f}s]", flush=True)

    path = DATA_OUT / f"posthoc_null_{model_name}_{TAG}.npz"
    save_controls(path, model_name=model_name, ks=np.array(ks), conventions=np.array(convs),
                  layers_all=np.arange(n_layers), layer_subset=np.array(subset),
                  prompts=table["prompts"], roles=table["roles"], pair_ids=table["pair_ids"],
                  real_flip=real_flip, real_energy=real_en,
                  null_flip=null_flip, null_energy=null_en, n_null=n_null,
                  recompute_mismatch=mism, own_vs_saved_mismatch=own_vs_saved,
                  sanity_real_preLN=sanity["preLN"][0], sanity_null_preLN=sanity["preLN"][1],
                  sanity_real_postLN=sanity["postLN"][0], sanity_null_postLN=sanity["postLN"][1])
    print(f"  saved → {path.name}")


def stage_intervention(model, model_name, corpus, table, R, n_null, clean_top1):
    from ablation_compute import compute_intervention_ablation
    from ablation_controls_compute import intervention_flips_batched, layer_subset, save_controls
    from math_utils import compute_wu_svd

    W_U, ln_final = model.W_U.detach().float(), model.ln_final
    d, n_layers   = model.cfg.d_model, model.cfg.n_layers
    Vh  = compute_wu_svd(W_U).float().cpu()
    ks  = [int(k) for k in table["ks"]]
    subset = layer_subset(n_layers)
    prompts = list(table["prompts"])
    li_saved = {int(L): i for i, L in enumerate(table["int_layers"])}

    # (b) existing intervention code at the subset vs saved
    t0 = time.time()
    mism, tot = 0, 0
    for i, rec in enumerate(_ACT_RECORDS):
        for ar in compute_intervention_ablation(rec, model, W_U, ln_final, Vh, ks, subset):
            ki, li = ks.index(ar.k), li_saved[ar.intervention_layer]
            mism += int((not ar.top1_preserved[0]) != table["intervention_flip"][ki, li, i])
            tot += 1
    print(f"  (b) intervention recompute vs saved: {mism}/{tot} mismatching [{time.time() - t0:.1f}s]")

    # real W_U through the batched path (consistency with the null's code path)
    real_flip = np.zeros((len(ks), len(subset), len(prompts)), dtype=bool)
    for ki, k in enumerate(ks):
        Q = Vh[:k, :].T.contiguous()[None]
        for li, L in enumerate(subset):
            for i, p in enumerate(prompts):
                real_flip[ki, li, i] = intervention_flips_batched(model, p, L, Q, int(clean_top1[i]))[0]
    own_vs_saved = int(sum((real_flip[ki, li] != table["intervention_flip"][ki, li_saved[L]]).sum()
                           for ki in range(len(ks)) for li, L in enumerate(subset)))
    print(f"  batched W_U vs saved: {own_vs_saved} mismatching cells")

    # sanity k = d_model
    Qd = Vh[:d, :].T.contiguous()[None]
    Qn = _null_bases(d, d, n_null)
    sr = sn = 0
    for li, L in enumerate(subset):
        for i, p in enumerate(prompts):
            sr += int(intervention_flips_batched(model, p, L, Qd, int(clean_top1[i]))[0])
            sn += int(intervention_flips_batched(model, p, L, Qn, int(clean_top1[i])).sum())
    print(f"  sanity k=d_model intervention: real flips={sr}, null flips={sn} over {n_null} draws")

    t0 = time.time()
    null_flip = np.zeros((len(ks), n_null, len(subset), len(prompts)), dtype=bool)
    for ki, k in enumerate(ks):
        Qs = _null_bases(d, k, n_null)
        for li, L in enumerate(subset):
            for i, p in enumerate(prompts):
                null_flip[ki, :, li, i] = intervention_flips_batched(model, p, L, Qs, int(clean_top1[i]))
        print(f"    k={k:5d} done [{time.time() - t0:.0f}s]", flush=True)

    path = DATA_OUT / f"intervention_null_{model_name}_{TAG}.npz"
    save_controls(path, model_name=model_name, ks=np.array(ks), layer_subset=np.array(subset),
                  prompts=table["prompts"], roles=table["roles"], pair_ids=table["pair_ids"],
                  real_flip=real_flip, null_flip=null_flip, n_null=n_null,
                  recompute_mismatch=mism, own_vs_saved_mismatch=own_vs_saved,
                  sanity_real=sr, sanity_null=sn)
    print(f"  saved → {path.name}")


def stage_seedcheck(model, model_name, table, R, clean_top1, n_post=100, n_int=5):
    from ablation_controls_compute import (posthoc_flips_batched, intervention_flips_batched,
                                           layer_subset, load_controls)
    W_U, ln_final = model.W_U.detach().float(), model.ln_final
    d = model.cfg.d_model
    subset = layer_subset(model.cfg.n_layers)
    ph = load_controls(DATA_OUT / f"posthoc_null_{model_name}_{TAG}.npz")
    Rt = torch.from_numpy(R)
    ok = True
    for ki, k in enumerate(ph["ks"]):
        Qs = _null_bases(d, int(k), n_post)
        for c, conv in enumerate(["preLN", "postLN"]):
            for li, L in enumerate(subset):
                f, e = posthoc_flips_batched(Rt[:, L, :], Qs, W_U, ln_final, conv)
                ok &= np.array_equal(f, ph["null_flip"][c, ki, :n_post, li])
                ok &= np.array_equal(e.astype(np.float32), ph["null_energy"][c, ki, :n_post, li])
    print(f"  seedcheck posthoc (all k, both conventions, {n_post} draws): identical={ok}")
    ivp = DATA_OUT / f"intervention_null_{model_name}_{TAG}.npz"
    if ivp.exists():
        iv = load_controls(ivp)
        ok2 = True
        ki = 6
        Qs = _null_bases(d, int(iv["ks"][ki]), n_int)
        for li, L in enumerate(subset):
            for i, p in enumerate(table["prompts"]):
                f = intervention_flips_batched(model, p, L, Qs, int(clean_top1[i]))
                ok2 &= np.array_equal(f, iv["null_flip"][ki, :n_int, li, i])
        print(f"  seedcheck intervention (k index 6, {n_int} draws, all subset layers): identical={ok2}")
        ok &= ok2
    return 0 if ok else 3


# ----------------------------------------------------------------------------
_ACT_RECORDS: list = []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt2-small")
    ap.add_argument("--stages", nargs="+", default=["reproduce"],
                    choices=["reproduce", "margins", "posthoc", "intervention", "seedcheck"])
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--n-null-int", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    try:
        corpus = json.loads(CORPUS.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading corpus: {e}")
        return 1

    rc = 0
    if "reproduce" in args.stages:
        rc = stage_reproduce(corpus)
        if rc:
            print("\nReproduction criteria FAILED — stopping.")
            return rc

    model_stages = [s for s in args.stages if s != "reproduce"]
    if not model_stages:
        return rc

    from ablation_controls_compute import saved_flip_table, final_token_residuals, clean_output_margins
    from extraction import extract_corpus
    sp = _saved_path(args.model)
    if not sp.exists():
        print(f"No saved ablation file for {args.model}: {sp}")
        return 1
    table = saved_flip_table(sp, corpus)
    model, cfg = _load(args.model, args.device)

    by_prompt = {e["prompt"]: e for e in corpus}
    ordered   = [by_prompt[p] for p in table["prompts"]]
    t0 = time.time()
    acts = extract_corpus(model, ordered, ["resid_post"], model_name=args.model, device=cfg["device"])
    _ACT_RECORDS[:] = acts["resid_post"]
    R = final_token_residuals(_ACT_RECORDS)
    print(f"  extracted {R.shape} in {time.time() - t0:.1f}s")

    mg = stage_margins(model, args.model, corpus, table) if "margins" in model_stages \
        else clean_output_margins(model, list(table["prompts"]))
    if "posthoc" in model_stages:
        stage_posthoc(model, args.model, corpus, table, R, args.n_null)
    if "intervention" in model_stages:
        stage_intervention(model, args.model, corpus, table, R, args.n_null_int, mg["top1_token"])
    if "seedcheck" in model_stages:
        rc = stage_seedcheck(model, args.model, table, R, mg["top1_token"])
    return rc


if __name__ == "__main__":
    sys.exit(main() or 0)
