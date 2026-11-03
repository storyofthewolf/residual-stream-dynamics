"""ablation_controls_compute.py — Batch 1 statistical controls on ablation findings.

Consumes ActivationRecords (extraction.py) and saved AblationRecord .npz files
(ablation_compute.py). Produces plain result dicts persisted as .npz under
data/ablation_controls/ for ablation_controls_plots.py and the Batch 1 summary.

This module does NOT modify or subclass AblationRecord. The control results
are multi-draw arrays ([k, draw, layer, prompt]) that do not fit the
one-record-per-(prompt, k) shape, so they are stored as array dicts with a
save/load pair here.

Controls:
    reproduce_published_quantities  — notebook fig. "top-1 vs k" numbers from saved .npz
    clean_output_margins            — final logit gap (top1 − top2) and top-1 probability
    posthoc_flips_batched           — post-hoc ablation (pre-LN or post-LN) for many bases
    intervention_flips_batched      — forward-pass ablation for many bases in one batch
    random_orthonormal_basis        — seeded QR-of-Gaussian rank-k basis
    null_comparison / tercile_rates / margin_regression / verdicts

Forward passes: only intervention_flips_batched and clean_output_margins run
the model, and only on a live model explicitly passed in (same allowance as
the intervention path of ablation_compute.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from stats_utils import (
    DEFAULT_B,
    DEFAULT_SEED,
    logistic_irls,
    pair_bootstrap,
    pair_bootstrap_indices,
    pair_bootstrap_paired_diff,
    pair_bootstrap_rate,
    percentile_ci,
)

PRIMARY_EV_INDEX = 6          # 7th of 9 EV-threshold ks = EV 0.99 (decision log #4)
HIGH_EV_INDICES  = (5, 6, 7)  # 95%, 99%, 99.9% (decision log #21)
NULL_SEED        = DEFAULT_SEED


# ============================================================================
# BASIC HELPERS
# ============================================================================

def layer_subset(n_layers: int, n: int = 5) -> list[int]:
    """unique(round(linspace(0, n_layers-1, n))) — decision log #11."""
    return sorted(set(int(round(x)) for x in np.linspace(0, n_layers - 1, n)))


def random_orthonormal_basis(d_model: int, k: int, draw: int,
                             seed: int = NULL_SEED) -> torch.Tensor:
    """Seeded rank-k orthonormal basis [d_model, k]: QR of a Gaussian (decision log #9)."""
    rng = np.random.default_rng([seed, d_model, k, draw])
    G   = rng.standard_normal((d_model, k))
    Q, R = np.linalg.qr(G)
    Q   = Q * np.sign(np.diag(R))[None, :]       # unique QR (Haar-distributed)
    return torch.from_numpy(Q.astype(np.float32))


def pair_ids_for_prompts(prompts, corpus: list[dict]) -> np.ndarray:
    """Join pair_id from the corpus JSON on exact prompt text (decision log #2)."""
    lookup = {e["prompt"]: e["pair_id"] for e in corpus}
    missing = [p for p in prompts if p not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} prompts not found in corpus, e.g. {missing[0]!r}")
    return np.array([lookup[p] for p in prompts], dtype=np.int64)


def final_token_residuals(act_records: list) -> np.ndarray:
    """Stack final-token residuals → [n_prompts, n_layers, d_model] float32."""
    return np.stack([r.activations[:, -1, :] for r in act_records]).astype(np.float32)


# ============================================================================
# REPRODUCTION (task 2)
# ============================================================================

def saved_flip_table(npz_path, corpus: list[dict]) -> dict:
    """
    Read a saved ablation .npz into dense flip arrays, joined to pair IDs.

    Returns dict with:
        prompts, roles, categories, pair_ids     [n_prompts]  (posthoc order)
        ks                                        [n_k]        sorted
        posthoc_flip      bool [n_k, n_layers, n_prompts]
        int_layers                                [n_int]
        intervention_flip bool [n_k, n_int, n_prompts]  (NaN-free; missing → False with mask)
        intervention_mask bool [n_k, n_int, n_prompts]  True where a record exists
    """
    # Read arrays once. ablation_compute.load_ablation_records re-reads every
    # array from the zip per record and keeps slice views of each copy alive,
    # which needs >23 GB for the gpt2-xl file (decision log, anomaly A1).
    with np.load(npz_path, allow_pickle=True) as d:
        a_type  = d["ablation_types"].astype(str)
        a_k     = d["ks"].astype(int)
        a_lyr   = d["intervention_lyrs"].astype(int)
        a_len   = d["arr_lens"].astype(int)
        a_top1  = d["top1_preserved"].astype(bool)
        a_prm   = d["prompts"].astype(str)
        a_role  = d["roles"].astype(str)
        a_cat   = d["categories"].astype(str)
        model_name = str(d["model_names"][0])

    ph = np.flatnonzero(a_type == "posthoc")
    iv = np.flatnonzero(a_type == "intervention")
    ks = sorted(set(a_k.tolist()))
    prompts = list(dict.fromkeys(a_prm[ph].tolist()))
    p_idx = {p: i for i, p in enumerate(prompts)}
    k_idx = {k: i for i, k in enumerate(ks)}
    n_layers = int(a_len[ph[0]])

    roles = np.empty(len(prompts), dtype=object)
    cats  = np.empty(len(prompts), dtype=object)
    ph_flip = np.zeros((len(ks), n_layers, len(prompts)), dtype=bool)
    for r in ph:
        i = p_idx[a_prm[r]]
        roles[i], cats[i] = a_role[r], a_cat[r]
        ph_flip[k_idx[a_k[r]], :, i] = ~a_top1[r, :n_layers]

    int_layers = sorted(set(a_lyr[iv].tolist()))
    l_idx = {L: i for i, L in enumerate(int_layers)}
    iv_flip = np.zeros((len(ks), len(int_layers), len(prompts)), dtype=bool)
    iv_mask = np.zeros_like(iv_flip)
    for r in iv:
        j = (k_idx[a_k[r]], l_idx[a_lyr[r]], p_idx[a_prm[r]])
        iv_flip[j] = not a_top1[r, 0]
        iv_mask[j] = True

    return dict(
        prompts=np.array(prompts, dtype=object), roles=roles, categories=cats,
        pair_ids=pair_ids_for_prompts(prompts, corpus),
        ks=np.array(ks), posthoc_flip=ph_flip,
        int_layers=np.array(int_layers), intervention_flip=iv_flip,
        intervention_mask=iv_mask, model_name=model_name,
    )


def published_quantities(table: dict, frac_lo: float = 0.3, frac_hi: float = 0.6) -> dict:
    """
    The two numbers behind the notebook's "~20% at 99% EV" figure (decision log #6c).

    posthoc: flip rate at the final layer, base+contrast pooled.
    intervention: flip rate averaged over intervention layers with
                  fractional depth L/n_layers in [frac_lo, frac_hi]
                  (matches notebook FRAC_LO/FRAC_HI and its mean-of-layer-means).
    Returned per k (index aligned with table['ks']).
    """
    ph = table["posthoc_flip"][:, -1, :].mean(axis=1)
    lyrs = table["int_layers"]
    n_total = int(lyrs.max()) + 1
    mid = [i for i, L in enumerate(lyrs) if frac_lo <= L / n_total <= frac_hi] or list(range(len(lyrs)))
    iv = np.array([
        np.mean([table["intervention_flip"][ki, li, :][table["intervention_mask"][ki, li, :]].mean()
                 for li in mid])
        for ki in range(len(table["ks"]))
    ])
    return {"ks": table["ks"], "posthoc_final": ph, "intervention_mid": iv,
            "mid_layers": lyrs[mid]}


# ============================================================================
# MARGINS (task 3)
# ============================================================================

def clean_output_margins(model, prompts) -> dict:
    """
    Clean forward pass per prompt; final position logits (incl. b_U).

    Returns logit_gap (top1 − top2), p_top1, top1_token, each [n_prompts].
    """
    gap, p1, tok = [], [], []
    with torch.no_grad():
        for p in prompts:
            logits = model(model.to_tokens(p, prepend_bos=True))[0, -1, :].float()
            top2   = torch.topk(logits, 2)
            gap.append((top2.values[0] - top2.values[1]).item())
            p1.append(torch.softmax(logits, -1)[top2.indices[0]].item())
            tok.append(int(top2.indices[0].item()))
    return {"logit_gap": np.array(gap), "p_top1": np.array(p1), "top1_token": np.array(tok)}


# ============================================================================
# POST-HOC ABLATION, BATCHED OVER BASES (tasks 4, 5)
# ============================================================================

def _project(X: torch.Tensor, Qs: torch.Tensor) -> torch.Tensor:
    """X [n, d], Qs [B, d, k] → Q_b Q_bᵀ x for every (b, n): [B, n, d]."""
    return torch.einsum("bdk,bnk->bnd", Qs, torch.einsum("nd,bdk->bnk", X, Qs))


def posthoc_flips_batched(
    X:          torch.Tensor,       # [n_prompts, d_model] residuals at ONE layer
    Qs:         torch.Tensor,       # [B, d_model, k] bases
    W_U:        torch.Tensor,       # [d_model, vocab]
    ln_final:   Callable,
    convention: str,                # "preLN" or "postLN"
    chunk_rows: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Post-hoc ablation for B bases at once.

    preLN  (existing convention): logits = ln_final(QQᵀ r) @ W_U
    postLN (commented-out alt.):  logits = (QQᵀ ln_final(r)) @ W_U
    Clean comparator in both: ln_final(r) @ W_U (no b_U, as in ablation_compute).

    Returns:
        flips  bool    [B, n_prompts] — top-1 changed vs clean
        energy float64 [B, n_prompts] — 1 − ‖QQᵀx‖²/‖x‖², x = vector projected
    """
    if convention not in ("preLN", "postLN"):
        raise ValueError(convention)
    with torch.no_grad():
        normed    = ln_final(X)
        clean_top = (normed @ W_U).argmax(-1)                       # [n]
        x_in      = X if convention == "preLN" else normed
        B, n      = Qs.shape[0], X.shape[0]
        per_chunk = max(1, chunk_rows // n)
        flips  = np.zeros((B, n), dtype=bool)
        energy = np.zeros((B, n), dtype=np.float64)
        x_sq   = (x_in.double() ** 2).sum(-1)
        for b0 in range(0, B, per_chunk):
            P = _project(x_in, Qs[b0:b0 + per_chunk])               # [b, n, d]
            energy[b0:b0 + per_chunk] = (1.0 - (P.double() ** 2).sum(-1) / x_sq).numpy()
            v = ln_final(P) if convention == "preLN" else P
            top = (v @ W_U).argmax(-1)                              # [b, n]
            flips[b0:b0 + per_chunk] = (top != clean_top[None, :]).numpy()
    return flips, energy


# ============================================================================
# INTERVENTION ABLATION, BATCHED OVER BASES (task 4)
# ============================================================================

def intervention_flips_batched(
    model,
    prompt:    str,
    layer:     int,
    Qs:        torch.Tensor,        # [B, d_model, k]
    clean_top1: int,
) -> np.ndarray:
    """
    Zero r⊥ at blocks.{layer}.hook_resid_post, final position only, using a
    different basis Q_b for each batch element b (same prompt repeated B times).

    Returns flips bool [B] vs clean_top1 (the clean model output token).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)
    B      = Qs.shape[0]
    Qs_dev = Qs.to(tokens.device)

    def hook_fn(value, hook):
        r = value[:, -1, :].float()                                       # [B, d]
        r_abl = torch.einsum("bdk,bk->bd", Qs_dev, torch.einsum("bd,bdk->bk", r, Qs_dev))
        value[:, -1, :] = r_abl.to(value.dtype)
        return value

    with torch.no_grad():
        logits = model.run_with_hooks(
            tokens.repeat(B, 1),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)],
        )[:, -1, :]
    return (logits.argmax(-1).cpu().numpy() != clean_top1)


# ============================================================================
# STATISTICS ON CONTROL RESULTS (tasks 4, 6)
# ============================================================================

def null_comparison(observed: float, null: np.ndarray) -> dict:
    """Decision log #12."""
    null = np.asarray(null, dtype=np.float64)
    return {
        "observed":   float(observed),
        "null_median": float(np.median(null)),
        "null_q05":   float(np.quantile(null, 0.05)),
        "null_q95":   float(np.quantile(null, 0.95)),
        "percentile": float((null < observed).mean() + 0.5 * (null == observed).mean()),
        "p_one_sided": float((1 + (null >= observed).sum()) / (null.size + 1)),
        "n_null":     int(null.size),
    }


def fragility_verdict(cmp: dict) -> str:
    """Decision log #18."""
    if cmp["observed"] > cmp["null_q95"]:
        return "survives"
    if cmp["observed"] > cmp["null_median"]:
        return "weakened"
    return "dies"


def margin_terciles(logit_gap: np.ndarray) -> np.ndarray:
    """Tercile label 0/1/2 from fixed full-sample quantiles (decision log #16)."""
    q1, q2 = np.quantile(logit_gap, [1 / 3, 2 / 3])
    return np.where(logit_gap <= q1, 0, np.where(logit_gap <= q2, 1, 2))


def tercile_rates(flips, logit_gap, roles, pair_ids,
                  B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> list[dict]:
    """Flip rate per (tercile, role ∈ {all, base, contrast}) with pair-bootstrap CI."""
    flips = np.asarray(flips, dtype=np.float64)
    roles = np.asarray(roles)
    terc  = margin_terciles(np.asarray(logit_gap))
    out   = []
    for t in range(3):
        for role in ("all", "base", "contrast"):
            sel = (terc == t) & ((roles == role) if role != "all" else True)

            def stat(idx, sel=sel):
                s = sel[idx]
                return flips[idx][s].mean() if s.any() else np.nan

            bs = pair_bootstrap(stat, pair_ids, B, seed)
            lo, hi = percentile_ci(bs)
            out.append({"tercile": t, "role": role, "n": int(sel.sum()),
                        "rate": float(flips[sel].mean()) if sel.any() else np.nan,
                        "ci_lo": lo, "ci_hi": hi})
    return out


def _design(logit_gap, roles):
    z = (logit_gap - logit_gap.mean()) / logit_gap.std(ddof=0)
    return np.column_stack([np.ones_like(z), z, (np.asarray(roles) == "contrast").astype(float)])


def margin_regression(flips, logit_gap, roles, pair_ids,
                      B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> dict:
    """
    flip ~ 1 + z(logit_gap) + role(contrast=1), unpenalized IRLS (decision log #17).

    z is standardized once on the full sample. Returns full-sample coefs,
    convergence flag, pair-bootstrap percentile CIs over converged resamples,
    and the converged fraction.
    """
    y = np.asarray(flips, dtype=np.float64)
    X = _design(np.asarray(logit_gap, dtype=np.float64), roles)
    coef, ok = logistic_irls(X, y)
    boots = []
    for idx in pair_bootstrap_indices(pair_ids, B, seed):
        c, conv = logistic_irls(X[idx], y[idx])
        boots.append(c if conv else np.full(3, np.nan))
    boots = np.array(boots)
    conv_frac = float(np.isfinite(boots[:, 0]).mean())
    res = {"n_flips": int(y.sum()), "n": int(y.size), "converged": bool(ok),
           "conv_frac": conv_frac}
    for j, name in enumerate(("intercept", "margin_z", "role_contrast")):
        res[name] = float(coef[j]) if ok else np.nan
        res[f"{name}_ci"] = percentile_ci(boots[:, j]) if ok else (np.nan, np.nan)
    return res


def differential_verdict(paired: dict, reg: dict) -> str:
    """Decision log #19."""
    if not (paired["mean_diff"] > 0):
        return "dies"
    adj_lo = reg["role_contrast_ci"][0]
    if paired["ci_lo"] > 0 and reg["converged"] and np.isfinite(adj_lo) and adj_lo > 0:
        return "survives"
    return "weakened"


def majority_verdict(labels: list[str]) -> str:
    """Decision log #20."""
    for lab in set(labels):
        if labels.count(lab) >= 2:
            return lab
    return "weakened"


# ============================================================================
# SERIALIZATION
# ============================================================================

def save_controls(path, **arrays) -> None:
    """Save a control-result dict to .npz (object arrays allowed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) if not isinstance(v, np.ndarray) else v
                                 for k, v in arrays.items()})


def load_controls(path) -> dict:
    """Load a control-result .npz into a plain dict."""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}
