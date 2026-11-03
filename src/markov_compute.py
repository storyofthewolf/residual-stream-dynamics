"""markov_compute.py — Entropy-rate analysis for the Markov-chain corpus.

Consumes ActivationRecords (extraction.py) and EntropyRecords
(entropy_compute.py) for prompts from corpus/markov_gen.py. Produces a
per-prompt summary dict persisted as .npz under data/markov/.

No forward passes: the output distribution is the final-layer logit lens,
ln_final(r) @ W_U, computed from stored activations (same convention as
entropy_compute.compute_logit_lens_entropy, i.e. no b_U).

Decision log entries M9–M12 define every quantity here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.stats import rankdata

from math_utils import compute_device
from stats_utils import DEFAULT_B, DEFAULT_SEED, pair_bootstrap_indices, percentile_ci


def second_half_positions(seq_len: int) -> np.ndarray:
    """Word positions L//2+1 … L of a BOS + L-word sequence (decision log M9)."""
    L = seq_len - 1
    return np.arange(L // 2 + 1, L + 1)


def output_distribution_stats(
    record,
    W_U:         torch.Tensor,
    ln_final:    Callable,
    alphabet_ids: np.ndarray,      # [V] token ids of the alphabet words
    token_ids:   np.ndarray,       # [seq_len] token ids of the prompt (incl. BOS)
) -> dict:
    """
    Final-layer next-token statistics at every position (decision log M10, M11).

    Returns per-position arrays [seq_len] (bits):
        H_full    entropy over the full vocabulary
        H_alpha   entropy after renormalizing over the alphabet tokens
        mass      probability mass on the alphabet tokens
        xent      −log2 p(actual next token); NaN at the last position
    """
    W = compute_device(W_U.float())
    r = torch.from_numpy(record.activations[-1]).float().to(W.device)   # [seq, d]
    with torch.no_grad():
        logp = torch.log_softmax(ln_final(r) @ W, dim=-1).double().cpu()  # [seq, vocab]
    p = logp.exp()
    H_full = -(p * logp).sum(-1).numpy() / np.log(2)
    pa = p[:, alphabet_ids]
    mass = pa.sum(-1)
    q = pa / mass[:, None]
    H_alpha = -(q * torch.log(q.clamp_min(1e-300))).sum(-1).numpy() / np.log(2)
    xent = np.full(len(token_ids), np.nan)
    nxt = torch.from_numpy(np.asarray(token_ids[1:], dtype=np.int64))
    xent[:-1] = -logp[torch.arange(len(token_ids) - 1), nxt].numpy() / np.log(2)
    return {"H_full": H_full, "H_alpha": H_alpha, "mass": mass.numpy(), "xent": xent}


def summarize_prompt(resid_surface: np.ndarray, out_stats: dict) -> dict:
    """
    Second-half averages for one prompt.

    resid_surface: [n_layers, seq_len] energy-norm α=1 entropy (EntropyRecord.surface)
    """
    pos = second_half_positions(resid_surface.shape[1])
    return {
        "resid":   resid_surface[:, pos].mean(axis=1),                  # [n_layers]
        "H_full":  float(out_stats["H_full"][pos].mean()),
        "H_alpha": float(out_stats["H_alpha"][pos].mean()),
        "mass":    float(out_stats["mass"][pos].mean()),
        "xent":    float(np.nanmean(out_stats["xent"][pos])),
    }


# ============================================================================
# STATISTICS (decision log M12)
# ============================================================================

def _spearman_cols(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman ρ of x [n] with each column of Y [n, m] (average ranks for ties)."""
    rx = rankdata(x)
    rY = rankdata(Y, axis=0)
    rx = rx - rx.mean()
    rY = rY - rY.mean(axis=0)
    denom = np.sqrt((rx ** 2).sum() * (rY ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (rx @ rY) / denom


def spearman_with_pair_ci(H, Y, roles, pair_ids, role: str,
                          B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> dict:
    """
    Spearman ρ(H, Y[:, j]) over prompts of `role`, per column j, with a pair
    bootstrap CI (resampling (base, shuffle) pairs).

    H [n], Y [n] or [n, m]. Returns rho, ci_lo, ci_hi, each [m].
    """
    H = np.asarray(H, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Y = Y[:, None] if Y.ndim == 1 else Y
    sel = np.asarray(roles) == role
    rho = _spearman_cols(H[sel], Y[sel])
    boots = np.empty((B, Y.shape[1]))
    for b, idx in enumerate(pair_bootstrap_indices(pair_ids, B, seed)):
        idx = idx[sel[idx]]
        boots[b] = _spearman_cols(H[idx], Y[idx])
    ci = np.array([percentile_ci(boots[:, j]) for j in range(Y.shape[1])])
    return {"rho": rho, "ci_lo": ci[:, 0], "ci_hi": ci[:, 1]}


def level_means_with_ci(H, y, roles, pair_ids, role: str,
                        B: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> dict:
    """Mean of y per H level for `role`, CI by resampling that level's pairs."""
    H, y, roles, pair_ids = map(np.asarray, (H, y, roles, pair_ids))
    levels = np.unique(H)
    mean, lo, hi = [], [], []
    for h in levels:
        sel = (H == h) & (roles == role)
        yy = y[sel].astype(np.float64)
        rng = np.random.default_rng([seed, int(round(h * 1000))])
        n  = yy.shape[0]
        bs = yy[rng.integers(0, n, size=(B, n))].mean(axis=1)
        mean.append(yy.mean(axis=0))
        c = np.quantile(bs, [0.025, 0.975], axis=0)
        lo.append(c[0]); hi.append(c[1])
    return {"levels": levels, "mean": np.array(mean), "ci_lo": np.array(lo), "ci_hi": np.array(hi)}


# ============================================================================
# SERIALIZATION
# ============================================================================

def save_markov_summary(path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})


def load_markov_summary(path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}
