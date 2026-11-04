"""toy_patching_compute.py — Interpretability tools for the Phase 1 toy transformers.

Helpers for notebooks/toy_induction_guided.ipynb: load checkpoints, measure
loss and KL to the Bayes-optimal bigram predictor, ablate or patch one
component, and recompute layer 1's attention with keys built from a chosen
subset of residual-stream components.

Runs forward passes on toy models loaded from data/toy/ (same allowance as
toy_training.py: these models exist only to be probed). No plotting.

Residual stream of the 2-layer attention-only model, entering layer 1:

    resid_pre[1] = hook_embed + hook_pos_embed + blocks.0.hook_attn_out

That sum is exact, which is what makes the key-component decomposition below
exact up to LayerNorm (whose scale is frozen at its clean value).
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch

from toy_training import ToyConfig, load_history, make_model, reference_logprobs, sample_markov_batch

LOG2 = float(np.log(2))

KEY_COMPONENTS = {
    "embed": "hook_embed",               # which token is here
    "pos":   "hook_pos_embed",           # which position this is
    "attn0": "blocks.0.hook_attn_out",   # what layer 0's head wrote here
}


# ============================================================================
# LOADING
# ============================================================================

def load_toy_run(run_dir) -> tuple[ToyConfig, dict, list[int]]:
    """Config, training history, and the sorted list of checkpoint steps."""
    run_dir = Path(run_dir)
    hist = load_history(run_dir / "history.npz")
    saved = hist["config"].item()
    cfg = ToyConfig(**{f.name: saved[f.name] for f in fields(ToyConfig) if f.name in saved})
    steps = sorted(int(p.stem[4:]) for p in (run_dir / "ckpt").glob("step*.pt"))
    return cfg, hist, steps


def load_checkpoint(run_dir, cfg: ToyConfig, step: int):
    """The model as it was at training step `step` (eval mode)."""
    model = make_model(cfg)
    state = torch.load(Path(run_dir) / "ckpt" / f"step{step:06d}.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def make_eval_batch(cfg: ToyConfig, n: int = 512, seed: int = 12345):
    """Fresh evaluation sequences (seeded) and their exact reference log-probs."""
    x = sample_markov_batch(n, cfg.length, cfg.k, cfg.alpha, torch.Generator().manual_seed(seed))
    return x, reference_logprobs(x, cfg.k, cfg.alpha)


def sweep(run_dir, cfg: ToyConfig, steps: Iterable[int], fn: Callable) -> list:
    """Apply fn(model) to the checkpoint at each step; returns the list of results."""
    return [fn(load_checkpoint(run_dir, cfg, s)) for s in steps]


# ============================================================================
# METRICS
# ============================================================================

def _kl_bits(logp_ref: torch.Tensor, logp: torch.Tensor) -> float:
    """Mean KL(ref ‖ p) in bits over all leading dims."""
    return float((logp_ref.exp() * (logp_ref - logp)).sum(-1).mean() / LOG2)


def run_metrics(model, x, refs, fwd_hooks=()) -> dict:
    """Cross-entropy (bits) and KL(bigram ‖ model) (bits), optionally under hooks."""
    with torch.no_grad():
        logits = model.run_with_hooks(x, fwd_hooks=list(fwd_hooks))
    logp = torch.log_softmax(logits[:, :-1].double(), -1)
    return {"xent": float(-logp.gather(-1, x[:, 1:, None]).mean() / LOG2),
            "kl_bigram": _kl_bits(refs["bigram"], logp)}


def previous_token_score(pattern: torch.Tensor) -> float:
    """Mean attention from position t to t−1. pattern: [B, L, L] (one head)."""
    L = pattern.shape[-1]
    return float(pattern[:, torch.arange(1, L), torch.arange(L - 1)].mean())


def statistical_induction_score(pattern: torch.Tensor, x: torch.Tensor) -> tuple[float, float]:
    """
    Mean total attention from t to positions s ≤ t whose PREVIOUS token equals x_t,
    and the same quantity for uniform causal attention (baseline).
    """
    L = x.shape[1]
    prev = torch.cat([torch.full_like(x[:, :1], -1), x[:, :-1]], 1)
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    mask = ((prev[:, None, :] == x[:, :, None]) & causal).to(pattern.dtype)
    score = (pattern * mask).sum(-1)[:, 1:].mean()
    base = (mask.sum(-1) / causal.sum(-1))[:, 1:].mean()
    return float(score), float(base)


def attention_patterns(model, x) -> list[torch.Tensor]:
    """Attention pattern of head 0 in every layer: list of [B, L, L]."""
    with torch.no_grad():
        _, cache = model.run_with_cache(x, names_filter=lambda n: n.endswith("hook_pattern"))
    return [cache[f"blocks.{l}.attn.hook_pattern"][:, 0] for l in range(model.cfg.n_layers)]


# ============================================================================
# ABLATION AND PATCHING
# ============================================================================

def ablate(model, x, refs, hook_name: str, mode: str = "mean") -> dict:
    """
    Replace one activation and report run_metrics.

    mode="zero": set it to 0.
    mode="mean": set it to its average over the batch at each position. This
                 keeps whatever is the same for every sequence at that position
                 and removes what is specific to the sequence.
    """
    if mode == "zero":
        hook = lambda v, hook: torch.zeros_like(v)
    elif mode == "mean":
        with torch.no_grad():
            _, cache = model.run_with_cache(x, names_filter=hook_name)
        mean = cache[hook_name].mean(dim=0, keepdim=True)
        hook = lambda v, hook: mean.expand_as(v).clone()
    else:
        raise ValueError(f"mode must be 'zero' or 'mean', got {mode!r}")
    return run_metrics(model, x, refs, [(hook_name, hook)])


def shuffle_within_sequences(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Corrupted input: each sequence's tokens permuted (same counts, order destroyed)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.stack([torch.randperm(x.shape[1], generator=g) for _ in range(x.shape[0])])
    return x.gather(1, idx)


def patch_recovery(model, x_clean, x_corrupt, hook_name: str) -> dict:
    """
    Run on x_corrupt but overwrite `hook_name` with its value from the x_clean run.

    Measures how far each run's predictions are from the clean run's:
        kl_corrupt  KL(clean ‖ corrupt run)
        kl_patched  KL(clean ‖ patched run)
        recovered   1 − kl_patched / kl_corrupt   (1 = patching this alone restores clean behavior)
    """
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(x_clean, names_filter=hook_name)
        corrupt_logits = model(x_corrupt)
        patched_logits = model.run_with_hooks(
            x_corrupt, fwd_hooks=[(hook_name, lambda v, hook: cache[hook_name].clone())])
    lc = torch.log_softmax(clean_logits.double(), -1)
    kl_c = _kl_bits(lc, torch.log_softmax(corrupt_logits.double(), -1))
    kl_p = _kl_bits(lc, torch.log_softmax(patched_logits.double(), -1))
    return {"kl_corrupt": kl_c, "kl_patched": kl_p, "recovered": 1.0 - kl_p / kl_c}


# ============================================================================
# WHAT DOES LAYER 1'S KEY READ?
# ============================================================================

def layer1_pattern_from_key_components(model, x, components=("embed", "pos", "attn0")) -> torch.Tensor:
    """
    Recompute layer 1's attention pattern with KEYS built from only the named
    residual components (see KEY_COMPONENTS); queries use the full residual.

    LayerNorm is linear once its scale is fixed, so each component is centered
    and divided by the clean run's ln1 scale, then the chosen ones are summed.
    With all three components this reproduces the model's own pattern
    (the notebook checks this).

    Returns [B, L, L] for head 0.
    """
    blk = model.blocks[1]
    with torch.no_grad():
        _, cache = model.run_with_cache(x)
        scale = cache["blocks.1.ln1.hook_scale"]                      # [B, L, 1]

        def normed(v):
            return (v - v.mean(-1, keepdim=True)) / scale * blk.ln1.w

        q_in = normed(cache["blocks.1.hook_resid_pre"]) + blk.ln1.b
        k_in = blk.ln1.b.expand_as(q_in).clone()
        for c in components:
            k_in = k_in + normed(cache[KEY_COMPONENTS[c]])
        q = torch.einsum("bpd,hde->bphe", q_in, blk.attn.W_Q) + blk.attn.b_Q
        k = torch.einsum("bpd,hde->bphe", k_in, blk.attn.W_K) + blk.attn.b_K
        scores = torch.einsum("bqhe,bkhe->bhqk", q, k) / blk.attn.attn_scale
        L = x.shape[1]
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        return torch.softmax(scores, dim=-1)[:, 0]


def key_component_table(model, x, subsets: dict[str, tuple]) -> dict[str, float]:
    """Statistical-induction score of layer 1 for each named subset of key components."""
    return {name: statistical_induction_score(layer1_pattern_from_key_components(model, x, comps), x)[0]
            for name, comps in subsets.items()}
