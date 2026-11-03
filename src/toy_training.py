"""toy_training.py — Train small attention-only transformers on in-context Markov chains.

Phase 1 of the model hierarchy (decision log P1–P7): a transformer small
enough to explain completely, trained on idealized data with a known optimal
algorithm. Reproduces the staged learning of Edelman et al. (2024)
(uniform → in-context unigram → in-context bigram) and the statistical
induction head that implements the bigram stage.

Pipeline position: like extraction.py, this module PRODUCES the object that
later analysis consumes (a trained HookedTransformer plus checkpoints), so it
runs forward/backward passes. Analysis of the trained model goes through the
usual extraction → compute path.

Contents:
    sample_markov_batch       fresh Dirichlet transition matrix per sequence
    reference_logprobs        exact uniform / unigram / bigram predictors
    make_model                2-layer attn-only HookedTransformer
    evaluate                  loss, KL to each reference, attention signatures
    train                     training loop with periodic evaluation + checkpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

LOG2 = float(np.log(2))


# ============================================================================
# DATA
# ============================================================================

def sample_markov_batch(batch: int, length: int, k: int, alpha: float,
                        gen: torch.Generator) -> torch.Tensor:
    """[batch, length] int64 sequences; each from its own P with rows ~ Dirichlet(alpha)."""
    g = torch.empty(batch, k, k).exponential_(1.0, generator=gen) if alpha == 1.0 else \
        torch.distributions.Gamma(torch.full((batch, k, k), alpha), 1.0).sample()
    P = g / g.sum(-1, keepdim=True)                                   # [B, k, k]
    x = torch.empty(batch, length, dtype=torch.long)
    x[:, 0] = torch.randint(0, k, (batch,), generator=gen)
    u = torch.rand(batch, length, generator=gen)
    cum = P.cumsum(-1)
    b = torch.arange(batch)
    for t in range(1, length):
        row = cum[b, x[:, t - 1]]                                     # [B, k]
        x[:, t] = (u[:, t:t + 1] > row).sum(-1).clamp(max=k - 1)
    return x


def reference_logprobs(x: torch.Tensor, k: int, alpha: float = 1.0) -> dict:
    """
    Exact next-token log-probabilities (natural log) of the three reference
    strategies at every position t (predicting x[t+1]), shape [B, L-1, k].

        uniform : 1/k
        unigram : (count_j(x_0..x_t) + α) / (t+1 + kα)
        bigram  : (C[x_t, j] over transitions in x_0..x_t + α) / (row total + kα)
                  — Bayes-optimal for rows ~ Dirichlet(α)
    """
    B, L = x.shape
    oh = F.one_hot(x, k).double()                                     # [B, L, k]
    uni = (oh.cumsum(1)[:, :-1] + alpha) / (torch.arange(1, L, dtype=torch.double)[None, :, None] + k * alpha)
    # transition counts C_t[i, j] = #{s < t : x_s = i, x_{s+1} = j}, for s+1 <= t
    trans = oh[:, :-1, :, None] * oh[:, 1:, None, :]                  # [B, L-1, k, k], transition s→s+1
    C = torch.cat([torch.zeros(B, 1, k, k, dtype=torch.double), trans.cumsum(1)], 1)[:, :-1]  # counts up to t
    row = C[torch.arange(B)[:, None], torch.arange(L - 1)[None, :], x[:, :-1]]               # [B, L-1, k]
    bi = (row + alpha) / (row.sum(-1, keepdim=True) + k * alpha)
    return {"uniform": torch.full_like(uni, 1.0 / k).log(), "unigram": uni.log(), "bigram": bi.log()}


# ============================================================================
# MODEL
# ============================================================================

@dataclass
class ToyConfig:
    k: int = 3
    length: int = 100
    alpha: float = 1.0
    n_layers: int = 2
    n_heads: int = 1
    d_model: int = 64
    lr: float = 1e-3
    batch: int = 64
    steps: int = 20_000
    eval_every: int = 250
    n_eval: int = 2048
    seed: int = 0
    eval_seed: int = 20261103


def make_model(cfg: ToyConfig) -> HookedTransformer:
    torch.manual_seed(cfg.seed)
    hcfg = HookedTransformerConfig(
        n_layers=cfg.n_layers, d_model=cfg.d_model, n_heads=cfg.n_heads,
        d_head=cfg.d_model // cfg.n_heads, n_ctx=cfg.length, d_vocab=cfg.k,
        attn_only=True, normalization_type="LN", positional_embedding_type="standard",
        act_fn=None, seed=cfg.seed, device="cpu",
    )
    return HookedTransformer(hcfg)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model: HookedTransformer, x_eval: torch.Tensor, refs: dict, k: int) -> dict:
    """
    Cross-entropy (bits), KL(reference ‖ model) to each reference (bits), and
    attention signatures (decision log P5, P6). All averaged over batch and
    positions 1..L-2 predictions (every position with a target).
    """
    model.eval()
    with torch.no_grad():
        logits, cache = model.run_with_cache(x_eval, names_filter=lambda n: n.endswith("hook_pattern"))
        logp = torch.log_softmax(logits[:, :-1].double(), -1)         # [B, L-1, k]
        tgt  = x_eval[:, 1:]
        out = {"xent": float(-logp.gather(-1, tgt[..., None]).mean() / LOG2)}
        for name, lr in refs.items():
            out[f"xent_{name}"] = float(-lr.gather(-1, tgt[..., None]).mean() / LOG2)
            out[f"kl_{name}"]   = float((lr.exp() * (lr - logp)).sum(-1).mean() / LOG2)

        L = x_eval.shape[1]
        A0 = cache["blocks.0.attn.hook_pattern"][:, 0]                # [B, L, L]
        out["prev_token_score"] = float(A0[:, torch.arange(1, L), torch.arange(L - 1)].mean())
        A1 = cache[f"blocks.{model.cfg.n_layers - 1}.attn.hook_pattern"][:, 0]
        # mask[b, t, s] = 1 if s >= 1, s <= t and x[b, s-1] == x[b, t]
        prev = torch.cat([torch.full_like(x_eval[:, :1], -1), x_eval[:, :-1]], 1)  # x[s-1]
        match = (prev[:, None, :] == x_eval[:, :, None])
        causal = torch.ones(L, L, dtype=torch.bool).tril()
        mask = (match & causal).double()
        out["induction_score"] = float((A1 * mask).sum(-1)[:, 1:].mean())
        out["induction_baseline"] = float((mask.sum(-1) / causal.sum(-1))[:, 1:].mean())
    model.train()
    return out


# ============================================================================
# TRAINING
# ============================================================================

def train(cfg: ToyConfig, ckpt_dir: Path | None = None, log=print) -> tuple[HookedTransformer, dict]:
    """Train; return (model, history) where history maps metric → array over evals."""
    model = make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(cfg.seed)
    x_eval = sample_markov_batch(cfg.n_eval, cfg.length, cfg.k, cfg.alpha,
                                 torch.Generator().manual_seed(cfg.eval_seed))
    refs = reference_logprobs(x_eval, cfg.k, cfg.alpha)
    hist: dict[str, list] = {"step": []}

    def record(step):
        m = evaluate(model, x_eval, refs, cfg.k)
        hist["step"].append(step)
        for key, v in m.items():
            hist.setdefault(key, []).append(v)
        if ckpt_dir is not None:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"step{step:06d}.pt")
        log(f"  step {step:6d}  xent {m['xent']:.4f}  KL→uni {m['kl_unigram']:.4f}  "
            f"KL→bi {m['kl_bigram']:.4f}  prev {m['prev_token_score']:.3f}  "
            f"ind {m['induction_score']:.3f} (base {m['induction_baseline']:.3f})")

    record(0)
    for step in range(1, cfg.steps + 1):
        x = sample_markov_batch(cfg.batch, cfg.length, cfg.k, cfg.alpha, gen)
        logits = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.k), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0:
            record(step)
    hist = {k_: np.array(v) for k_, v in hist.items()}
    hist["config"] = np.array(asdict(cfg), dtype=object)
    return model, hist


def save_history(hist: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **hist)


def load_history(path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}
