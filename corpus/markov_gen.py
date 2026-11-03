"""
markov_gen.py — Synthetic prompts from Markov chains with a tunable entropy rate.

Each prompt is a sequence of L words drawn from a fixed 20-word alphabet by a
first-order Markov chain whose ENTROPY RATE (bits per token) is set exactly.
A shuffled copy of every sequence is its contrast: same words, same counts,
order destroyed.

--------------------------------------------------------------------------
The entropy-rate dial
--------------------------------------------------------------------------
Let V = 20 states, Π a random cyclic permutation matrix (state i → σ(i), one
cycle through all V states), and U the uniform matrix (every entry 1/V).
The transition matrix is the mixture

    P(λ) = (1 − λ) Π + λ U,        λ ∈ [0, 1].

Every row has one entry a = 1 − λ + λ/V (the "expected" successor) and V − 1
entries b = λ/V, so every row has the same entropy

    h(λ) = −a log2 a − (V − 1) b log2 b.

P(λ) is doubly stochastic, so its stationary distribution is uniform for
every λ, and the entropy rate is exactly H = Σ_i π_i h_i = h(λ). h is
monotone in λ, from 0 bits (λ = 0, a deterministic 20-cycle) to log2 20
≈ 4.32 bits (λ = 1, i.i.d. uniform words). For a target H we solve h(λ) = H
by bisection.

Because the stationary distribution is uniform at every setting, the word
frequencies are the same across the whole dial: only the sequential
structure changes. The shuffle contrast removes that structure while keeping
the exact word counts of its partner sequence.

Notes:
  - H is the entropy of the TRUE next-word distribution given the previous
    word. The model does not know P; it has to infer it in context. Early in
    a sequence even an ideal in-context learner is less certain than H.
  - A fresh random cycle σ is drawn for every sequence, so prompts at the same
    H differ in which successor is "expected".

Each corpus entry carries the standard fields (pair_id, role, category,
prompt) that extraction.py reads, plus corpus-only fields:
    entropy_rate_target, entropy_rate_true, lam, length, seed_key,
    empirical_cond_entropy, empirical_unigram_entropy, contrast_type.

Usage:
    python corpus/markov_gen.py --length 64             # → corpus/markov_entropy_L64.json
    python corpus/markov_gen.py --length 256 --stats
    (max --length 1023 for GPT-2: BOS + 1023 = n_ctx 1024)
    python corpus/markov_gen.py --length 64 --check-tokenizers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_CORPUS_DIR = Path(__file__).resolve().parent

# Single tokens WITH a leading space in both the GPT-2 and GPT-NeoX (Pythia)
# tokenizers (checked 2026-11-02). Concrete nouns from unrelated domains.
ALPHABET = ["apple", "river", "chair", "cloud", "horse", "window", "bread",
            "stone", "garden", "pencil", "ocean", "candle", "tiger", "mirror",
            "bottle", "forest", "jacket", "violin", "rabbit", "lamp"]
V = len(ALPHABET)

ENTROPY_LEVELS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, float(np.log2(V))]
N_SEEDS     = 10
MASTER_SEED = 20261102


# ============================================================================
# THE DIAL
# ============================================================================

def row_entropy(lam: float, v: int = V) -> float:
    """Entropy (bits) of every row of P(λ) = (1−λ)Π + λU."""
    a = 1.0 - lam + lam / v
    b = lam / v
    h = 0.0
    if a > 0:
        h -= a * np.log2(a)
    if b > 0:
        h -= (v - 1) * b * np.log2(b)
    return float(h)


def lam_for_entropy(H: float, v: int = V, tol: float = 1e-12) -> float:
    """Solve row_entropy(λ) = H by bisection (h is monotone on [0, 1])."""
    if H <= 0:
        return 0.0
    if H >= np.log2(v) - 1e-12:
        return 1.0
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if row_entropy(mid, v) < H:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def transition_matrix(lam: float, sigma: np.ndarray) -> np.ndarray:
    """P(λ) for the cyclic permutation sigma (state i → sigma[i])."""
    v = len(sigma)
    P = np.full((v, v), lam / v)
    P[np.arange(v), sigma] += 1.0 - lam
    return P


def random_cycle(rng: np.random.Generator, v: int = V) -> np.ndarray:
    """A uniformly random single v-cycle as a successor array."""
    order = rng.permutation(v)
    sigma = np.empty(v, dtype=int)
    sigma[order] = np.roll(order, -1)
    return sigma


def sample_chain(P: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Sample `length` states; start from the (uniform) stationary distribution."""
    v = P.shape[0]
    x = np.empty(length, dtype=int)
    x[0] = rng.integers(v)
    cum = P.cumsum(axis=1)
    u = rng.random(length)
    for t in range(1, length):
        x[t] = min(int(np.searchsorted(cum[x[t - 1]], u[t], side="right")), v - 1)
    return x


# ============================================================================
# EMPIRICAL DIAGNOSTICS (plug-in estimates on one sequence)
# ============================================================================

def empirical_unigram_entropy(x: np.ndarray, v: int = V) -> float:
    p = np.bincount(x, minlength=v) / len(x)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def empirical_cond_entropy(x: np.ndarray, v: int = V) -> float:
    """Plug-in H(X_t | X_{t−1}) from bigram counts. Biased low for short x."""
    C = np.zeros((v, v))
    np.add.at(C, (x[:-1], x[1:]), 1)
    n = C.sum()
    rows = C.sum(axis=1)
    h = 0.0
    for i in range(v):
        if rows[i] == 0:
            continue
        p = C[i][C[i] > 0] / rows[i]
        h += rows[i] / n * float(-(p * np.log2(p)).sum())
    return h


# ============================================================================
# CORPUS
# ============================================================================

def to_prompt(x: np.ndarray) -> str:
    """Leading space on every word so each is the ' word' token."""
    return "".join(" " + ALPHABET[i] for i in x)


def build_corpus(length: int, n_seeds: int = N_SEEDS,
                 levels=ENTROPY_LEVELS, seed: int = MASTER_SEED) -> list[dict]:
    corpus, pair_id = [], 0
    for li, H in enumerate(levels):
        lam = lam_for_entropy(H)
        for rep in range(n_seeds):
            key   = [seed, length, li, rep]
            rng   = np.random.default_rng(key)
            sigma = random_cycle(rng)
            x     = sample_chain(transition_matrix(lam, sigma), length, rng)
            xs    = rng.permutation(x)
            common = dict(
                pair_id=pair_id, category=f"markov_H{H:.2f}",
                contrast_type="shuffle", entropy_rate_target=float(H),
                entropy_rate_true=row_entropy(lam), lam=float(lam),
                length=length, seed_key=key,
                description=f"Markov chain, H={H:.2f} bits/token, L={length}, rep {rep}",
            )
            for role, seq in (("base", x), ("contrast", xs)):
                corpus.append(dict(
                    **common, role=role, prompt=to_prompt(seq),
                    empirical_cond_entropy=empirical_cond_entropy(seq),
                    empirical_unigram_entropy=empirical_unigram_entropy(seq),
                ))
            pair_id += 1
    return corpus


def validate_corpus(corpus: list[dict]) -> list[str]:
    """Fail-closed checks: unique prompts, two roles per pair, exact word count."""
    problems = []
    prompts = [e["prompt"] for e in corpus]
    if len(set(prompts)) != len(prompts):
        problems.append("duplicate prompt text (pair join on prompt would be ambiguous)")
    for pid in {e["pair_id"] for e in corpus}:
        roles = sorted(e["role"] for e in corpus if e["pair_id"] == pid)
        if roles != ["base", "contrast"]:
            problems.append(f"pair {pid}: roles {roles}")
    for e in corpus:
        if len(e["prompt"].split()) != e["length"]:
            problems.append(f"pair {e['pair_id']} {e['role']}: wrong word count")
    return problems


def check_tokenizers(corpus: list[dict]) -> None:
    """Every prompt must tokenize to exactly `length` tokens in both families."""
    from transformers import AutoTokenizer
    for name in ("gpt2", "EleutherAI/pythia-160m-deduped"):
        tok = AutoTokenizer.from_pretrained(name)
        bad = [e for e in corpus if len(tok.encode(e["prompt"])) != e["length"]]
        print(f"  {name:32s} token count == length for {len(corpus) - len(bad)}/{len(corpus)} prompts")
        if bad:
            raise SystemExit(f"tokenization check failed for {name}")


def print_stats(corpus: list[dict]) -> None:
    print(f"  {'H target':>8} {'lambda':>8} {'p(expected)':>11} "
          f"{'emp H(cond) base':>17} {'emp H(cond) shuf':>17} {'emp H(unigram)':>15}")
    for cat in dict.fromkeys(e["category"] for e in corpus):
        b = [e for e in corpus if e["category"] == cat and e["role"] == "base"]
        s = [e for e in corpus if e["category"] == cat and e["role"] == "contrast"]
        lam = b[0]["lam"]
        print(f"  {b[0]['entropy_rate_target']:8.2f} {lam:8.4f} {1 - lam + lam / V:11.4f} "
              f"{np.mean([e['empirical_cond_entropy'] for e in b]):17.3f} "
              f"{np.mean([e['empirical_cond_entropy'] for e in s]):17.3f} "
              f"{np.mean([e['empirical_unigram_entropy'] for e in b]):15.3f}")


def main():
    ap = argparse.ArgumentParser(description="Markov-chain entropy-rate corpus")
    ap.add_argument("--length", type=int, required=True, help="words (= tokens) per prompt")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--output", type=str, default=None,
                    help="default: corpus/markov_entropy_L{length}.json")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--check-tokenizers", action="store_true")
    args = ap.parse_args()

    corpus = build_corpus(args.length, args.n_seeds)
    problems = validate_corpus(corpus)
    if problems:
        raise SystemExit("Corpus validation failed:\n  " + "\n  ".join(problems[:10]))
    if args.check_tokenizers:
        check_tokenizers(corpus)
    if args.stats:
        print_stats(corpus)

    out = Path(args.output) if args.output else _CORPUS_DIR / f"markov_entropy_L{args.length}.json"
    out.write_text(json.dumps(corpus, indent=2))
    print(f"Wrote {len(corpus)} prompts ({len(corpus) // 2} pairs) → {out}")


if __name__ == "__main__":
    main()
