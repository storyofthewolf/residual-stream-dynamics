"""computation.py — Entropy and metric computation over ActivationRecords.

Consumes ActivationRecords from extraction.py.
Produces EntropyRecords for consumption by entropy_plots.py.

Pipeline position:
    extraction.py → COMPUTATION (this file) → entropy_plots.py

Two parallel computation paths, both producing EntropyRecords:

    compute_residual_stream_entropy(record, alphas, norm_keys)
        Geometric entropy measured directly in activation space.
        Normalization is a design choice: energy, abs, or softmax.
        No model components required — ActivationRecord is self-contained.
        Produces len(norm_keys) × len(alphas) EntropyRecords.

    compute_logit_lens_entropy(record, alphas, W_U, ln_final)
        Semantic entropy measured in token prediction space.
        Normalization path is fixed: ln_final → @ W_U → softmax.
        Requires W_U and ln_final passed from the workflow layer.
        Produces len(alphas) EntropyRecords with norm_key="logit_lens".

Both functions accept one ActivationRecord and return a list of EntropyRecords.
Iteration over corpora is handled by the workflow scripts, not here.

EntropyRecord fields:
    prompt      : str
    str_tokens  : list[str]
    model_name  : str
    hook_type   : str              — which hook this entropy was computed from
    norm_key    : str              — "energy", "abs", "softmax", or "logit_lens"
    alpha       : float            — Rényi order parameter
    surface     : np.ndarray       — shape [n_layers, seq_len]
    d_model     : int              — activation dimension of source record
    pair_id/role/category          — passed through from ActivationRecord

Normalization methods (residual stream path):
    normalize_energy    — v²/Σv²      energy weighting (effective rank)
    normalize_abs       — |v|/Σ|v|    linear magnitude weighting
    normalize_softmax   — softmax(v)   exponential weighting

Logit-lens normalization (fixed path):
    ln_final(resid) @ W_U → softmax  — projects into token prediction space

Entropy formula:
    renyi_entropy(probs, alpha)    — generalized Rényi entropy
        alpha → 1.0  recovers Shannon entropy
        alpha  < 1   sensitive to rare/small probability mass
        alpha  > 1   dominated by large probability mass

Convenience wrappers (importable by other scripts):
    effective_rank(activations, alpha)      — energy norm + Rényi
    effective_rank_abs(activations, alpha)  — abs norm + Rényi
    softmax_entropy(activations, alpha)     — softmax norm + Rényi

Expansion points:
    Future analysis types (RenyiSpectrumRecord, VonNeumannRecord,
    DynamicsRecord) will be added to this file as new dataclasses and
    compute functions following the same ActivationRecord → *Record pattern.
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from collections import defaultdict

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

from extraction import ActivationRecord


# ============================================================================
# NORMALIZATION METHODS TABLE
# Single source of truth — imported by entropy_plots.py for display labels.
#
# "logit_lens" is included here for label lookups only.
# Its normalization path is not a function in NORM_FN — it is handled
# internally by compute_logit_lens_entropy(), which requires model components
# that are not available at the computation layer.
# ============================================================================

def normalize_energy(v: torch.Tensor) -> torch.Tensor:
    """v^2/Σv^2 — energy weighting by squared magnitude.
    Treats activation dimensions as energy density.
    Primary normalization for effective rank entropy."""
    v = v.flatten().float()
    v2 = v ** 2
    return (v2 / v2.sum()).clamp(min=1e-12)


def normalize_abs(v: torch.Tensor) -> torch.Tensor:
    """|v|/Σ|v| — linear magnitude weighting.
    More sensitive to small activations than normalize_energy."""
    v = v.flatten().float().abs()
    return (v / v.sum()).clamp(min=1e-12)


def normalize_softmax(v: torch.Tensor) -> torch.Tensor:
    """softmax(v) — exponential weighting.
    Heavily concentrates probability on the largest activations.
    Least informative about geometric structure — included for comparison."""
    v = v.flatten().float()
    return torch.softmax(v, dim=0).clamp(min=1e-12)


# NORM_METHODS: ordered list of (key, display_label, function_or_None)
# function is None for logit_lens — normalization is handled separately.
NORM_METHODS = [
    ("energy",     "Energy v^2/Sv^2",            normalize_energy),
    ("abs",        "Absolute |v|/S|v|",           normalize_abs),
    ("softmax",    "Softmax",                     normalize_softmax),
    ("logit_lens", "Logit Lens (token space)",    None),
]

# Dict for lookup by key string — residual stream norms only
NORM_FN = {key: fn for key, _, fn in NORM_METHODS if fn is not None}

# Display labels for all norm keys including logit_lens
NORM_LABELS = {key: label for key, label, _ in NORM_METHODS}

# Ordered list of residual stream norm keys (excludes logit_lens)
RESIDUAL_NORM_KEYS = [key for key, _, fn in NORM_METHODS if fn is not None]


# ============================================================================
# ENTROPY FORMULA
# ============================================================================

def renyi_entropy(probs: torch.Tensor, alpha: float) -> float:
    """Renyi entropy of order alpha.

    H_alpha = (1/(1-alpha)) * log2(sum p_i^alpha)

    Special case alpha -> 1.0: Shannon entropy H = -sum p_i log2(p_i)

    Args:
        probs: valid probability distribution (non-negative, sums to 1)
        alpha: order parameter
    Returns:
        entropy in bits
    """
    if abs(alpha - 1.0) < 1e-6:
        return -(probs * probs.log2()).sum().item()
    return (1.0 / (1.0 - alpha)) * (probs.pow(alpha).sum().log2().item())


# ============================================================================
# CONVENIENCE WRAPPERS
# Named normalization + entropy combinations, importable by other scripts.
# ============================================================================

def effective_rank(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Renyi entropy of energy-weighted distribution (v^2/Sv^2).
    At alpha=1.0 this is the classical effective rank measure."""
    return renyi_entropy(normalize_energy(activations), alpha)


def effective_rank_abs(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Renyi entropy of absolute-value-weighted distribution."""
    return renyi_entropy(normalize_abs(activations), alpha)


def softmax_entropy(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Renyi entropy of softmax-weighted distribution."""
    return renyi_entropy(normalize_softmax(activations), alpha)


# ============================================================================
# ENTROPY RECORD
# Output of both compute_residual_stream_entropy() and
# compute_logit_lens_entropy(). One record per (norm_key, alpha) pair.
# ============================================================================

@dataclass
class EntropyRecord:
    """
    Stores a 2D entropy surface for one (hook_type, norm_key, alpha) combination.

    Produced by either compute_residual_stream_entropy() or
    compute_logit_lens_entropy(). The norm_key field distinguishes the source:
        "energy", "abs", "softmax"  — residual stream geometric entropy
        "logit_lens"                — token-space entropy via logit lens projection

    surface shape: [n_layers, seq_len]
        surface[:, t]      — entropy vs layer at token position t
        surface[layer, :]  — entropy vs token position at fixed layer
        surface[:, -1]     — entropy vs layer at final token position

    d_model is carried through from the source ActivationRecord so that
    plotting functions can annotate whether values come from residual stream
    space (d_model=768 for GPT-2 small) or MLP-internal space (d_model=3072).
    For logit_lens records, d_model reflects the source activation dimension,
    not the vocab dimension.

    One ActivationRecord → multiple EntropyRecords (one per norm x alpha combo).
    """
    # Identity
    prompt:     str
    str_tokens: list
    model_name: str
    hook_type:  str       # e.g. "resid_post", "attn_out"
    norm_key:   str       # "energy", "abs", "softmax", or "logit_lens"
    alpha:      float

    # Data
    surface:    np.ndarray   # shape: [n_layers, seq_len]
    d_model:    int          # activation dimension of source ActivationRecord

    # Corpus metadata (None for single-prompt runs)
    pair_id:    Optional[str] = None
    role:       Optional[str] = None
    category:   Optional[str] = None

    @property
    def n_layers(self) -> int:
        return self.surface.shape[0]

    @property
    def seq_len(self) -> int:
        return self.surface.shape[1]

    def final_token_curve(self) -> np.ndarray:
        """Entropy vs layer at final token position. Shape: [n_layers]."""
        return self.surface[:, -1]

    def layer_curve(self, layer: int) -> np.ndarray:
        """Entropy vs token position at fixed layer. Shape: [seq_len]."""
        return self.surface[layer, :]

    def token_curve(self, t: int) -> np.ndarray:
        """Entropy vs layer at fixed token position. Shape: [n_layers]."""
        return self.surface[:, t]


# ============================================================================
# SHARED PACKAGING HELPER
# Internal utility — builds EntropyRecords from a surfaces dict.
# Used by both compute functions to avoid duplicated packaging code.
# ============================================================================

def _package_entropy_records(
    surfaces:         dict,
    norm_alpha_pairs: list,
    record:           ActivationRecord,
) -> list:
    """
    Package a dict of computed surfaces into a list of EntropyRecords.

    Args:
        surfaces:         dict keyed by (norm_key, alpha) -> np.ndarray [n_layers, seq_len]
        norm_alpha_pairs: list of (norm_key, alpha) tuples in desired output order
        record:           source ActivationRecord for metadata

    Returns:
        list of EntropyRecord
    """
    return [
        EntropyRecord(
            prompt     = record.prompt,
            str_tokens = record.str_tokens,
            model_name = record.model_name,
            hook_type  = record.hook_type,
            norm_key   = nk,
            alpha      = alpha,
            surface    = surfaces[(nk, alpha)],
            d_model    = record.d_model,
            pair_id    = record.pair_id,
            role       = record.role,
            category   = record.category,
        )
        for nk, alpha in norm_alpha_pairs
    ]


# ============================================================================
# COMPUTATION PATH 1: RESIDUAL STREAM ENTROPY
# Geometric entropy measured directly in activation space.
# No model components required.
# ============================================================================

def compute_residual_stream_entropy(
    record:    ActivationRecord,
    alphas:    list,
    norm_keys: Optional[list] = None,
) -> list:
    """
    Compute geometric entropy at every (layer, token_position) for all
    requested normalization methods and alpha values.

    Normalization converts the raw [d_model] activation vector into a
    probability distribution over activation dimensions. The choice of
    normalization is a geometric design decision — each reflects a different
    physical interpretation of what "probability mass" means in activation space.

    Args:
        record:    ActivationRecord from extraction.py
        alphas:    list of Renyi alpha values e.g. [0.5, 1.0, 2.0, 3.0]
        norm_keys: list of normalization keys to compute.
                   Default None uses all residual stream norms:
                   ["energy", "abs", "softmax"]
                   Do not pass "logit_lens" here — use
                   compute_logit_lens_entropy() for that path.

    Returns:
        list of EntropyRecord, one per (norm_key, alpha) combination.
        Length = len(norm_keys) x len(alphas)
    """
    if norm_keys is None:
        norm_keys = RESIDUAL_NORM_KEYS

    if "logit_lens" in norm_keys:
        raise ValueError(
            "'logit_lens' is not a valid norm_key for "
            "compute_residual_stream_entropy(). "
            "Use compute_logit_lens_entropy() instead."
        )

    n_layers = record.n_layers
    seq_len  = record.seq_len

    surfaces = {
        (nk, alpha): np.zeros((n_layers, seq_len), dtype=np.float32)
        for nk in norm_keys
        for alpha in alphas
    }

    for layer in range(n_layers):
        for t in range(seq_len):
            vec = torch.from_numpy(record.activations[layer, t, :])
            for nk in norm_keys:
                probs = NORM_FN[nk](vec)
                for alpha in alphas:
                    surfaces[(nk, alpha)][layer, t] = renyi_entropy(probs, alpha)

    norm_alpha_pairs = [(nk, alpha) for nk in norm_keys for alpha in alphas]
    return _package_entropy_records(surfaces, norm_alpha_pairs, record)


# ============================================================================
# COMPUTATION PATH 2: LOGIT LENS ENTROPY
# Semantic entropy measured in token prediction space.
# Normalization path is fixed: ln_final -> @ W_U -> softmax.
# Requires model components passed from the workflow layer.
# ============================================================================

def compute_logit_lens_entropy(
    record:   ActivationRecord,
    alphas:   list,
    W_U:      torch.Tensor,
    ln_final: Callable,
) -> list:
    """
    Compute token-space entropy at every (layer, token_position) by projecting
    the residual stream through the unembedding matrix.

    At each layer L and token position t, the residual stream is projected as:
        normed  = ln_final(resid[L, :, :])   # apply final layer norm [seq, d_model]
        logits  = normed @ W_U               # project to vocab space [seq, vocab_size]
        probs   = softmax(logits)            # token probability distributions
        H[L, t] = renyi_entropy(probs[t], alpha)

    The normalization path is fixed by the definition of the logit lens —
    it is not a free parameter. The alpha sweep remains free, allowing
    Renyi analysis of the token probability distribution.

    Caveat: ln_final is trained against the final-layer residual stream.
    Applying it to intermediate layers is an approximation that becomes
    less reliable at early layers (0-2). Interpret early-layer results
    with appropriate caution.

    Args:
        record:   ActivationRecord from extraction.py
        alphas:   list of Renyi alpha values e.g. [0.5, 1.0, 2.0, 3.0]
        W_U:      unembedding matrix tensor, shape [d_model, vocab_size]
                  obtained from model.W_U in the workflow layer
        ln_final: final layer norm callable
                  obtained from model.ln_final in the workflow layer

    Returns:
        list of EntropyRecord with norm_key="logit_lens", one per alpha.
        Length = len(alphas)

    FORTRAN analogy:
        Identical loop structure to compute_residual_stream_entropy(),
        but the normalization step calls an external operator (the unembedding
        projection) rather than a local function. Analogous to a parameterized
        physics scheme that requires externally supplied coefficients.
    """
    n_layers = record.n_layers
    seq_len  = record.seq_len
    norm_key = "logit_lens"

    surfaces = {
        (norm_key, alpha): np.zeros((n_layers, seq_len), dtype=np.float32)
        for alpha in alphas
    }

    W_U_cpu = W_U.float().cpu()

    for layer in range(n_layers):
        resid = torch.from_numpy(
            record.activations[layer, :, :]
        ).float()                                          # [seq_len, d_model]

        with torch.no_grad():
            normed    = ln_final(resid).cpu()              # [seq_len, d_model]
            logits    = normed @ W_U_cpu                   # [seq_len, vocab_size]
            probs_all = torch.softmax(logits, dim=-1)      # [seq_len, vocab_size]

        for t in range(seq_len):
            probs = probs_all[t].clamp(min=1e-12)
            for alpha in alphas:
                surfaces[(norm_key, alpha)][layer, t] = renyi_entropy(probs, alpha)

    norm_alpha_pairs = [(norm_key, alpha) for alpha in alphas]
    return _package_entropy_records(surfaces, norm_alpha_pairs, record)


# ============================================================================
# LOOKUP HELPERS
# ============================================================================

def filter_records(
    records:    list,
    hook_type:  Optional[str]   = None,
    norm_key:   Optional[str]   = None,
    alpha:      Optional[float] = None,
    role:       Optional[str]   = None,
    category:   Optional[str]   = None,
) -> list:
    """Filter a list of EntropyRecords by any combination of fields."""
    out = records
    if hook_type is not None:
        out = [r for r in out if r.hook_type == hook_type]
    if norm_key is not None:
        out = [r for r in out if r.norm_key == norm_key]
    if alpha is not None:
        out = [r for r in out if abs(r.alpha - alpha) < 1e-6]
    if role is not None:
        out = [r for r in out if r.role == role]
    if category is not None:
        out = [r for r in out if r.category == category]
    return out


# ============================================================================
# SERIALIZATION
# ============================================================================

def save_entropy_records(records: list, path) -> None:
    """Save a list of EntropyRecords to a single .npz file."""
    n = len(records)
    surfaces    = np.stack([r.surface    for r in records], axis=0)
    prompts     = np.array([r.prompt     for r in records], dtype=object)
    model_names = np.array([r.model_name for r in records], dtype=object)
    hook_types  = np.array([r.hook_type  for r in records], dtype=object)
    norm_keys   = np.array([r.norm_key   for r in records], dtype=object)
    alphas      = np.array([r.alpha      for r in records], dtype=np.float32)
    d_models    = np.array([r.d_model    for r in records], dtype=np.int32)
    roles       = np.array([r.role      or "" for r in records], dtype=object)
    categories  = np.array([r.category  or "" for r in records], dtype=object)
    pair_ids    = np.array([r.pair_id   or "" for r in records], dtype=object)

    np.savez(
        path,
        surfaces    = surfaces,
        prompts     = prompts,
        model_names = model_names,
        hook_types  = hook_types,
        norm_keys   = norm_keys,
        alphas      = alphas,
        d_models    = d_models,
        roles       = roles,
        categories  = categories,
        pair_ids    = pair_ids,
    )
    print(f"  Saved {n} EntropyRecords to {path}")


def load_entropy_records(path) -> list:
    """Load a list of EntropyRecords from a .npz file."""
    d = np.load(path, allow_pickle=True)
    n = len(d["prompts"])
    records = []
    for i in range(n):
        records.append(EntropyRecord(
            prompt     = str(d["prompts"][i]),
            str_tokens = [],
            model_name = str(d["model_names"][i]),
            hook_type  = str(d["hook_types"][i]),
            norm_key   = str(d["norm_keys"][i]),
            alpha      = float(d["alphas"][i]),
            surface    = d["surfaces"][i],
            d_model    = int(d["d_models"][i]),
            role       = str(d["roles"][i])      or None,
            category   = str(d["categories"][i]) or None,
            pair_id    = str(d["pair_ids"][i])   or None,
        ))
    return records


# ============================================================================
# SUMMARY PRINTING
# ============================================================================

def print_summary(
    entropy_records: list,
    alphas:          list,
    norm_key:        str = "energy",
) -> None:
    """
    Print mean trajectory summary for a given norm_key at each alpha.

    Args:
        entropy_records: flat list of EntropyRecord
        alphas:          list of alpha values to summarize
        norm_key:        normalization key to summarize (default "energy").
                         Pass "logit_lens" to summarize logit-lens records.
    """
    label_str = NORM_LABELS.get(norm_key, norm_key)
    print("\n" + "=" * 60)
    print(f"ENTROPY TRAJECTORY SUMMARY ({label_str}, final token position)")
    print("=" * 60)

    for alpha in alphas:
        subset = filter_records(entropy_records, norm_key=norm_key, alpha=alpha)
        if not subset:
            continue
        alpha_label = "Shannon" if abs(alpha - 1.0) < 1e-6 else f"Renyi a={alpha}"
        print(f"\n  {alpha_label}:")

        by_role = defaultdict(list)
        for r in subset:
            by_role[r.role].append(r.final_token_curve())

        for role in ["base", "contrast"]:
            curves = by_role[role]
            if not curves:
                continue
            mean      = np.mean(curves, axis=0)
            min_layer = int(np.argmin(mean))
            n_layers  = len(mean)
            print(f"    {role.upper()} ({len(curves)} prompts): "
                  f"L0={mean[0]:.2f}  "
                  f"L{n_layers-1}={mean[-1]:.2f}  "
                  f"min=L{min_layer}({mean[min_layer]:.2f})")
            print(f"      {' '.join(f'{v:.1f}' for v in mean)}")
    print()
