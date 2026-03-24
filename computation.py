"""computation.py — Entropy and metric computation over ActivationRecords.

Consumes ActivationRecords from extraction.py.
Produces EntropyRecords for consumption by entropy_plots.py.

Pipeline position:
    extraction.py → COMPUTATION (this file) → entropy_plots.py

EntropyRecord fields:
    prompt      : str
    str_tokens  : list[str]
    model_name  : str
    hook_type   : str              — which hook this entropy was computed from
    norm_key    : str              — "energy", "abs", or "softmax"
    alpha       : float            — Rényi order parameter
    surface     : np.ndarray       — shape [n_layers, seq_len]
    d_model     : int              — activation dimension of source record
    pair_id/role/category          — passed through from ActivationRecord

Normalization functions:
    normalize_energy    — v²/Σv²      energy weighting (effective rank)
    normalize_abs       — |v|/Σ|v|    linear magnitude weighting
    normalize_softmax   — softmax(v)   exponential weighting

Entropy functions:
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
from typing import Optional
from collections import defaultdict

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

from extraction import ActivationRecord


# ============================================================================
# NORMALIZATION METHODS TABLE
# Single source of truth — imported by entropy_plots.py.
# Each entry: (key, display_label, function)
# ============================================================================

def normalize_energy(v: torch.Tensor) -> torch.Tensor:
    """v²/Σv² — energy weighting by squared magnitude.
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


NORM_METHODS = [
    ("energy",  "Energy v²/Σv²",    normalize_energy),
    ("abs",     "Absolute |v|/Σ|v|", normalize_abs),
    ("softmax", "Softmax",           normalize_softmax),
]

# Dict for lookup by key string
NORM_FN = {key: fn for key, _, fn in NORM_METHODS}


# ============================================================================
# ENTROPY FORMULA
# ============================================================================

def renyi_entropy(probs: torch.Tensor, alpha: float) -> float:
    """Rényi entropy of order alpha.

    H_alpha = (1/(1-alpha)) * log2(Σ p_i^alpha)

    Special case alpha → 1.0: Shannon entropy H = -Σ p_i log2(p_i)

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
    """Rényi entropy of energy-weighted distribution (v²/Σv²).
    At alpha=1.0 this is the classical effective rank measure."""
    return renyi_entropy(normalize_energy(activations), alpha)


def effective_rank_abs(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Rényi entropy of absolute-value-weighted distribution."""
    return renyi_entropy(normalize_abs(activations), alpha)


def softmax_entropy(activations: torch.Tensor, alpha: float = 1.0) -> float:
    """Rényi entropy of softmax-weighted distribution."""
    return renyi_entropy(normalize_softmax(activations), alpha)


# ============================================================================
# ENTROPY RECORD
# Output of compute_entropy_surface(). One record per (norm_key, alpha) pair.
# ============================================================================

@dataclass
class EntropyRecord:
    """
    Stores a 2D entropy surface for one (hook_type, norm_key, alpha) combination.

    surface shape: [n_layers, seq_len]
        surface[:, t]      — entropy vs layer at token position t
        surface[layer, :]  — entropy vs token position at fixed layer
        surface[:, -1]     — entropy vs layer at final token (old 1D behavior)

    d_model is carried through from the source ActivationRecord so that
    plotting functions can annotate whether values come from residual stream
    space (d_model=768 for GPT-2 small) or MLP-internal space (d_model=3072).

    One ActivationRecord → multiple EntropyRecords (one per norm×alpha combo).
    """
    # Identity
    prompt:     str
    str_tokens: list
    model_name: str
    hook_type:  str       # e.g. "resid_post", "attn_out"
    norm_key:   str       # "energy", "abs", "softmax"
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
# CORE COMPUTATION: 2D ENTROPY SURFACE
# Consumes one ActivationRecord, produces a list of EntropyRecords.
# ============================================================================

def compute_entropy_surface(
    record:   ActivationRecord,
    alphas:   list,
    norm_keys: Optional[list] = None,
) -> list:
    """
    Compute entropy at every (layer, token_position) for all requested
    normalization methods and alpha values.

    Args:
        record:    ActivationRecord from extraction.py
        alphas:    list of Rényi alpha values e.g. [0.5, 1.0, 2.0, 3.0]
        norm_keys: list of normalization keys to compute.
                   Default None uses all three: ["energy", "abs", "softmax"]

    Returns:
        list of EntropyRecord, one per (norm_key, alpha) combination.
        Length = len(norm_keys) × len(alphas)

    FORTRAN analogy:
        Outer loop over norm methods and alpha values (parameter combinations).
        Inner doubly-nested loop over layer and token position, filling a 2D
        array ENTROPY(layer, pos) for each parameter combination.
        Analogous to computing multiple diagnostic fields from the same
        model state — the state is read once, multiple fields are derived.
    """
    if norm_keys is None:
        norm_keys = [key for key, _, _ in NORM_METHODS]

    n_layers = record.n_layers
    seq_len  = record.seq_len

    # Initialize 2D accumulator arrays: one per (norm_key, alpha)
    surfaces = {
        (nk, alpha): np.zeros((n_layers, seq_len), dtype=np.float32)
        for nk in norm_keys
        for alpha in alphas
    }

    # Fill surfaces: loop over layer × token position
    for layer in range(n_layers):
        for t in range(seq_len):
            # Raw activation vector at this (layer, token) — shape: [d_model]
            vec = torch.from_numpy(record.activations[layer, t, :])

            for nk in norm_keys:
                norm_fn = NORM_FN[nk]
                probs   = norm_fn(vec)
                for alpha in alphas:
                    surfaces[(nk, alpha)][layer, t] = renyi_entropy(probs, alpha)

    # Package into EntropyRecords
    entropy_records = []
    for nk in norm_keys:
        for alpha in alphas:
            entropy_records.append(EntropyRecord(
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
            ))

    return entropy_records


# ============================================================================
# BATCH COMPUTATION OVER CORPUS RECORDS
# ============================================================================

def compute_entropy_corpus(
    activation_records: list,
    alphas:   list,
    norm_keys: Optional[list] = None,
) -> list:
    """
    Run compute_entropy_surface over a list of ActivationRecords.

    Args:
        activation_records: list of ActivationRecord (same hook type)
        alphas:             list of Rényi alpha values
        norm_keys:          normalization methods (default: all three)

    Returns:
        flat list of EntropyRecord across all prompts and (norm, alpha) combos
    """
    all_entropy = []
    for i, record in enumerate(activation_records):
        records = compute_entropy_surface(record, alphas, norm_keys)
        all_entropy.extend(records)
        if (i + 1) % 10 == 0 or (i + 1) == len(activation_records):
            print(f"  Computed entropy {i+1}/{len(activation_records)} prompts...")
    return all_entropy


# ============================================================================
# LOOKUP HELPERS
# Convenience functions for filtering EntropyRecord lists.
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
# EntropyRecords saved as .npz for multi-model comparison plotting.
# ============================================================================

def save_entropy_records(records: list, path) -> None:
    """Save a list of EntropyRecords to a single .npz file."""
    n = len(records)
    surfaces   = np.stack([r.surface    for r in records], axis=0)  # [n, layers, seq]
    prompts    = np.array([r.prompt     for r in records], dtype=object)
    model_names= np.array([r.model_name for r in records], dtype=object)
    hook_types = np.array([r.hook_type  for r in records], dtype=object)
    norm_keys  = np.array([r.norm_key   for r in records], dtype=object)
    alphas     = np.array([r.alpha      for r in records], dtype=np.float32)
    d_models   = np.array([r.d_model    for r in records], dtype=np.int32)
    roles      = np.array([r.role      or "" for r in records], dtype=object)
    categories = np.array([r.category  or "" for r in records], dtype=object)
    pair_ids   = np.array([r.pair_id   or "" for r in records], dtype=object)

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
    print(f"  ✓ Saved {n} EntropyRecords to {path}")


def load_entropy_records(path) -> list:
    """Load a list of EntropyRecords from a .npz file."""
    d = np.load(path, allow_pickle=True)
    n = len(d["prompts"])
    records = []
    for i in range(n):
        # str_tokens not serialized at corpus level (use prompt for display)
        records.append(EntropyRecord(
            prompt     = str(d["prompts"][i]),
            str_tokens = [],        # not stored at corpus level
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

def print_summary(entropy_records: list, alphas: list) -> None:
    """Print mean trajectory summary for energy normalization at each alpha."""
    print("\n" + "=" * 60)
    print("ENTROPY TRAJECTORY SUMMARY (energy norm, final token position)")
    print("=" * 60)

    for alpha in alphas:
        subset = filter_records(entropy_records, norm_key="energy", alpha=alpha)
        label  = "Shannon" if abs(alpha - 1.0) < 1e-6 else f"Rényi α={alpha}"
        print(f"\n  {label}:")

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
