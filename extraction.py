"""extraction.py — Forward pass and activation extraction.

Defines ActivationRecord, the fundamental data structure passed through
the analysis pipeline. All other modules consume ActivationRecords;
none of them run forward passes.

Pipeline position:
    EXTRACTION (this file) → entropy_compute.py → entropy_plots.py

ActivationRecord fields:
    prompt      : str                  — original prompt string
    str_tokens  : list[str]            — tokenized string labels
    model_name  : str                  — e.g. "gpt2-small"
    hook_type   : str                  — short name e.g. "resid_post", "attn_out"
    hook_pattern: str                  — full template e.g. "blocks.{layer}.hook_resid_post"
    activations : np.ndarray           — shape [n_layers, seq_len, d_model]
    d_model     : int                  — actual last dimension (768, 3072, etc.)
    n_layers    : int                  — number of transformer layers
    seq_len     : int                  — number of token positions (including BOS)
    has_resid_mid: bool                — whether this model exposes hook_resid_mid
    pair_id     : str | None           — corpus pair identifier
    role        : str | None           — "base" or "contrast"
    category    : str | None           — corpus category

Key function:
    extract_activations(model, prompt, hook_types, model_name, device, corpus_entry)
        → dict[str, ActivationRecord]

    Accepts a LIST of hook_types, runs ONE forward pass, returns one
    ActivationRecord per hook type. This guarantees that comparing
    resid_post vs attn_out vs mlp_out always uses identical forward pass data.

Supported hook_types (short names → full patterns):
    "resid_pre"   → blocks.{layer}.hook_resid_pre
    "resid_mid"   → blocks.{layer}.hook_resid_mid   (GPT-2, Gemma-2, Llama only)
    "resid_post"  → blocks.{layer}.hook_resid_post
    "attn_out"    → blocks.{layer}.hook_attn_out
    "mlp_out"     → blocks.{layer}.hook_mlp_out
    "mlp_pre"     → blocks.{layer}.mlp.hook_pre     (internal MLP, d_model=4×hidden)
    "mlp_post"    → blocks.{layer}.mlp.hook_post    (internal MLP, d_model=4×hidden)

    This module is the data I/O layer — it reads from the neural network
    residual stream and writes to a structured data record.
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)


# ============================================================================
# HOOK TYPE REGISTRY
# Maps short human-readable names to TransformerLens hook name templates.
# Add new hook types here as needed — no other changes required.
# ============================================================================

HOOK_TYPES = {
    "resid_pre":  "blocks.{layer}.hook_resid_pre",
    "resid_mid":  "blocks.{layer}.hook_resid_mid",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "attn_out":   "blocks.{layer}.hook_attn_out",
    "mlp_out":    "blocks.{layer}.hook_mlp_out",
    "mlp_pre":    "blocks.{layer}.mlp.hook_pre",
    "mlp_post":   "blocks.{layer}.mlp.hook_post",
}

# Human-readable labels for each hook type (used in plot titles)
HOOK_LABELS = {
    "resid_pre":  "Residual pre-attention",
    "resid_mid":  "Residual post-attention / pre-MLP",
    "resid_post": "Residual post-MLP",
    "attn_out":   "Attention output",
    "mlp_out":    "MLP output",
    "mlp_pre":    "MLP internal pre-activation",
    "mlp_post":   "MLP internal post-activation",
}

# Hook types that live in residual stream space (d_model = hidden_size).
# mlp_pre and mlp_post are 4× wider — flag this for downstream awareness.
RESIDUAL_STREAM_HOOKS = {"resid_pre", "resid_mid", "resid_post", "attn_out", "mlp_out"}
MLP_INTERNAL_HOOKS    = {"mlp_pre", "mlp_post"}


# ============================================================================
# ACTIVATION RECORD
# The fundamental data structure of the pipeline.
# Produced by extraction.py, consumed by entropy_compute.py and entropy_plots.py.
# ============================================================================

@dataclass
class ActivationRecord:
    """
    Stores raw activations for ONE hook type across all layers and token positions.

    Shape of activations: [n_layers, seq_len, d_model]
        n_layers : number of transformer blocks
        seq_len  : token positions including BOS (position 0)
        d_model  : actual activation dimension for this hook type
                   (768 for residual-stream hooks in GPT-2 small,
                    3072 for mlp_pre/mlp_post in GPT-2 small)

    One ActivationRecord per hook type. Multiple hook types from the same
    forward pass are returned as dict[str, ActivationRecord] by
    extract_activations().

    Optional corpus fields (pair_id, role, category) are None for
    single-prompt exploratory runs and populated for corpus analysis.
    """
    # Core identity
    prompt:       str
    str_tokens:   list
    model_name:   str
    hook_type:    str           # short name e.g. "resid_post"
    hook_pattern: str           # full template e.g. "blocks.{layer}.hook_resid_post"

    # Activation tensor
    activations:  np.ndarray   # shape: [n_layers, seq_len, d_model]
    d_model:      int           # actual last dimension
    n_layers:     int
    seq_len:      int

    # Model capability flag
    has_resid_mid: bool = False

    # Optional corpus metadata
    pair_id:      Optional[str] = None
    role:         Optional[str] = None
    category:     Optional[str] = None

    def label(self) -> str:
        """Human-readable label for this hook type, used in plot titles."""
        return HOOK_LABELS.get(self.hook_type, self.hook_type)

    def is_residual_stream(self) -> bool:
        """True if this hook lives in residual stream space (not MLP-internal)."""
        return self.hook_type in RESIDUAL_STREAM_HOOKS

    def token_slice(self, skip_bos: bool = True) -> tuple:
        """
        Return (activations_slice, str_tokens_slice) with optional BOS removal.
        BOS is position 0 and identified by the '<|endoftext|>' token label.
        """
        if skip_bos and self.str_tokens[0] == '<|endoftext|>':
            return self.activations[:, 1:, :], self.str_tokens[1:]
        return self.activations, self.str_tokens


# ============================================================================
# SERIALIZATION
# ActivationRecords can be saved to / loaded from .npz files for
# later multi-model comparison plotting without re-running forward passes.
# ============================================================================

def save_activation_record(record: ActivationRecord, path) -> None:
    """Save an ActivationRecord to a .npz file."""
    np.savez(
        path,
        activations  = record.activations,
        prompt       = record.prompt,
        str_tokens   = np.array(record.str_tokens, dtype=object),
        model_name   = record.model_name,
        hook_type    = record.hook_type,
        hook_pattern = record.hook_pattern,
        d_model      = record.d_model,
        n_layers     = record.n_layers,
        seq_len      = record.seq_len,
        has_resid_mid= record.has_resid_mid,
        pair_id      = record.pair_id  if record.pair_id  else "",
        role         = record.role     if record.role     else "",
        category     = record.category if record.category else "",
    )


def load_activation_record(path) -> ActivationRecord:
    """Load an ActivationRecord from a .npz file."""
    d = np.load(path, allow_pickle=True)
    return ActivationRecord(
        activations   = d["activations"],
        prompt        = str(d["prompt"]),
        str_tokens    = list(d["str_tokens"]),
        model_name    = str(d["model_name"]),
        hook_type     = str(d["hook_type"]),
        hook_pattern  = str(d["hook_pattern"]),
        d_model       = int(d["d_model"]),
        n_layers      = int(d["n_layers"]),
        seq_len       = int(d["seq_len"]),
        has_resid_mid = bool(d["has_resid_mid"]),
        pair_id       = str(d["pair_id"])   or None,
        role          = str(d["role"])      or None,
        category      = str(d["category"]) or None,
    )


# ============================================================================
# CORE EXTRACTION FUNCTION
# Runs ONE forward pass and returns one ActivationRecord per requested hook.
# ============================================================================

def extract_activations(
    model,
    prompt:       str,
    hook_types:   list,
    model_name:   str,
    device:       str,
    corpus_entry: Optional[dict] = None,
) -> dict:
    """
    Run one forward pass and extract activations for all requested hook types.

    Args:
        model:        loaded TransformerLens HookedTransformer
        prompt:       input string
        hook_types:   list of short hook names e.g. ["resid_post", "attn_out"]
                      see HOOK_TYPES for all supported values
        model_name:   string identifier e.g. "gpt2-small"
        device:       torch device string
        corpus_entry: optional dict with keys pair_id, role, category, description
                      if provided, these are stored in the returned records

    Returns:
        dict mapping hook_type string → ActivationRecord
        e.g. {"resid_post": ActivationRecord(...), "attn_out": ActivationRecord(...)}

    Raises:
        ValueError: if an unsupported hook_type is requested
        KeyError:   if a requested hook is not present in the model cache
                    (e.g. "resid_mid" on Pythia)

    FORTRAN analogy:
        This function is the equivalent of a single model integration step
        that writes all diagnostic output variables simultaneously.
        Requesting multiple hooks does not cost additional forward passes —
        the cache contains everything after one pass, exactly as a model
        time step produces all output fields at once.
    """
    # Validate hook types before running the forward pass
    for ht in hook_types:
        if ht not in HOOK_TYPES:
            raise ValueError(
                f"Unknown hook_type '{ht}'. "
                f"Supported: {sorted(HOOK_TYPES.keys())}"
            )

    # Tokenize
    tokens    = model.to_tokens(prompt, prepend_bos=True).to(device)
    seq_len   = tokens.shape[1]
    str_tokens = model.to_str_tokens(prompt)
    n_layers  = model.cfg.n_layers

    # Detect whether this model exposes hook_resid_mid
    # Run a minimal probe on the first token to check cache keys
    with torch.no_grad():
        _, probe_cache = model.run_with_cache(tokens[:, :1])
    has_resid_mid = "blocks.0.hook_resid_mid" in probe_cache

    # Check that all requested hooks are available in this model
    for ht in hook_types:
        test_key = HOOK_TYPES[ht].format(layer=0)
        if test_key not in probe_cache:
            raise KeyError(
                f"Hook '{ht}' ('{test_key}') is not available for model "
                f"'{model_name}'. Available residual hooks: "
                f"resid_pre, resid_post, attn_out, mlp_out"
                + (", resid_mid" if has_resid_mid else "")
            )

    # Single forward pass — cache contains all hooks simultaneously
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    # Extract one ActivationRecord per requested hook type
    records = {}

    for ht in hook_types:
        pattern = HOOK_TYPES[ht]

        # Collect activations across all layers into [n_layers, seq_len, d_model]
        # FORTRAN analogy: filling a 3D array ACTIV(layer, pos, d_model)
        layer_arrays = []
        for layer in range(n_layers):
            hook_name = pattern.format(layer=layer)
            # cache tensor shape: [batch=1, seq_len, d_model]
            act = cache[hook_name][0].float().cpu().numpy()  # [seq_len, d_model]
            layer_arrays.append(act)

        activations = np.stack(layer_arrays, axis=0)  # [n_layers, seq_len, d_model]
        d_model     = activations.shape[2]

        # Populate optional corpus metadata if provided
        pair_id  = corpus_entry.get("pair_id")  if corpus_entry else None
        role     = corpus_entry.get("role")     if corpus_entry else None
        category = corpus_entry.get("category") if corpus_entry else None

        records[ht] = ActivationRecord(
            prompt        = prompt,
            str_tokens    = str_tokens,
            model_name    = model_name,
            hook_type     = ht,
            hook_pattern  = pattern,
            activations   = activations,
            d_model       = d_model,
            n_layers      = n_layers,
            seq_len       = seq_len,
            has_resid_mid = has_resid_mid,
            pair_id       = pair_id,
            role          = role,
            category      = category,
        )

    return records


# ============================================================================
# CORPUS EXTRACTION
# Convenience wrapper for running extract_activations over a full corpus.
# ============================================================================

def extract_corpus(
    model,
    corpus:       list,
    hook_types:   list,
    model_name:   str,
    device:       str,
    category_filter: Optional[str] = None,
) -> dict:
    """
    Run extract_activations over every entry in a corpus JSON.

    Args:
        model:           loaded TransformerLens model
        corpus:          list of dicts from corpus_gen.py
        hook_types:      list of hook type short names
        model_name:      string identifier
        device:          torch device
        category_filter: if provided, only process entries with this category

    Returns:
        dict mapping hook_type → list[ActivationRecord]
        e.g. {"resid_post": [record_0, record_1, ...],
               "attn_out":  [record_0, record_1, ...]}

    All records at index i across hook types correspond to the same prompt
    and the same forward pass.
    """
    filtered = corpus
    if category_filter:
        filtered = [e for e in corpus if e["category"] == category_filter]
        print(f"  Filtered to category '{category_filter}': {len(filtered)} prompts")

    # Initialize output lists per hook type
    all_records = {ht: [] for ht in hook_types}

    for i, entry in enumerate(filtered):
        prompt = entry["prompt"]
        records = extract_activations(
            model, prompt, hook_types, model_name, device,
            corpus_entry=entry
        )
        for ht in hook_types:
            all_records[ht].append(records[ht])

        if (i + 1) % 10 == 0 or (i + 1) == len(filtered):
            print(f"  Extracted {i+1}/{len(filtered)} prompts...")

    return all_records
