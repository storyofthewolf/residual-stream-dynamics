"""ck_spectrum_compute.py — c_k spectrum computation over ActivationRecords.

Consumes ActivationRecords from extraction.py.
Produces CkRecords for consumption by ck_spectrum_plots.py.

Pipeline position:
    extraction.py → COMPUTATION (this file) → ck_spectrum_plots.py

The c_k spectrum decomposes the residual stream into the W_U singular basis:

    c_k = σ_k · (r · v_k)

where r is the residual stream vector at a given (layer, token), v_k is the
k-th right singular vector of W_U (row k of Vh), and σ_k is the k-th singular
value of W_U.

This is an exact decomposition: logits = W_U @ r = Σ_k c_k · u_k.
The spectrum is ordered by descending σ_k, same ordering as Vh rows.

Computation entry point:
    compute_ck_spectrum(record, S, Vh)  → CkRecord

SVD helpers (shared with entropy_compute.py paths):
    compute_wu_svd_full(W_U)  → (S, Vh)

Serialization:
    save_ck_records(records, path)
    load_ck_records(path)      → list[CkRecord]

CkRecord fields:
    ck_spectrum     : np.ndarray  shape [n_layers, seq_len, d_model]
    singular_values : np.ndarray  shape [d_model], descending σ_k
    prompt          : str
    str_tokens      : list[str]
    model_name      : str
    hook_type       : str
    n_layers        : int
    seq_len         : int
    d_model         : int
    pair_id/role/category  — passed through from ActivationRecord; None for
                             single-prompt runs
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

from extraction import ActivationRecord
from math_utils import svd_device, compute_device


# ============================================================================
# SVD HELPERS
# ============================================================================

def compute_wu_svd_full(W_U: torch.Tensor) -> tuple:
    """
    Compute the full SVD of W_U, returning both singular values and right
    singular vectors. Used by compute_ck_spectrum(); callers that only need
    Vh should use entropy_compute.compute_wu_svd() instead.

    Args:
        W_U:  unembedding matrix, shape [d_model, vocab_size]

    Returns:
        S:    singular values, shape [d_model], descending order
        Vh:   right singular vectors, shape [d_model, d_model]
              Row k is the k-th principal direction of W_U (v_k).
    """
    _, S, Vh = torch.linalg.svd(svd_device(W_U.T.float()), full_matrices=False)
    return S, Vh


# ============================================================================
# CK RECORD
# ============================================================================

@dataclass
class CkRecord:
    """
    Stores the c_k spectrum for one prompt across all layers and token positions.

    c_k = σ_k · (r · v_k)  where:
        r    is the residual stream vector at a given (layer, token)
        v_k  is the k-th right singular vector of W_U (row k of Vh)
        σ_k  is the k-th singular value of W_U

    This is an exact decomposition: logits = W_U @ r = Σ_k c_k · u_k.
    The spectrum is ordered by descending σ_k (same ordering as Vh rows).

    ck_spectrum shape: [n_layers, seq_len, d_model]
        ck_spectrum[layer, t, k]  — the k-th coefficient at (layer, token t)
        ck_spectrum[:, -1, :]     — spectrum at final token, all layers
        ck_spectrum[layer, :, :]  — full spectrum at fixed layer

    singular_values shape: [d_model] — the σ_k values, descending.
    Stored on the record so plots can recover the σ_k axis without re-running SVD.
    """
    # Core data
    ck_spectrum:     np.ndarray    # shape [n_layers, seq_len, d_model]
    singular_values: np.ndarray    # shape [d_model], descending σ_k

    # Standard metadata (mirrors ActivationRecord fields)
    prompt:      str
    str_tokens:  list
    model_name:  str
    hook_type:   str
    n_layers:    int
    seq_len:     int
    d_model:     int

    # Corpus metadata (None for single-prompt runs)
    pair_id:     Optional[str] = None
    role:        Optional[str] = None
    category:    Optional[str] = None

    def final_token_spectrum(self) -> np.ndarray:
        """c_k spectrum at the final token, all layers. Shape: [n_layers, d_model]."""
        return self.ck_spectrum[:, -1, :]

    def layer_spectrum(self, layer: int) -> np.ndarray:
        """c_k spectrum at a fixed layer, all token positions. Shape: [seq_len, d_model]."""
        return self.ck_spectrum[layer, :, :]


# ============================================================================
# COMPUTATION
# ============================================================================

def compute_ck_spectrum(
    record: ActivationRecord,
    S:      torch.Tensor,
    Vh:     torch.Tensor,
) -> CkRecord:
    """
    Compute the c_k spectrum for all layers and token positions.

    For each (layer, token), the residual stream vector r is projected onto
    the W_U singular basis:
        coeffs = Vh @ r          # [d_model] — projections onto v_k directions
        c_k    = S * coeffs      # [d_model] — scaled by singular values

    This is an exact decomposition: W_U @ r = (U @ diag(S) @ Vh) @ r
    = U @ (S * (Vh @ r)) = Σ_k c_k · u_k.

    The computation is vectorized over all token positions simultaneously
    per layer: Vh @ r_mat gives [d_model, seq_len], then scaled by S.

    Args:
        record: ActivationRecord from extraction.py
        S:      singular values, shape [d_model], from compute_wu_svd_full()
        Vh:     right singular vectors, shape [d_model, d_model],
                from compute_wu_svd_full(). Row k = v_k.

    Returns:
        CkRecord with ck_spectrum of shape [n_layers, seq_len, d_model]
    """
    n_layers = record.n_layers
    seq_len  = record.seq_len
    d_model  = record.d_model

    # Device note: one [d_model, d_model] @ [d_model, seq_len] matmul per
    # layer with a single transfer back per layer — no per-element sync — so
    # this path benefits from CUDA. On MPS it pins to CPU as before.
    S_dev  = compute_device(S.float())
    Vh_dev = compute_device(Vh.float())
    dev    = Vh_dev.device

    S_cpu = S.float().cpu()   # kept on CPU for the returned record

    ck = np.zeros((n_layers, seq_len, d_model), dtype=np.float32)

    with torch.no_grad():
        for layer in range(n_layers):
            # r_mat: [seq_len, d_model] → transpose to [d_model, seq_len]
            r_mat = torch.from_numpy(
                record.activations[layer, :, :]
            ).float().T.to(dev)                            # [d_model, seq_len]

            # coeffs: [d_model, seq_len] — projection of each token onto each v_k
            coeffs = Vh_dev @ r_mat                        # [d_model, seq_len]

            # c_k: scale each row k by σ_k
            ck_layer = S_dev.unsqueeze(1) * coeffs         # [d_model, seq_len]

            # store as [seq_len, d_model]
            ck[layer] = ck_layer.T.cpu().numpy()

    return CkRecord(
        ck_spectrum     = ck,
        singular_values = S_cpu.numpy(),
        prompt      = record.prompt,
        str_tokens  = record.str_tokens,
        model_name  = record.model_name,
        hook_type   = record.hook_type,
        n_layers    = n_layers,
        seq_len     = seq_len,
        d_model     = d_model,
        pair_id     = record.pair_id,
        role        = record.role,
        category    = record.category,
    )


# ============================================================================
# SERIALIZATION
# ============================================================================

def save_ck_records(records: list, path) -> None:
    """Save a list of CkRecords to a single .npz file.

    Variable seq_len across prompts is handled by NaN-padding ck_spectrum
    to a common [max_layers, max_seq_len, d_model] shape.
    """
    n = len(records)
    if n == 0:
        print("  No CkRecords to save.")
        return

    max_layers  = max(r.n_layers for r in records)
    max_seq_len = max(r.seq_len  for r in records)
    d_model     = records[0].d_model

    padded = np.full((n, max_layers, max_seq_len, d_model), np.nan, dtype=np.float32)
    n_layers_arr = np.zeros(n, dtype=np.int32)
    seq_lens_arr = np.zeros(n, dtype=np.int32)

    for i, r in enumerate(records):
        padded[i, :r.n_layers, :r.seq_len, :] = r.ck_spectrum
        n_layers_arr[i] = r.n_layers
        seq_lens_arr[i] = r.seq_len

    np.savez(
        path,
        ck_spectrum      = padded,
        singular_values  = np.stack([r.singular_values for r in records]),
        n_layers         = n_layers_arr,
        seq_lens         = seq_lens_arr,
        prompts          = np.array([r.prompt      for r in records], dtype=object),
        str_tokens       = np.array([r.str_tokens  for r in records], dtype=object),
        model_names      = np.array([r.model_name  for r in records], dtype=object),
        hook_types       = np.array([r.hook_type   for r in records], dtype=object),
        d_models         = np.array([r.d_model     for r in records], dtype=np.int32),
        pair_ids         = np.array([r.pair_id  or "" for r in records], dtype=object),
        roles            = np.array([r.role      or "" for r in records], dtype=object),
        categories       = np.array([r.category or "" for r in records], dtype=object),
    )
    print(f"  Saved {n} CkRecords to {path}")


def load_ck_records(path) -> list:
    """Load a list of CkRecords from a .npz file."""
    d = np.load(path, allow_pickle=True)
    n = len(d["prompts"])
    has_str_tokens = "str_tokens" in d
    records = []
    for i in range(n):
        nl = int(d["n_layers"][i])
        sl = int(d["seq_lens"][i])
        records.append(CkRecord(
            ck_spectrum     = d["ck_spectrum"][i, :nl, :sl, :],
            singular_values = d["singular_values"][i],
            prompt      = str(d["prompts"][i]),
            str_tokens  = list(d["str_tokens"][i]) if has_str_tokens else [],
            model_name  = str(d["model_names"][i]),
            hook_type   = str(d["hook_types"][i]),
            n_layers    = nl,
            seq_len     = sl,
            d_model     = int(d["d_models"][i]),
            pair_id     = str(d["pair_ids"][i])    or None,
            role        = str(d["roles"][i])        or None,
            category    = str(d["categories"][i])  or None,
        ))
    return records
