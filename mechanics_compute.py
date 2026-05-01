"""mechanics_compute.py — Mechanical quantity computation over ActivationRecords.

Consumes ActivationRecords from extraction.py.
Produces MechanicsRecords for consumption by mechanics_plots.py.

Pipeline position:
    extraction.py → COMPUTATION (this file) → mechanics_plots.py

Interprets the residual stream as a discrete particle trajectory in d_model-
dimensional space:

    Position:     X_l          — residual stream vector at layer l
    Velocity:     ΔX_l         — update vector (displacement per layer)
    Acceleration: ΔV_l         — change in velocity between layers

Five scalar curves are computed from a [n_layers, d_model] array of layer vectors:

    speed                  — ||ΔX_l||₂                  shape [n_layers-1]
    acceleration_magnitude — ||ΔV_l||₂                  shape [n_layers-2]
    cosine_sim_state       — cos(X_l, X_{l+1})           shape [n_layers-1]
    cosine_sim_update_state  — cos(ΔX_l, X_l)           shape [n_layers-1]
    cosine_sim_update_update — cos(ΔX_l, ΔX_{l+1})      shape [n_layers-2]

All computation is pure numpy — no torch, no model access.

MechanicsRecord fields:
    prompt      : str
    model_name  : str
    pair_id     : str | None
    role        : str | None
    category    : str | None
    speed                  : np.ndarray  [n_layers-1]
    acceleration_magnitude : np.ndarray  [n_layers-2]
    cosine_sim_state       : np.ndarray  [n_layers-1]
    cosine_sim_update_state  : np.ndarray  [n_layers-1]
    cosine_sim_update_update : np.ndarray  [n_layers-2]

Public function:
    compute_mechanics(record, token_pos=-1) -> MechanicsRecord

Serialization:
    save_mechanics_records(records, path)
    load_mechanics_records(path) -> list[MechanicsRecord]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from extraction import ActivationRecord


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class MechanicsRecord:
    """
    Stores five scalar mechanical curves for one prompt at one token position.

    Produced by compute_mechanics(). All arrays are pure numpy — no torch.

    Layer-indexing note:
        X           n_layers entries:    layers 0 .. n_layers-1
        velocity    n_layers-1 entries:  transitions 0→1 .. (n-2)→(n-1)
        acceleration n_layers-2 entries: one fewer again

    pair_id / role / category are passed through from the source ActivationRecord
    and are None for single-prompt exploratory runs.
    """
    # Identity
    prompt:     str
    model_name: str

    # Corpus metadata (None for single-prompt runs)
    pair_id:    Optional[str] = None
    role:       Optional[str] = None
    category:   Optional[str] = None

    # Mechanical curves
    speed:                   np.ndarray = None   # [n_layers-1]
    acceleration_magnitude:  np.ndarray = None   # [n_layers-2]
    cosine_sim_state:        np.ndarray = None   # [n_layers-1]
    cosine_sim_update_state:  np.ndarray = None  # [n_layers-1]
    cosine_sim_update_update: np.ndarray = None  # [n_layers-2]

    @property
    def n_transitions(self) -> int:
        return len(self.speed)

    @property
    def n_accelerations(self) -> int:
        return len(self.acceleration_magnitude)


# ============================================================================
# PRIVATE COMPUTE FUNCTIONS
# All operate on layer_vecs: np.ndarray of shape [n_layers, d_model].
# ============================================================================

def _compute_velocity(layer_vecs: np.ndarray) -> np.ndarray:
    """ΔX_l = X_{l+1} - X_l. Shape: [n_layers-1, d_model]."""
    return layer_vecs[1:] - layer_vecs[:-1]


def _compute_speed(velocity: np.ndarray) -> np.ndarray:
    """||ΔX_l||₂ — L2 norm of each displacement vector. Shape: [n_layers-1]."""
    return np.linalg.norm(velocity, axis=1)


def _compute_acceleration_magnitude(velocity: np.ndarray) -> np.ndarray:
    """||ΔV_l||₂ — norm of velocity change. Shape: [n_layers-2]."""
    acceleration = velocity[1:] - velocity[:-1]   # [n_layers-2, d_model]
    return np.linalg.norm(acceleration, axis=1)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. a, b: [n, d]. Returns: [n]."""
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    denom = (norm_a * norm_b).clip(min=1e-12)
    return (a * b).sum(axis=1) / denom.squeeze(1)


def _compute_cosine_sim_state(layer_vecs: np.ndarray) -> np.ndarray:
    """cos(X_l, X_{l+1}) — curvature of the trajectory. Shape: [n_layers-1]."""
    return _cosine(layer_vecs[:-1], layer_vecs[1:])


def _compute_cosine_sim_update_state(layer_vecs: np.ndarray,
                                     velocity: np.ndarray) -> np.ndarray:
    """cos(ΔX_l, X_l) — alignment of update with current state. Shape: [n_layers-1]."""
    return _cosine(velocity, layer_vecs[:-1])


def _compute_cosine_sim_update_update(velocity: np.ndarray) -> np.ndarray:
    """cos(ΔX_l, ΔX_{l+1}) — coherence of consecutive updates. Shape: [n_layers-2]."""
    return _cosine(velocity[:-1], velocity[1:])


# ============================================================================
# PUBLIC COMPUTE FUNCTION
# ============================================================================

def compute_mechanics(
    record:    ActivationRecord,
    token_pos: int = -1,
) -> MechanicsRecord:
    """
    Compute five mechanical scalar curves for one token position.

    Slices record.activations[:, token_pos, :] to get shape [n_layers, d_model],
    then computes all five quantities in pure numpy.

    Args:
        record:    ActivationRecord from extraction.py
        token_pos: which token position to analyse (default -1 = final token)

    Returns:
        MechanicsRecord with five scalar curve arrays
    """
    layer_vecs = record.activations[:, token_pos, :].astype(np.float64)
    # layer_vecs: [n_layers, d_model]

    velocity = _compute_velocity(layer_vecs)

    return MechanicsRecord(
        prompt     = record.prompt,
        model_name = record.model_name,
        pair_id    = record.pair_id,
        role       = record.role,
        category   = record.category,
        speed                    = _compute_speed(velocity),
        acceleration_magnitude   = _compute_acceleration_magnitude(velocity),
        cosine_sim_state         = _compute_cosine_sim_state(layer_vecs),
        cosine_sim_update_state  = _compute_cosine_sim_update_state(layer_vecs, velocity),
        cosine_sim_update_update = _compute_cosine_sim_update_update(velocity),
    )


# ============================================================================
# SERIALIZATION
# Save/load MechanicsRecords to .npz, following entropy_compute.py conventions.
# Variable-length arrays are NaN-padded to a common shape and trimmed on load.
# ============================================================================

def save_mechanics_records(records: list, path) -> None:
    """Save a list of MechanicsRecords to a single .npz file.

    Pads to the maximum length across records (different models may have
    different n_layers). Original lengths are stored for trimming on load.
    """
    n = len(records)

    max_trans = max(len(r.speed)                   for r in records)
    max_accel = max(len(r.acceleration_magnitude)  for r in records)

    speed_pad   = np.full((n, max_trans), np.nan, dtype=np.float32)
    accel_pad   = np.full((n, max_accel), np.nan, dtype=np.float32)
    cos_st_pad  = np.full((n, max_trans), np.nan, dtype=np.float32)
    cos_us_pad  = np.full((n, max_trans), np.nan, dtype=np.float32)
    cos_uu_pad  = np.full((n, max_accel), np.nan, dtype=np.float32)

    n_trans = np.zeros(n, dtype=np.int32)
    n_accel = np.zeros(n, dtype=np.int32)

    for i, r in enumerate(records):
        nt = len(r.speed)
        na = len(r.acceleration_magnitude)
        speed_pad[i, :nt]  = r.speed
        accel_pad[i, :na]  = r.acceleration_magnitude
        cos_st_pad[i, :nt] = r.cosine_sim_state
        cos_us_pad[i, :nt] = r.cosine_sim_update_state
        cos_uu_pad[i, :na] = r.cosine_sim_update_update
        n_trans[i] = nt
        n_accel[i] = na

    np.savez(
        path,
        speed                    = speed_pad,
        acceleration_magnitude   = accel_pad,
        cosine_sim_state         = cos_st_pad,
        cosine_sim_update_state  = cos_us_pad,
        cosine_sim_update_update = cos_uu_pad,
        n_trans                  = n_trans,
        n_accel                  = n_accel,
        prompts                  = np.array([r.prompt     for r in records], dtype=object),
        model_names              = np.array([r.model_name for r in records], dtype=object),
        roles                    = np.array([r.role     or "" for r in records], dtype=object),
        categories               = np.array([r.category or "" for r in records], dtype=object),
        pair_ids                 = np.array([r.pair_id  or "" for r in records], dtype=object),
    )
    print(f"  Saved {n} MechanicsRecords to {path}")


def load_mechanics_records(path) -> list:
    """Load a list of MechanicsRecords from a .npz file.

    Trims NaN-padded arrays back to their original lengths using the stored
    n_trans and n_accel arrays.
    """
    d = np.load(path, allow_pickle=True)
    n = len(d["prompts"])

    records = []
    for i in range(n):
        nt = int(d["n_trans"][i])
        na = int(d["n_accel"][i])

        records.append(MechanicsRecord(
            prompt     = str(d["prompts"][i]),
            model_name = str(d["model_names"][i]),
            role       = str(d["roles"][i])      or None,
            category   = str(d["categories"][i]) or None,
            pair_id    = str(d["pair_ids"][i])   or None,
            speed                    = d["speed"][i, :nt].astype(np.float64),
            acceleration_magnitude   = d["acceleration_magnitude"][i, :na].astype(np.float64),
            cosine_sim_state         = d["cosine_sim_state"][i, :nt].astype(np.float64),
            cosine_sim_update_state  = d["cosine_sim_update_state"][i, :nt].astype(np.float64),
            cosine_sim_update_update = d["cosine_sim_update_update"][i, :na].astype(np.float64),
        ))
    return records
