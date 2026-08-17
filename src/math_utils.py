"""src/math_utils.py — Shared mathematical utilities.

Stateless functions that depend only on torch/numpy and are used by more
than one compute module. No model access, no file I/O, no plotting.
"""

import torch


# ============================================================================
# DEVICE POLICY
# Single source of truth for where linear algebra runs.
#
# torch.linalg.svd is unstable on MPS for large matrices, so MPS tensors are
# pinned to CPU before decomposition. CUDA has no such problem and is much
# faster for the [vocab_size, d_model] matmuls in the logit-lens and ablation
# hot paths, so CUDA tensors stay on device.
#
# On CPU and MPS these helpers are no-ops relative to the previous
# unconditional .cpu() behavior — the local numerics are unchanged.
# ============================================================================

def is_mps(t: torch.Tensor) -> bool:
    """True if the tensor lives on an MPS device."""
    return t.device.type == "mps"


def svd_device(t: torch.Tensor) -> torch.Tensor:
    """
    Move a tensor to a device where torch.linalg.svd is numerically stable.

    MPS -> cpu (svd is unstable there for large matrices; see CLAUDE.md).
    CPU, CUDA -> unchanged.

    Args:
        t: any tensor destined for torch.linalg.svd
    Returns:
        the tensor on a device safe for decomposition
    """
    return t.cpu() if is_mps(t) else t


def compute_device(t: torch.Tensor) -> torch.Tensor:
    """
    Move a tensor to the device where the entropy/ablation hot loops should run.

    MPS -> cpu, matching the project's existing MPS-avoidance policy.
    CPU, CUDA -> unchanged, so CUDA keeps activations and W_U co-resident
    on the GPU instead of round-tripping every layer through host memory.

    Args:
        t: activation, unembedding, or projection tensor
    Returns:
        the tensor on the appropriate compute device
    """
    return t.cpu() if is_mps(t) else t


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


def compute_wu_svd(W_U: torch.Tensor) -> torch.Tensor:
    """
    Compute the right singular vectors of W_U for subspace decomposition.

    W_U has shape [d_model, vocab_size]. We take the SVD of W_U.T
    (shape [vocab_size, d_model]) to get the right singular vectors Vh,
    whose rows are orthonormal basis vectors for d_model space, ordered
    by how much variance in W_U each direction explains.

    Args:
        W_U:  unembedding matrix, shape [d_model, vocab_size],
              obtained from model.W_U.detach() in the workflow layer

    Returns:
        Vh:   right singular vectors, shape [d_model, d_model]
              Row i is the i-th principal direction of W_U, ordered by
              decreasing singular value.

    Note: pins to .cpu() before SVD on MPS — torch.linalg.svd is unstable
    there for large matrices. CUDA is stable and stays on device.
    """
    _, _, Vh = torch.linalg.svd(svd_device(W_U.T.float()), full_matrices=False)
    return Vh
