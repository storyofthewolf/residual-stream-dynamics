"""src/math_utils.py — Shared mathematical utilities.

Stateless functions that depend only on torch/numpy and are used by more
than one compute module. No model access, no file I/O, no plotting.
"""

import torch


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

    Note: forces .cpu() before SVD — torch.linalg.svd is unstable on MPS
    for large matrices.
    """
    _, _, Vh = torch.linalg.svd(W_U.T.float().cpu(), full_matrices=False)
    return Vh
