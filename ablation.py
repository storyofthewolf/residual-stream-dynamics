"""ablation.py — Interventional ablation of the r⊥ orthogonal complement.

Consumes ActivationRecords from extraction.py.
Produces AblationRecords for consumption by ablation_plots.py.

Pipeline position (parallel to the entropy path):
    extraction.py → ABLATION (this file) → ablation_plots.py

Two ablation stages, both producing AblationRecords:

    compute_posthoc_ablation(record, W_U, ln_final, Vh, k_values, alpha)
        Post-hoc projection ablation. Takes already-extracted activations,
        projects each layer's residual stream onto the top-k W_U singular
        directions (keeping only r‖, zeroing r⊥), applies ln_final and W_U
        to get ablated token probabilities, and compares to the full
        (unablated) baseline. No forward pass required.
        Produces len(k_values) AblationRecords per call.

    compute_intervention_ablation(record, model, W_U, ln_final, Vh,
                                   k_values, intervention_layers, alpha)
        Single-layer forward pass intervention. Uses TransformerLens hooks
        to zero r⊥ at a specific layer during a live forward pass, then
        compares the final-layer token prediction to the clean baseline.
        Requires a live HookedTransformer model and one forward pass per
        (k, intervention_layer) combination.
        Produces len(k_values) × len(intervention_layers) AblationRecords.

SVD utilities (called from the workflow layer, not from ablation functions):
    compute_wu_svd(W_U)                         — returns Vh
    wu_explained_variance(Vh, W_U, k_values)    — returns {k: fraction}

Validation:
    validate_ablation(W_U, Vh, d_model)         — linear algebra checks

Scientific motivation:
    The k-sweep analysis in computation.py shows that the base-vs-contrast
    entropy difference in r⊥ persists robustly until k approaches full rank.
    This is a correlational finding. Ablation converts it to an interventional
    claim: if removing r⊥ degrades base prompt predictions more than contrast
    prompt predictions, then r⊥ carries computationally meaningful content
    that differs systematically between prompt types.

    This is the falsifiable behavioral prediction the paper requires.
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

from extraction import ActivationRecord
from computation import renyi_entropy


# ============================================================================
# ABLATION RECORD
# Output of both compute_posthoc_ablation() and
# compute_intervention_ablation(). One record per k value (posthoc)
# or per (k, intervention_layer) combination (intervention).
# ============================================================================

@dataclass
class AblationRecord:
    """
    Stores ablation results for one (prompt, k, ablation_type) combination.

    For posthoc ablation, the metric arrays have shape [n_layers] — one value
    per layer, measuring what happens when r⊥ is removed at that layer's
    activations and logits are recomputed locally.

    For intervention ablation, the metric arrays have shape [1] — a single
    value measuring the effect at the final layer after the model propagates
    through remaining layers with the ablated residual stream.

    FORTRAN analogy:
        Think of this as a derived type that holds the output fields of a
        diagnostic subroutine. The arrays are indexed like REAL :: kl(n_layers)
        for posthoc, or REAL :: kl(1) for intervention. The metadata fields
        are like integer/character tags that identify which experiment
        produced this record.
    """
    # Core results
    kl_divergence:      np.ndarray    # KL(probs_full || probs_ablated) per layer
    entropy_change:     np.ndarray    # H(probs_ablated) - H(probs_full) per layer
    top1_preserved:     np.ndarray    # bool: top predicted token unchanged
                                      # shape [n_layers] for posthoc, [1] for intervention

    # Experiment parameters
    k:                  int           # subspace rank used for projection
    ablation_type:      str           # "posthoc" or "intervention"
    intervention_layer: Optional[int] # None for posthoc; layer index for intervention

    # Standard metadata (mirror ActivationRecord fields)
    model_name:         str
    prompt:             str
    role:               str           # "base" or "contrast" (matches ActivationRecord.role)
    category:           Optional[str] # corpus category e.g. "pattern", "factual"
                                      # (matches ActivationRecord.category)
    hook_type:          str           # hook used during extraction (matches ActivationRecord)


# ============================================================================
# SVD UTILITIES
# Precomputed once per model in the workflow layer, passed into ablation
# functions. These are NOT called inside the ablation functions themselves.
# ============================================================================

def compute_wu_svd(W_U: torch.Tensor) -> torch.Tensor:
    """
    Compute SVD of W_U and return right singular vectors Vh.

    Args:
        W_U: unembedding matrix, shape [d_model, vocab_size]

    Returns:
        Vh: right singular vectors, shape [d_model, d_model]
            rows are ordered by descending singular value
            i.e. Vh[0] is the direction W_U is most sensitive to

    Note: computed on CPU regardless of W_U device, due to MPS
    instability with torch.linalg.svd on large matrices.

    FORTRAN analogy: computing the eigenvectors of a covariance matrix
    for EOF analysis. Done once; reused for all subsequent projections.
    Like calling DSYEV on the covariance matrix at the start of a run
    and storing the eigenvectors for repeated use in the time loop.
    """
    U, S, Vh = torch.linalg.svd(W_U.T.float().cpu(), full_matrices=False)
    return Vh    # shape [d_model, d_model]


def wu_explained_variance(
    Vh:       torch.Tensor,
    W_U:      torch.Tensor,
    k_values: list[int],
) -> dict:
    """
    Compute fraction of W_U Frobenius norm explained by top-k singular
    directions, for each k in k_values.

    Args:
        Vh:       precomputed right singular vectors from compute_wu_svd
        W_U:      original unembedding matrix [d_model, vocab_size]
        k_values: list of rank values to evaluate

    Returns:
        dict mapping k -> explained variance fraction (0.0 to 1.0)

    Use this to choose k values in terms of explained variance rather
    than raw integer rank. Report in paper as percentage thresholds
    (e.g. k corresponding to 50%, 75%, 90%, 95%, 99% explained variance).

    FORTRAN analogy: computing cumulative explained variance from
    eigenvalues: sum(lambda_1..k) / sum(lambda_all). The squared
    singular values play the role of eigenvalues of the covariance matrix.
    """
    _, S, _ = torch.linalg.svd(W_U.T.float().cpu(), full_matrices=False)
    total = (S**2).sum()
    return {k: ((S[:k]**2).sum() / total).item() for k in k_values}


# ============================================================================
# VALIDATION
# Run before any ablation experiment to confirm the linear algebra is sound.
# ============================================================================

def validate_ablation(
    W_U:     torch.Tensor,
    Vh:      torch.Tensor,
    d_model: int,
) -> None:
    """
    Validate the projection decomposition before running ablation experiments.

    Checks:
        1. Reconstruction: r‖ + r⊥ = r (to floating point tolerance)
        2. Orthogonality:  r‖ · r⊥ ≈ 0
        3. Full rank:      k=d_model recovers the original vector exactly
        4. Vh shape:       consistent with d_model

    Args:
        W_U:     unembedding matrix, shape [d_model, vocab_size]
        Vh:      precomputed right singular vectors from compute_wu_svd()
        d_model: model's residual stream dimension

    FORTRAN analogy: a validation subroutine you call once at the start
    of a simulation — like checking that your basis set is orthonormal
    and complete before beginning time-stepping.
    """
    assert Vh.shape[0] == d_model, (
        f"Vh has {Vh.shape[0]} rows but d_model={d_model}"
    )

    # Tolerance note: float32 arithmetic over d_model dimensions accumulates
    # rounding errors roughly as O(sqrt(d_model) * machine_eps). For d_model=768
    # and two matmuls (Q.T @ r then Q @ result), errors of O(1e-4) are normal.
    # This is the same effect you'd see reconstructing a field from its full
    # EOF expansion in single-precision FORTRAN — the round-trip through two
    # DGEMM calls loses a few ulps per dimension.
    tol_tight = 1e-4    # for algebraic identities (recon, orthogonality)
    tol_loose = 1e-3    # for full-rank round-trip through two d_model×d_model matmuls

    # 1. Reconstruction check: r‖ + r⊥ = r  (algebraic identity, no matmul error)
    r = torch.randn(d_model)
    Q_k = Vh[:100, :].T          # [d_model, 100]
    r_par  = Q_k @ (Q_k.T @ r)
    r_perp = r - r_par
    recon_error = (r - (r_par + r_perp)).norm().item()
    assert recon_error < tol_tight, f"Reconstruction failed: {recon_error}"

    # 2. Orthogonality check: r‖ · r⊥ ≈ 0
    dot = (r_par * r_perp).sum().item()
    assert abs(dot) < tol_tight, f"Orthogonality failed: {dot}"

    # 3. Limit check: k=d_model gives zero ablation effect
    #    This is the most error-prone check: full-rank Q is [d_model, d_model],
    #    so the round-trip Q @ (Q.T @ r) accumulates more floating-point error
    #    than the partial-rank projections used in actual ablation.
    Q_full = Vh[:d_model, :].T   # [d_model, d_model]
    r_full = Q_full @ (Q_full.T @ r)
    full_error = (r - r_full).norm().item()
    assert full_error < tol_loose, f"Full rank reconstruction failed: {full_error}"

    print(f"All validation checks passed. "
          f"(recon={recon_error:.1e}, ortho={abs(dot):.1e}, "
          f"full_rank={full_error:.1e})")


# ============================================================================
# STAGE 1: POST-HOC PROJECTION ABLATION
# Operates on stored ActivationRecord activations — no forward pass needed.
# At each layer, projects residual stream onto top-k W_U directions,
# applies ln_final and W_U, and compares to the full (k=d_model) baseline.
# ============================================================================

def compute_posthoc_ablation(
    record:   ActivationRecord,
    W_U:      torch.Tensor,       # [d_model, vocab_size], from model.W_U
    ln_final: Callable,           # model.ln_final, same as logit lens
    Vh:       torch.Tensor,       # precomputed SVD, shape [d_model, d_model]
    k_values: list[int],          # subspace ranks to sweep
    alpha:    float = 1.0,        # Renyi alpha for entropy_change (Shannon default)
) -> list[AblationRecord]:
    """
    Post-hoc projection ablation: remove r⊥ from stored activations
    and measure how token predictions change.

    For each layer L and subspace rank k:
        1. Project resid_post[L, -1, :] onto top-k W_U singular directions
        2. Apply ln_final and W_U to get ablated token probabilities
        3. Compare to full (unablated) probabilities via KL divergence,
           entropy change, and top-1 token preservation

    The full baseline (k=d_model, no ablation) is computed once and reused
    for all k values. This is the inner loop optimization — the baseline
    is the expensive part (ln_final + matmul) and it's the same for every k.

    Args:
        record:   ActivationRecord from extraction.py
        W_U:      unembedding matrix, shape [d_model, vocab_size]
        ln_final: final layer norm callable (model.ln_final)
        Vh:       right singular vectors from compute_wu_svd(),
                  shape [d_model, d_model]
        k_values: list of subspace ranks to sweep, e.g. [10, 50, 100, 200, 400, 600]
        alpha:    Renyi order for entropy_change computation (default 1.0 = Shannon)

    Returns:
        list of AblationRecord, one per k value. Length = len(k_values).

    FORTRAN analogy:
        Think of this as a two-level loop: the outer loop over k values
        is like varying a truncation wavenumber in a spectral model,
        and the inner loop over layers is like iterating over vertical
        levels to compute a diagnostic profile. The baseline is computed
        once (like a reference run), and each k-truncation is compared
        against it.

        The arrays are dimensioned as:
            REAL :: kl_div(n_layers)
            REAL :: ent_change(n_layers)
            LOGICAL :: top1_pres(n_layers)

        One set of arrays per k value, packaged into an AblationRecord.
    """
    n_layers = record.n_layers
    d_model  = record.d_model

    W_U_cpu = W_U.float().cpu()
    Vh_cpu  = Vh.float().cpu()

    # ------------------------------------------------------------------
    # Step 1: Compute full baseline at every layer (k = d_model, no ablation)
    # ------------------------------------------------------------------
    # These arrays are indexed by layer, like REAL :: probs_full(vocab, n_layers)
    probs_full = [None] * n_layers
    top1_full  = np.zeros(n_layers, dtype=np.int64)
    H_full     = np.zeros(n_layers, dtype=np.float64)

    with torch.no_grad():
        for layer in range(n_layers):
            r_full = torch.from_numpy(
                record.activations[layer, -1, :]
            ).float()                                       # [d_model]

            normed_full = ln_final(r_full).cpu()            # [d_model]
            logits_full = normed_full @ W_U_cpu             # [vocab_size]
            pf = torch.softmax(logits_full, dim=-1).clamp(min=1e-12)

            probs_full[layer] = pf
            top1_full[layer]  = pf.argmax().item()
            H_full[layer]     = renyi_entropy(pf, alpha)

    # ------------------------------------------------------------------
    # Step 2: For each k, project and compare to baseline
    # ------------------------------------------------------------------
    results = []

    for k in k_values:
        Q_k = Vh_cpu[:k, :].T.contiguous()   # [d_model, k]

        kl_div     = np.zeros(n_layers, dtype=np.float64)
        ent_change = np.zeros(n_layers, dtype=np.float64)
        top1_pres  = np.zeros(n_layers, dtype=bool)

        with torch.no_grad():
            for layer in range(n_layers):
                r = torch.from_numpy(
                    record.activations[layer, -1, :]
                ).float()                                   # [d_model]

                # Project into top-k W_U subspace: keep r‖, zero r⊥
                r_ablated = Q_k @ (Q_k.T @ r)              # [d_model]

                normed_abl = ln_final(r_ablated).cpu()      # [d_model]
                logits_abl = normed_abl @ W_U_cpu           # [vocab_size]
                probs_abl  = torch.softmax(logits_abl, dim=-1).clamp(min=1e-12)
                top1_abl   = probs_abl.argmax().item()
                H_abl      = renyi_entropy(probs_abl, alpha)

                # KL(probs_full || probs_ablated)
                pf = probs_full[layer]
                kl_div[layer] = (
                    pf * (pf.log() - probs_abl.log())
                ).sum().item()

                ent_change[layer] = H_abl - H_full[layer]
                top1_pres[layer]  = (top1_abl == top1_full[layer])

        results.append(AblationRecord(
            kl_divergence      = kl_div,
            entropy_change     = ent_change,
            top1_preserved     = top1_pres,
            k                  = k,
            ablation_type      = "posthoc",
            intervention_layer = None,
            model_name         = record.model_name,
            prompt             = record.prompt,
            role               = record.role,
            category           = record.category,
            hook_type          = record.hook_type,
        ))

    return results


# ============================================================================
# STAGE 2: SINGLE-LAYER FORWARD PASS INTERVENTION
# Uses TransformerLens hooks to zero r⊥ at a specific layer during a live
# forward pass. Measures the downstream effect at the final layer.
# ============================================================================

def make_ablation_hook(Q_k: torch.Tensor):
    """
    Returns a TransformerLens hook function that projects resid_post onto
    top-k W_U directions, zeroing the orthogonal complement r⊥.

    The hook modifies the residual stream in-place at the final token
    position only, leaving all other positions unchanged.

    Args:
        Q_k: projection basis, shape [d_model, k]
             columns are the top-k right singular vectors of W_U

    Returns:
        hook_fn: callable with signature (value, hook) -> value
                 compatible with TransformerLens run_with_hooks()

    FORTRAN analogy:
        This is like a callback subroutine that the model's time-stepper
        calls at a specific vertical level. It receives the state variable,
        applies a spectral filter (zeroing high-wavenumber modes), and
        hands the modified state back to the time-stepper to continue
        propagating.
    """
    def hook_fn(value, hook):
        # value shape: [batch, seq_len, d_model]
        # operate on final token position only
        r = value[0, -1, :].float()                     # [d_model]
        r_ablated = Q_k @ (Q_k.T @ r)                   # [d_model]
        value[0, -1, :] = r_ablated.to(value.dtype)
        return value
    return hook_fn


def compute_intervention_ablation(
    record:              ActivationRecord,
    model,                                  # HookedTransformer (live model)
    W_U:                 torch.Tensor,      # [d_model, vocab_size]
    ln_final:            Callable,
    Vh:                  torch.Tensor,      # precomputed SVD
    k_values:            list[int],
    intervention_layers: list[int],         # which layers to intervene at
    alpha:               float = 1.0,
) -> list[AblationRecord]:
    """
    Single-layer forward pass intervention: zero r⊥ at a specific layer
    during the model's forward pass and measure the effect on final-layer
    token predictions.

    For each intervention layer L and subspace rank k:
        1. Register a hook at blocks.{L}.hook_resid_post that projects
           the residual stream onto top-k W_U directions
        2. Run a forward pass with this hook active
        3. Compare the final-layer token predictions to the clean baseline

    Unlike posthoc ablation, this captures the causal downstream effect:
    the model's remaining layers process the ablated representation,
    potentially amplifying or compensating for the missing r⊥ content.

    Args:
        record:              ActivationRecord from extraction.py
        model:               live HookedTransformer instance
        W_U:                 unembedding matrix, shape [d_model, vocab_size]
        ln_final:            final layer norm callable
        Vh:                  right singular vectors from compute_wu_svd()
        k_values:            list of subspace ranks to sweep
        intervention_layers: list of layer indices to intervene at
        alpha:               Renyi order for entropy computation

    Returns:
        list of AblationRecord, one per (k, intervention_layer) combination.
        Length = len(k_values) × len(intervention_layers).
        Each record has metric arrays of shape [1] (single final-layer value).

    FORTRAN analogy:
        This is a perturbed-simulation experiment. The clean forward pass
        is the control run. For each (layer, k) combination, you restart
        the simulation, inject a perturbation at one vertical level, let
        the model propagate forward, and compare the final state to the
        control. It's the transformer equivalent of an adjoint sensitivity
        test — except instead of computing gradients analytically, you
        measure the finite perturbation response directly.
    """
    # ------------------------------------------------------------------
    # Step 1: Clean baseline forward pass
    # ------------------------------------------------------------------
    tokens = model.to_tokens(record.prompt)

    with torch.no_grad():
        logits_full = model(tokens)[0, -1, :]              # [vocab_size]
        probs_full  = torch.softmax(logits_full, dim=-1).clamp(min=1e-12)
        top1_full   = probs_full.argmax().item()
        H_full      = renyi_entropy(probs_full.cpu(), alpha)

    # ------------------------------------------------------------------
    # Step 2: Intervene at each (layer, k) and measure effect
    # ------------------------------------------------------------------
    Vh_cpu  = Vh.float().cpu()
    results = []

    for L in intervention_layers:
        hook_name = f"blocks.{L}.hook_resid_post"

        for k in k_values:
            Q_k = Vh_cpu[:k, :].T.contiguous()              # [d_model, k]

            # Move Q_k to model device for the hook
            Q_k_device = Q_k.to(tokens.device)
            hook_fn    = make_ablation_hook(Q_k_device)

            with torch.no_grad():
                logits_abl = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_name, hook_fn)]
                )[0, -1, :]                                  # [vocab_size]

                probs_abl = torch.softmax(logits_abl, dim=-1).clamp(min=1e-12)
                top1_abl  = probs_abl.argmax().item()
                H_abl     = renyi_entropy(probs_abl.cpu(), alpha)

            # KL(probs_full || probs_ablated)
            kl_div = (
                probs_full * (probs_full.log() - probs_abl.log())
            ).sum().item()

            ent_chg   = H_abl - H_full
            top1_pres = (top1_abl == top1_full)

            results.append(AblationRecord(
                kl_divergence      = np.array([kl_div]),
                entropy_change     = np.array([ent_chg]),
                top1_preserved     = np.array([top1_pres]),
                k                  = k,
                ablation_type      = "intervention",
                intervention_layer = L,
                model_name         = record.model_name,
                prompt             = record.prompt,
                role               = record.role,
                category           = record.category,
                hook_type          = record.hook_type,
            ))

    return results
