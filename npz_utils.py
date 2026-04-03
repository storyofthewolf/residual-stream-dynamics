"""
npz_utils.py
------------
Loading and filtering utilities for entropy and ablation .npz files.

All computation and plotting lives elsewhere. This module is pure data access.
"""

from pathlib import Path
import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")


# ── Loading ────────────────────────────────────────────────────────────────────

def load_entropy_npz(path):
    """
    Load an entropy .npz file produced by entropy_analysis.py --save-data.

    Parameters
    ----------
    path : str or Path
        Path to the .npz file. If not found as given, DATA_DIR is prepended.

    Returns
    -------
    data : NpzFile
        Raw numpy archive. Access arrays by key, e.g. data["surfaces"].
    """
    path = Path(path)
    if not path.exists():
        path = DATA_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Entropy .npz not found: {path}")
    return np.load(path, allow_pickle=True)


def load_ablation_npz(path):
    """
    Load an ablation .npz file produced by ablation_analysis.py --save-data.

    Parameters
    ----------
    path : str or Path
        Path to the .npz file. If not found as given, DATA_DIR is prepended.

    Returns
    -------
    data : NpzFile
        Raw numpy archive.
    """
    path = Path(path)
    if not path.exists():
        path = DATA_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Ablation .npz not found: {path}")
    return np.load(path, allow_pickle=True)


# ── Filtering ──────────────────────────────────────────────────────────────────

def get_final_token_profiles(
    data,
    norm_key,
    role,
    alpha=1.0,
    hook_type="resid_post",
):
    """
    Extract final-token entropy profiles for a given filtering condition.

    Each prompt has a variable sequence length and potentially a different
    number of layers (across model families). Profiles are returned as a list
    of 1D arrays rather than a padded 2D array to avoid NaNs.

    Parameters
    ----------
    data : NpzFile
        Loaded entropy .npz archive from load_entropy_npz().
    norm_key : str
        Normalization key to select. One of: 'energy', 'abs', 'softmax',
        'logit_lens'.
    role : str
        Prompt role to select. One of: 'base', 'contrast'.
    alpha : float
        Renyi alpha value. Use 1.0 for Shannon entropy.
    hook_type : str
        Hook type to select. Default 'resid_post'.

    Returns
    -------
    profiles : list of np.ndarray
        Each element is shape [n_layers] — the entropy at each layer
        for the final token of one prompt.
    pair_ids : np.ndarray, shape [n_prompts]
        Pair identifiers for matched base/contrast comparisons.
    categories : np.ndarray, shape [n_prompts]
        Prompt category labels.
    n_layers : np.ndarray, shape [n_prompts]
        Number of layers for each prompt (varies across model families).
    """
    norm_keys  = data["norm_keys"]
    hook_types = data["hook_types"]
    alphas     = data["alphas"]
    roles      = data["roles"]
    surfaces   = data["surfaces"]
    seq_lens   = data["seq_lens"]
    n_layers   = data["n_layers"]
    pair_ids   = data["pair_ids"]
    categories = data["categories"]

    mask = (
        (norm_keys  == norm_key)  &
        (hook_types == hook_type) &
        (alphas     == alpha)     &
        (roles      == role)
    )

    indices = np.where(mask)[0]

    profiles = []
    for i in indices:
        final_tok = int(seq_lens[i]) - 1
        nl        = int(n_layers[i])
        # surfaces shape: [n_layers, seq_len] — extract final token column
        profile   = surfaces[i, :nl, final_tok]
        profiles.append(profile)

    return (
        profiles,
        pair_ids[indices],
        categories[indices],
        n_layers[indices],
    )


def get_ablation_records(
    data,
    role,
    ablation_type,
    k,
    hook_type="resid_post",
    intervention_lyr=-1,
):
    """
    Extract ablation metric profiles for a given filtering condition.

    Parameters
    ----------
    data : NpzFile
        Loaded ablation .npz archive from load_ablation_npz().
    role : str
        Prompt role. One of: 'base', 'contrast'.
    ablation_type : str
        One of: 'posthoc', 'intervention'.
    k : int
        Subspace rank to select.
    hook_type : str
        Hook type to select. Default 'resid_post'.
    intervention_lyr : int
        For posthoc ablation use -1 (default).
        For intervention ablation, specify the intervention layer index.

    Returns
    -------
    kl_divergence : np.ndarray, shape [n_prompts, n_layers]
        KL divergence from full-model distribution after ablation.
    entropy_change : np.ndarray, shape [n_prompts, n_layers]
        H(ablated) - H(full). Positive = more diffuse after ablation.
    top1_preserved : np.ndarray of bool, shape [n_prompts, n_layers]
        Whether top-1 token is unchanged after ablation.
    categories : np.ndarray, shape [n_prompts]
        Prompt category labels.
    arr_lens : np.ndarray, shape [n_prompts]
        Number of valid layers per record (analogous to n_layers in entropy).
    """
    roles             = data["roles"]
    ablation_types    = data["ablation_types"]
    ks                = data["ks"]
    hook_types        = data["hook_types"]
    intervention_lyrs = data["intervention_lyrs"]
    categories        = data["categories"]
    arr_lens          = data["arr_lens"]

    mask = (
        (roles             == role)           &
        (ablation_types    == ablation_type)  &
        (ks                == k)              &
        (hook_types        == hook_type)      &
        (intervention_lyrs == intervention_lyr)
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        raise ValueError(
            f"No records found for role='{role}', ablation_type='{ablation_type}', "
            f"k={k}, hook_type='{hook_type}', intervention_lyr={intervention_lyr}.\n"
            f"Available ks: {np.unique(ks)}, "
            f"ablation_types: {np.unique(ablation_types)}, "
            f"intervention_lyrs: {np.unique(intervention_lyrs)}"
        )

    # slice valid layers per record using arr_lens
    kl_div  = data["kl_divergence"]
    ent_chg = data["entropy_change"]
    top1    = data["top1_preserved"]
    lens    = arr_lens[indices]

    kl_out   = []
    ent_out  = []
    top1_out = []
    for i, idx in enumerate(indices):
        nl = int(lens[i])
        kl_out.append(kl_div[idx,  :nl])
        ent_out.append(ent_chg[idx, :nl])
        top1_out.append(top1[idx,   :nl])

    return (
        kl_out,
        ent_out,
        top1_out,
        categories[indices],
        lens,
    )


def build_intervention_heatmap(
    data,
    metric,
    role,
    k_values,
    intervention_lyrs,
    hook_type="resid_post",
):
    """
    Build a 2D matrix of a scalar summary statistic across (k, intervention_layer)
    for intervention ablation records. Designed for plot_intervention_heatmap().

    The scalar summary per cell is:
        - entropy_change  : mean ΔEntropy across prompts and layers
        - kl_divergence   : mean KL divergence across prompts and layers
        - top1_preserved  : mean fraction of prompts preserving top-1 token,
                            averaged across layers

    Parameters
    ----------
    data : NpzFile
        Loaded ablation .npz from load_ablation_npz().
    metric : str
        One of: 'entropy_change', 'kl_divergence', 'top1_preserved'.
    role : str
        One of: 'base', 'contrast'.
    k_values : list of int
        Subspace ranks to include (rows of output matrix).
    intervention_lyrs : list of int
        Intervention layer indices to include (columns of output matrix).
    hook_type : str
        Default 'resid_post'.

    Returns
    -------
    matrix : np.ndarray, shape [n_k, n_intervention_lyrs]
        Summary statistic for each (k, intervention_layer) cell.
        NaN where no records exist.
    """
    matrix = np.full((len(k_values), len(intervention_lyrs)), np.nan)

    for i, k in enumerate(k_values):
        for j, lyr in enumerate(intervention_lyrs):
            try:
                kl_out, ent_out, top1_out, _, _ = get_ablation_records(
                    data,
                    role=role,
                    ablation_type="intervention",
                    k=k,
                    hook_type=hook_type,
                    intervention_lyr=lyr,
                )
            except ValueError:
                continue  # no records for this (k, lyr) — leave as NaN

            if metric == "entropy_change":
                vals = [np.mean(p) for p in ent_out]
            elif metric == "kl_divergence":
                vals = [np.mean(p) for p in kl_out]
            elif metric == "top1_preserved":
                vals = [np.mean(p.astype(float)) for p in top1_out]
            else:
                raise ValueError(f"Unknown metric: {metric}. "
                                 f"Choose from: entropy_change, kl_divergence, top1_preserved")

            matrix[i, j] = np.mean(vals)

    return matrix

