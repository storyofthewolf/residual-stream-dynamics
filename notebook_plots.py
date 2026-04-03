"""
notebook_plots.py
-----------------
Curated plotting functions for notebook narrative figures.

Design principles:
- All functions accept pre-filtered profiles (list of 1D np.ndarray, one per prompt)
  rather than raw .npz data. Filtering is done upstream via npz_utils.py.
- All functions accept ax=None: if None, a figure is created internally;
  if provided, the plot is drawn into the supplied axes for multi-panel composition.
- All functions return (fig, ax) for downstream annotation or saving.
- save_path is always optional.

Typical notebook cell:
    data = load_entropy_npz("data/entropy_records_gpt2-small_base_vs_contrast_n50.npz")
    base     = get_final_token_profiles(data, norm_key="energy",      role="base")
    contrast = get_final_token_profiles(data, norm_key="energy",      role="contrast")
    base_ll  = get_final_token_profiles(data, norm_key="logit_lens",  role="base")
    con_ll   = get_final_token_profiles(data, norm_key="logit_lens",  role="contrast")
    fig, axes = plot_entropy_profiles(base, contrast, base_ll, con_ll,
                                      model_name="gpt2-small")
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Color palette ──────────────────────────────────────────────────────────────
BASE_COLOR     = "#2166ac"   # blue
CONTRAST_COLOR = "#d6604d"   # red
DIFF_COLOR     = "#4d9e4d"   # green — for paired difference

# ── Model parameter counts (actual TransformerLens counts) ─────────────────────
_GPT2_PARAMS = {
    "gpt2-small":  124_000_000,
    "gpt2-medium": 355_000_000,
    "gpt2-large":  774_000_000,
    "gpt2-xl":   1_558_000_000,
}

def get_param_count(model_name: str) -> int:
    """
    Return parameter count for a model.
    GPT-2 sizes are looked up from a table.
    Pythia sizes are parsed from the model name (e.g. 'pythia-160m', 'pythia-6.9b').
    """
    if model_name in _GPT2_PARAMS:
        return _GPT2_PARAMS[model_name]

    if "pythia" in model_name:
        # parse suffix: pythia-160m, pythia-1b, pythia-2.8b, pythia-6.9b
        suffix = model_name.split("-")[-1].lower()
        if suffix.endswith("m"):
            return int(float(suffix[:-1]) * 1_000_000)
        if suffix.endswith("b"):
            return int(float(suffix[:-1]) * 1_000_000_000)

    raise ValueError(f"Unknown model name for parameter lookup: {model_name}")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _mean_and_ci(profiles):
    """
    Compute layer-wise mean and 95% CI across a list of variable-length profiles.
    Profiles are aligned by fractional layer depth (0 → 1) via interpolation
    onto a common 100-point grid — safe for cross-model comparison.

    Returns
    -------
    grid : np.ndarray, shape [100]
    mean : np.ndarray, shape [100]
    lo   : np.ndarray, shape [100]   (2.5th percentile)
    hi   : np.ndarray, shape [100]   (97.5th percentile)
    """
    grid = np.linspace(0, 1, 100)
    interpolated = []
    for p in profiles:
        n = len(p)
        x = np.linspace(0, 1, n)
        interpolated.append(np.interp(grid, x, p))
    arr  = np.array(interpolated)          # [n_prompts, 100]
    mean = np.nanmean(arr, axis=0)
    lo   = np.nanpercentile(arr, 2.5,  axis=0)
    hi   = np.nanpercentile(arr, 97.5, axis=0)
    return grid, mean, lo, hi


def _layer_axis(n_layers):
    """
    Return integer layer indices for a single-model plot (no interpolation needed).
    Use when all profiles share the same n_layers.
    """
    return np.arange(n_layers)


def _save(fig, save_path):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def _get_or_create_ax(ax, figsize):
    """Return (fig, ax): create a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


# ── Notebook 1: Core finding ───────────────────────────────────────────────────

def plot_entropy_profiles(
    base_resid,
    contrast_resid,
    base_logit,
    contrast_logit,
    model_name: str = "",
    save_path=None,
):
    """
    Two-panel figure: residual stream entropy (left) and logit lens entropy (right).
    Shows mean ± 95% CI for base and contrast prompts across fractional layer depth.

    Parameters
    ----------
    base_resid, contrast_resid : list of np.ndarray
        Final-token residual stream entropy profiles (e.g. energy norm, Shannon).
    base_logit, contrast_logit : list of np.ndarray
        Final-token logit lens entropy profiles (Shannon).
    model_name : str
        Used in figure title.
    save_path : str or Path, optional

    Returns
    -------
    fig, axes : matplotlib Figure and array of 2 Axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    fig.suptitle(f"Entropy profiles — final token  [{model_name}]", fontsize=12)

    panel_data = [
        (base_resid,  contrast_resid,  "Residual stream entropy (energy norm, Shannon)"),
        (base_logit,  contrast_logit,  "Logit lens entropy (Shannon)"),
    ]

    for ax, (base, contrast, title) in zip(axes, panel_data):
        for profiles, color, label in [
            (base,     BASE_COLOR,     "base"),
            (contrast, CONTRAST_COLOR, "contrast"),
        ]:
            grid, mean, lo, hi = _mean_and_ci(profiles)
            ax.plot(grid, mean, color=color, label=label, linewidth=1.8)
            ax.fill_between(grid, lo, hi, color=color, alpha=0.2)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Fractional layer depth")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_paired_difference(
    base_profiles,
    contrast_profiles,
    model_name: str = "",
    label: str = "base − contrast",
    ax=None,
    save_path=None,
):
    """
    Single-panel paired difference: mean(base) − mean(contrast) vs fractional
    layer depth, with 95% CI. Positive = base > contrast.

    Parameters
    ----------
    base_profiles, contrast_profiles : list of np.ndarray
        Matched profiles — must be in pair_id order (call get_final_token_profiles
        and sort by pair_id before passing).
    model_name : str
    label : str
        Legend label for the difference curve.
    ax : matplotlib Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = _get_or_create_ax(ax, figsize=(6, 4))

    # interpolate both onto common grid then difference
    grid = np.linspace(0, 1, 100)
    base_interp     = np.array([np.interp(grid, np.linspace(0,1,len(p)), p)
                                 for p in base_profiles])
    contrast_interp = np.array([np.interp(grid, np.linspace(0,1,len(p)), p)
                                 for p in contrast_profiles])
    diff = base_interp - contrast_interp      # [n_pairs, 100]

    mean = np.nanmean(diff, axis=0)
    lo   = np.nanpercentile(diff, 2.5,  axis=0)
    hi   = np.nanpercentile(diff, 97.5, axis=0)

    ax.plot(grid, mean, color=DIFF_COLOR, linewidth=1.8, label=label)
    ax.fill_between(grid, lo, hi, color=DIFF_COLOR, alpha=0.2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fractional layer depth")
    ax.set_ylabel("Δ Entropy (bits)  [base − contrast]")
    ax.set_title(f"Paired difference — final token  [{model_name}]", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_wu_explained_variance(
    k_values,
    explained_variance,
    model_name: str = "",
    ax=None,
    save_path=None,
):
    """
    Bar chart of cumulative explained variance of W_U by top-k singular directions.

    Parameters
    ----------
    k_values : array-like of int
    explained_variance : array-like of float
        Cumulative explained variance fraction (0–1) for each k.
    model_name : str
    ax : matplotlib Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = _get_or_create_ax(ax, figsize=(7, 4))

    k_vals = np.array(k_values)
    ev     = np.array(explained_variance)

    bars = ax.bar(range(len(k_vals)), ev, color=BASE_COLOR, alpha=0.8)
    ax.set_xticks(range(len(k_vals)))
    ax.set_xticklabels([str(k) for k in k_vals])
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Subspace rank k")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(
        f"W_U explained variance by top-k singular directions  [{model_name}]",
        fontsize=11
    )

    # annotate bars with percentage
    for bar, val in zip(bars, ev):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val*100:.1f}%",
            ha="center", va="bottom", fontsize=8
        )

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ── Notebook 2: Ablation / causal confirmation ─────────────────────────────────

def plot_ablation_entropy_change(
    base_profiles,
    contrast_profiles,
    model_name: str = "",
    k: int = 100,
    ax=None,
    save_path=None,
):
    """
    Single-panel entropy change vs layer for a fixed k value.
    ΔEntropy = H(ablated) − H(full). Positive = more diffuse after ablation.

    Parameters
    ----------
    base_profiles, contrast_profiles : list of np.ndarray
        Final-token entropy change profiles for the chosen k.
        Filter upstream by k value before passing.
    model_name : str
    k : int
        Displayed in title for context. Does not filter — filtering is upstream.
    ax : matplotlib Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = _get_or_create_ax(ax, figsize=(6, 4))

    for profiles, color, label in [
        (base_profiles,     BASE_COLOR,     "base"),
        (contrast_profiles, CONTRAST_COLOR, "contrast"),
    ]:
        grid, mean, lo, hi = _mean_and_ci(profiles)
        ax.plot(grid, mean, color=color, label=label, linewidth=1.8)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.2)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fractional layer depth")
    ax.set_ylabel("ΔEntropy: H(ablated) − H(full)  (bits)")
    ax.set_title(
        f"Post-hoc ablation: entropy change  [k={k}, {model_name}]",
        fontsize=11
    )
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


def plot_intervention_heatmap(
    diff_matrix,
    k_values,
    layer_indices,
    model_name: str = "",
    metric: str = "entropy_change",
    ax=None,
    save_path=None,
):
    """
    Heatmap of base − contrast difference for forward-pass intervention ablation.
    Shows only the difference panel (not base and contrast separately).

    Parameters
    ----------
    diff_matrix : np.ndarray, shape [n_k, n_layers]
        Base − contrast difference matrix. Rows = k values, cols = intervention layers.
    k_values : array-like of int
        Subspace ranks corresponding to rows of diff_matrix.
    layer_indices : array-like of int
        Intervention layer indices corresponding to columns.
    model_name : str
    metric : str
        One of 'entropy_change', 'top1_preserved', 'kl_divergence'.
        Used for colormap and label selection.
    ax : matplotlib Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    # metric-specific display settings
    metric_settings = {
        "entropy_change":  dict(cmap="RdBu_r",  label="Δ ΔEntropy (bits)",          center=0),
        "top1_preserved":  dict(cmap="RdBu_r",  label="Δ Top-1 preservation rate",  center=0),
        "kl_divergence":   dict(cmap="RdBu_r",  label="Δ KL divergence (nats)",     center=0),
    }
    settings = metric_settings.get(metric, metric_settings["entropy_change"])

    fig, ax = _get_or_create_ax(ax, figsize=(6, 5))

    vmax = np.nanmax(np.abs(diff_matrix))
    im = ax.imshow(
        diff_matrix,
        aspect="auto",
        origin="lower",
        cmap=settings["cmap"],
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([str(l) for l in layer_indices])
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([str(k) for k in k_values])
    ax.set_xlabel("Intervention layer")
    ax.set_ylabel("Subspace rank k")
    ax.set_title(
        f"Intervention ablation: base − contrast  [{metric}, {model_name}]",
        fontsize=11
    )

    fig.colorbar(im, ax=ax, label=settings["label"])
    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax


# ── Cross-model scaling summary ────────────────────────────────────────────────

def plot_scaling_summary(
    model_results,
    ax=None,
    save_path=None,
):
    """
    Peak paired difference vs model parameter count for residual stream
    and logit lens entropy. Summarizes the scaling story across model families.

    Parameters
    ----------
    model_results : list of dict, each with keys:
        'model_name'   : str
        'peak_resid'   : float  — max |mean paired difference| in residual stream
        'peak_logit'   : float  — max |mean paired difference| in logit lens
        'family'       : str    — 'gpt2' or 'pythia', used for marker style
    ax : matplotlib Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = _get_or_create_ax(ax, figsize=(7, 5))

    family_markers = {"gpt2": "o", "pythia": "s"}

    # collect points per family for connected lines
    for family, marker in family_markers.items():
        subset = [r for r in model_results if r["family"] == family]
        if not subset:
            continue
        subset = sorted(subset, key=lambda r: get_param_count(r["model_name"]))

        params      = [get_param_count(r["model_name"]) for r in subset]
        peak_resid  = [r["peak_resid"]  for r in subset]
        peak_logit  = [r["peak_logit"]  for r in subset]

        ax.plot(params, peak_resid,
                color=BASE_COLOR, marker=marker, linewidth=1.5,
                label=f"resid stream ({family})")
        ax.plot(params, peak_logit,
                color=CONTRAST_COLOR, marker=marker, linewidth=1.5,
                linestyle="--",
                label=f"logit lens ({family})")

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Peak |Δ entropy|  (bits)  [base − contrast]")
    ax.set_title("Scaling summary: entropy anti-correlation vs model size", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax
