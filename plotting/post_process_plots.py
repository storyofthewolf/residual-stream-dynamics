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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

def _mean_and_ci(profiles, ci_level=0.95):
    """
    Compute layer-wise mean and t-distribution confidence interval across a
    list of variable-length profiles. Profiles are aligned by fractional layer
    depth (0 -> 1) via interpolation onto a common 100-point grid.

    Uses t-distribution CI on the mean (not a percentile interval), consistent
    with the paired t-test approach used in entropy_plots.py.

    Returns
    -------
    grid : np.ndarray, shape [100]
    mean : np.ndarray, shape [100]
    lo   : np.ndarray, shape [100]   lower CI bound
    hi   : np.ndarray, shape [100]   upper CI bound
    """
    from scipy import stats as sp_stats

    grid = np.linspace(0, 1, 100)
    interpolated = []
    for p in profiles:
        n = len(p)
        x = np.linspace(0, 1, n)
        interpolated.append(np.interp(grid, x, p))

    arr    = np.array(interpolated)
    n      = arr.shape[0]
    mean   = np.nanmean(arr, axis=0)
    sem    = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(1 - (1 - ci_level) / 2, df=n - 1)
    lo     = mean - t_crit * sem
    hi     = mean + t_crit * sem
    return grid, mean, lo, hi


def _diff_and_ci(base_profiles, contrast_profiles, ci_level=0.95):
    """
    Compute layer-wise paired difference (base - contrast) with t-distribution
    CI on the mean difference. Matches the methodology in entropy_plots.py.

    Returns
    -------
    grid : np.ndarray, shape [100]
    mean : np.ndarray, shape [100]
    lo   : np.ndarray, shape [100]
    hi   : np.ndarray, shape [100]
    """
    from scipy import stats as sp_stats

    grid = np.linspace(0, 1, 100)

    def _interp(profiles):
        return np.array([
            np.interp(grid, np.linspace(0, 1, len(p)), p)
            for p in profiles
        ])

    b    = _interp(base_profiles)
    c    = _interp(contrast_profiles)
    diff = b - c

    n      = diff.shape[0]
    mean   = np.nanmean(diff, axis=0)
    sem    = np.nanstd(diff, axis=0, ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(1 - (1 - ci_level) / 2, df=n - 1)
    lo     = mean - t_crit * sem
    hi     = mean + t_crit * sem
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

    grid, mean, lo, hi = _diff_and_ci(base_profiles, contrast_profiles)

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


def plot_ablation_heatmap_tripanel(
    base_mat,
    contrast_mat,
    k_values,
    layer_indices,
    model_name: str = "",
    title: str = "",
    x_axis_label: str = "layers",
    y_mode: str = "relative",
    d_model: int = None,
    ev_dict: dict = None,
    y_tick_labels: list = None,
    cbar_label: str = "",
    cbar_diff_label: str = "",
    max_layer_ticks: int = 16,
    save_path=None,
):
    """
    Three-panel heatmap for forward-pass intervention ablation, top-1 preservation only.

    Panels: base prompts | contrast prompts | base − contrast difference.
    The split-subfigure layout gives the two absolute panels a shared colorbar
    (Blues, 0–1) and the difference panel its own symmetric colorbar (RdBu_r).

    This is the multi-panel companion to plot_intervention_heatmap(), which shows
    only the difference. Use this function when the absolute preservation rates
    for each role are needed alongside the difference — e.g. to verify that a
    blue patch in the difference panel reflects base robustness rather than mutual
    collapse.

    Parameters
    ----------
    base_mat : np.ndarray, shape [n_k, n_layers]
        Top-1 preservation rate for base prompts.
        Rows = k values (subspace ranks), cols = intervention layers.
        Build upstream with build_intervention_heatmap(..., metric='top1_preserved',
        role='base').
    contrast_mat : np.ndarray, same shape
        Top-1 preservation rate for contrast prompts.
    k_values : array-like of int
        Subspace ranks corresponding to rows (bottom to top, origin='lower').
    layer_indices : array-like of int
        Intervention layer indices corresponding to columns.
    model_name : str
        Used in the figure suptitle.
    title : str
        Used in the figure suptitle.
    x_axis_label : str
        Set axis label from arugment passing.
        Else defaults to "layers"
    y_mode : str
        Controls y-axis labelling of subspace rank. One of:
            'k'        — raw integer rank
            'relative' — k / d_model fraction; requires d_model.
            'ev'       — explained variance fraction; requires ev_dict mapping k -> float.
        'relative' is recommended for cross-model comparison since it normalises
        by architecture width.
        Ignored when y_tick_labels is provided.
    d_model : int, optional
        Model hidden dimension. Required when y_mode='relative'.
    ev_dict : dict, optional
        Mapping {k: explained_variance_fraction}. Required when y_mode='ev'.
        Produced by wu_explained_variance() in ablation_compute.py.
    y_tick_labels : list of str, optional
        Explicit y-axis labels, one per entry in k_values (bottom to top).
        Overrides y_mode entirely. Use this when EV values are known from
        the workflow run but not stored in the npz — e.g.:
            y_tick_labels=['10%', '25%', '50%', '75%', '90%', '95%',
                           '99%', '99.9%', '100%']    
        Must have the same length as k_values.
    cbar_label : str
        Colorbar label for two raw value panel
    cbar_diff_label : str
        Colorbar label for difference panel
    max_layer_ticks : int
        Maximum number of x-axis (intervention layer) tick labels to show.
        Prevents overlap on dense models. A stride is computed automatically
        so that the number of visible labels does not exceed this value.
        Default 16 is comfortable up to ~32-layer models at typical figure width.
    save_path : str or Path, optional
        If provided, saves at 150 DPI.

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple of three Axes — (ax_base, ax_contrast, ax_diff)
    """
    import matplotlib.gridspec as gridspec

    # ── Build y-axis (subspace rank) labels ───────────────────────────────────
    if y_tick_labels is not None:
        # Manual override — use as-is, skip y_mode logic entirely.
        if len(y_tick_labels) != len(k_values):
            raise ValueError(
                f"y_tick_labels has {len(y_tick_labels)} entries but "
                f"k_values has {len(k_values)}. Lengths must match."
            )
        y_labels     = list(y_tick_labels)
        y_axis_label = "Explained variance retained"
    else:
        # ── Validate y_mode inputs ────────────────────────────────────────────
        if y_mode == "relative" and d_model is None:
            raise ValueError("y_mode='relative' requires d_model.")
        if y_mode == "ev" and ev_dict is None:
            raise ValueError("y_mode='ev' requires ev_dict.")

        if y_mode == "relative":
            y_labels     = [f"{k / d_model:.2f}" for k in k_values]
            y_axis_label = "Relative subspace rank  k / d_model"
        elif y_mode == "ev":
            y_labels     = [f"{ev_dict[k]*100:.1f}%" if k in ev_dict else str(k)
                            for k in k_values]
            y_axis_label = "Explained variance retained"
        else:  # 'k'
            y_labels     = [str(k) for k in k_values]
            y_axis_label = "Subspace rank k"

    # ── Build x-axis (intervention layer) tick decimation ────────────────────
    # stride = smallest integer such that n_layers / stride <= max_layer_ticks.
    # This is the same logic used in atmospheric model output: when you have 128
    # pressure levels but only room for 16 axis labels, you show every 8th level.
    n_lyr  = len(layer_indices)
    stride = max(1, int(np.ceil(n_lyr / max_layer_ticks)))
    x_tick_positions = list(range(0, n_lyr, stride))
    x_tick_labels    = [str(layer_indices[i]) for i in x_tick_positions]

    # ── Compute difference and colormap limits ────────────────────────────────
    diff_mat  = base_mat - contrast_mat
    vmax_diff = np.nanmax(np.abs(diff_mat))
    n_k       = len(k_values)

    # ── Figure and subfigure layout ───────────────────────────────────────────
    # Two subfigures: left holds base+contrast with a shared colorbar (Blues);
    # right holds the difference with its own symmetric colorbar (RdBu_r).
    # width_ratios=[2.2, 1.1] gives the difference panel slightly less width
    # than the two absolute panels combined — matches ablation_plots.py.
    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
      #  f"Intervention ablation: top-1 preservation heatmap  [{model_name}]",
        f"{title} [{model_name}]",
        fontsize=12,
    )

    sf_left, sf_right = fig.subfigures(1, 2, width_ratios=[2.2, 1.1], wspace=0.12)

    gs_left  = gridspec.GridSpec(1, 3, width_ratios=[10, 10, 0.6],
                                 wspace=0.12, figure=sf_left)
    gs_right = gridspec.GridSpec(1, 2, width_ratios=[10, 0.6],
                                 wspace=0.12, figure=sf_right)

    ax_base     = sf_left.add_subplot(gs_left[0, 0])
    ax_contrast = sf_left.add_subplot(gs_left[0, 1])
    cbar_ax1    = sf_left.add_subplot(gs_left[0, 2])
    ax_diff     = sf_right.add_subplot(gs_right[0, 0])
    cbar_ax2    = sf_right.add_subplot(gs_right[0, 1])

    im_kw = dict(aspect="auto", origin="lower")

    im1 = ax_base.imshow(    base_mat,     cmap="Blues",  vmin=0,          vmax=1,         **im_kw)
    im2 = ax_contrast.imshow(contrast_mat, cmap="Blues",  vmin=0,          vmax=1,         **im_kw)
#    im3 = ax_diff.imshow(    diff_mat,     cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff, **im_kw)
    im3 = ax_diff.imshow(    diff_mat,     cmap="RdBu_r", vmin=-0.6, vmax=0.6, **im_kw)

    ax_base.set_title("Base prompts",      fontsize=10)
    ax_contrast.set_title("Contrast prompts", fontsize=10)
    ax_diff.set_title("Base − Contrast",   fontsize=10)

    fig.colorbar(im2, cax=cbar_ax1, label=cbar_label)
    fig.colorbar(im3, cax=cbar_ax2, label=cbar_diff_label)

    # ── Apply axis ticks and labels to all three panels ───────────────────────
    for ax in [ax_base, ax_contrast, ax_diff]:
        ax.set_xlabel(x_axis_label, fontsize=9)
        ax.set_xticks(x_tick_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=7)
        ax.set_yticks(range(n_k))
        #ax.set_yticklabels(y_labels, fontsize=7)

    # y-axis labels: shown on ax_base and ax_diff only.
    # ax_contrast shares the same y-scale as ax_base; showing labels on both
    # causes the contrast panel's tick labels to overlap the base panel's plot
    # area. Suppressing them here keeps the shared colorbar as the sole legend.
    ax_base.set_yticklabels(y_labels, fontsize=7)
    ax_contrast.set_yticklabels([])
    ax_diff.set_yticklabels(y_labels, fontsize=7)


    for ax in [ax_base, ax_diff]:
        ax.set_ylabel(y_axis_label, fontsize=9)

    _save(fig, save_path)
    return fig, (ax_base, ax_contrast, ax_diff)


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


# ── Multi-model grid ───────────────────────────────────────────────────────────

def plot_paired_difference_grid(
    model_data,
    save_path=None,
):
    """
    Grid of paired difference curves (base - contrast) for all models.
    Rows: GPT-2 family (top), Pythia family (bottom).
    Columns: one per model within each family.
    Both residual stream and logit lens shown per panel.

    Parameters
    ----------
    model_data : dict
        Keyed by model name. Each value is a dict with keys:
            'base_resid', 'contrast_resid', 'base_logit', 'contrast_logit'
        Each value is a list of np.ndarray profiles.
    save_path : str or Path, optional

    Returns
    -------
    fig, axes
    """
    gpt2_models   = [m for m in model_data if "gpt2"   in m]
    pythia_models = [m for m in model_data if "pythia" in m]

    n_cols = max(len(gpt2_models), len(pythia_models))
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), sharey="row")
    fig.suptitle("Paired difference: base - contrast  [all models]", fontsize=13)

    grid = np.linspace(0, 1, 100)

    def _plot_diff(ax, base, contrast, color, label):
        _, mean, lo, hi = _diff_and_ci(base, contrast)
        ax.plot(grid, mean, color=color, linewidth=1.8, label=label)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.2)

    for row, models in enumerate([gpt2_models, pythia_models]):
        for col, model in enumerate(models):
            ax = axes[row, col]
            md = model_data[model]
            _plot_diff(ax, md["base_resid"],  md["contrast_resid"],  BASE_COLOR,     "resid")
            _plot_diff(ax, md["base_logit"],  md["contrast_logit"],  CONTRAST_COLOR, "logit lens")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(model, fontsize=9)
            ax.set_xlabel("Fractional layer depth", fontsize=8)
            if col == 0:
                ax.set_ylabel("Delta Entropy (bits)", fontsize=8)
            ax.legend(fontsize=7)
        for col in range(len(models), n_cols):
            axes[row, col].set_visible(False)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, axes


def plot_top1_vs_k(
    abl_data,
    k_values,
    model_name="",
    hook_type="resid_post",
    ax=None,
    save_path=None,
):
    """
    Top-1 token preservation rate vs subspace rank k for post-hoc ablation.

    Parameters
    ----------
    abl_data : NpzFile
        Loaded ablation .npz from load_ablation_npz().
    k_values : list of int
        Subspace ranks to plot. Must exist in abl_data.
    model_name : str
    hook_type : str
    ax : matplotlib Axes, optional
    save_path : str or Path, optional

    Returns
    -------
    fig, ax
    """
    from npz_utils import get_ablation_records

    fig, ax = _get_or_create_ax(ax, figsize=(7, 4))

    base_rates     = []
    contrast_rates = []

    for k in k_values:
        _, _, base_top1,     _, _ = get_ablation_records(
            abl_data, role="base",     ablation_type="posthoc", k=k, hook_type=hook_type)
        _, _, contrast_top1, _, _ = get_ablation_records(
            abl_data, role="contrast", ablation_type="posthoc", k=k, hook_type=hook_type)

        base_rates.append(np.mean([np.mean(p.astype(float)) for p in base_top1]))
        contrast_rates.append(np.mean([np.mean(p.astype(float)) for p in contrast_top1]))

    ax.plot(k_values, base_rates,     color=BASE_COLOR,     marker="o", linewidth=1.8, label="base")
    ax.plot(k_values, contrast_rates, color=CONTRAST_COLOR, marker="o", linewidth=1.8, label="contrast")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Subspace rank k")
    ax.set_ylabel("Mean top-1 preservation rate")
    ax.set_title(f"Post-hoc ablation: top-1 preservation vs k  [{model_name}]", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, save_path)
    return fig, ax

def compute_scaling_summary(model_data):
    """
    Compute peak paired difference statistics across all models.
    Returns a list of dicts ready for plot_scaling_summary().

    Parameters
    ----------
    model_data : dict
        Keyed by model name. Each value has keys:
            'base_resid', 'contrast_resid', 'base_logit', 'contrast_logit'

    Returns
    -------
    model_results : list of dict
        Each dict has keys: 'model_name', 'family', 'peak_resid', 'peak_logit'
    """
    grid = np.linspace(0, 1, 100)

    def _peak(base, contrast):
        _, mean, _, _ = _diff_and_ci(base, contrast)
        return float(np.max(np.abs(mean)))

    results = []
    for model, md in model_data.items():
        results.append({
            "model_name": model,
            "family":     "gpt2" if "gpt2" in model else "pythia",
            "peak_resid": _peak(md["base_resid"], md["contrast_resid"]),
            "peak_logit": _peak(md["base_logit"], md["contrast_logit"]),
        })
    return results


def load_all_models(models, data_dir, corpus_tag):
    """
    Load entropy profiles for all models in a single call.
    Returns a dict ready for plot_paired_difference_grid() and compute_scaling_summary().

    Parameters
    ----------
    models : list of str
        Model names, e.g. ['gpt2-small', 'pythia-160m', ...]
    data_dir : str or Path
        Directory containing .npz files.
    corpus_tag : str
        Corpus stem used in filenames, e.g. 'base_vs_contrast_n50'.

    Returns
    -------
    model_data : dict
        Keyed by model name. Each value has keys:
            'base_resid', 'contrast_resid', 'base_logit', 'contrast_logit'
    """
    from npz_utils import load_entropy_npz, get_final_token_profiles

    data_dir   = Path(data_dir)
    model_data = {}

    for model in models:
        path = data_dir / f"entropy_records_{model}_{corpus_tag}.npz"
        d    = load_entropy_npz(path)
        model_data[model] = {
            "base_resid":     get_final_token_profiles(d, norm_key="energy",     role="base")[0],
            "contrast_resid": get_final_token_profiles(d, norm_key="energy",     role="contrast")[0],
            "base_logit":     get_final_token_profiles(d, norm_key="logit_lens", role="base")[0],
            "contrast_logit": get_final_token_profiles(d, norm_key="logit_lens", role="contrast")[0],
        }

    return model_data
