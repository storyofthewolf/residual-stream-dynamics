"""probe_plots.py — Visualization for linear probe results.

Consumes ProbeRecords from src/probe_compute.py.
Produced by workflows/probe_analysis.py.

Pipeline position:
    probe_compute.py → VISUALIZATION (this file)

All figure construction for ProbeRecords lives here; none belongs in the
workflow. No file I/O beyond saving figures through _save().

Public functions:
    plot_probe_accuracy(record, ...)        — accuracy vs depth, with null band
    plot_probe_generalization(record, ...)  — per-held-out-group accuracy
    plot_probe_comparison(records, ...)     — several records on shared axes
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CHANCE = 0.5
_ROLE_COLORS = ["#1565C0", "#B71C1C", "#2E7D32", "#EF6C00", "#6A1B9A"]


def _save(fig, path: Path, filename: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    out = path / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def _depth_axis(n_layers: int) -> np.ndarray:
    """Fractional depth, so models with different layer counts are comparable."""
    if n_layers <= 1:
        return np.zeros(n_layers)
    return np.arange(n_layers) / (n_layers - 1)


def plot_probe_accuracy(record, output_dir: Path, corpus_tag: str = "",
                        run_tag: str = "", use_depth: bool = False):
    """Accuracy vs layer, with the permutation null and the layer-0 baseline.

    The layer-0 line marks how much separation is available from token identity
    alone; the shaded region above it is the part attributable to computation.
    """
    plt.close("all")
    acc = record.accuracy
    x = _depth_axis(record.n_layers) if use_depth else np.arange(record.n_layers)
    xlabel = "Fractional depth" if use_depth else "Layer"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, acc, "-o", color="#1565C0", lw=2, ms=4, label="probe accuracy")

    if not np.all(np.isnan(record.null_mean)):
        ax.plot(x, record.null_mean, "--", color="#757575", lw=1.2,
                label="permutation null")
    ax.axhline(CHANCE, color="#BDBDBD", ls=":", lw=1, zorder=0)

    # Layer-0 lexical baseline and the rise above it.
    ax.axhline(acc[0], color="#B71C1C", ls="--", lw=1.2, alpha=.8,
               label=f"layer-0 lexical baseline ({acc[0]:.2f})")
    ax.fill_between(x, acc[0], acc, where=acc >= acc[0],
                    color="#1565C0", alpha=.12, interpolate=True)

    # Mark layers that beat the null.
    if not np.all(np.isnan(record.p_value)):
        sig = record.p_value < 0.05
        if sig.any():
            ax.plot(x[sig], acc[sig], "o", color="#1565C0", ms=9,
                    mfc="none", mew=1.6, label="p < .05")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right")

    peak = int(np.argmax(acc))
    ax.set_title(
        f"Linear probe: base vs contrast  [{record.model_name}]\n"
        f"{corpus_tag}   n={record.n_samples} prompts / {record.n_pairs} pairs   "
        f"peak {acc[peak]:.3f} @ L{peak}   rise over L0 = {acc[peak] - acc[0]:+.3f}",
        fontsize=10)

    _save(fig, output_dir,
          f"probe_accuracy_{record.hook_type}_{record.model_name}_{corpus_tag}{run_tag}.png")
    return fig


def plot_probe_generalization(record, output_dir: Path, corpus_tag: str = "",
                              run_tag: str = "", use_depth: bool = False):
    """Leave-one-group-out accuracy: mean plus one line per held-out group.

    This is the figure that distinguishes a represented property from a
    memorized vocabulary — each line is a group the classifier never trained on.
    """
    plt.close("all")
    if record.group_accuracy is None or not record.group_names:
        raise ValueError("Record has no generalization results; "
                         "use compute_probe_generalization to produce one.")

    x = _depth_axis(record.n_layers) if use_depth else np.arange(record.n_layers)
    xlabel = "Fractional depth" if use_depth else "Layer"

    fig, ax = plt.subplots(figsize=(8, 5))
    for gi, name in enumerate(record.group_names):
        ax.plot(x, record.group_accuracy[:, gi], "-", lw=1.2, alpha=.65,
                color=_ROLE_COLORS[gi % len(_ROLE_COLORS)],
                label=f"held out: {name}")
    ax.plot(x, record.accuracy, "-o", color="black", lw=2.2, ms=4,
            label="mean", zorder=5)
    ax.axhline(CHANCE, color="#BDBDBD", ls=":", lw=1.2, zorder=0, label="chance")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Held-out accuracy")
    ax.set_ylim(0.30, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    peak = int(np.argmax(record.accuracy))
    ax.set_title(
        f"Probe generalization: leave-one-{record.generalize_by}-out  "
        f"[{record.model_name}]\n{corpus_tag}   "
        f"peak mean {record.accuracy[peak]:.3f} @ L{peak}",
        fontsize=10)

    _save(fig, output_dir,
          f"probe_generalize_{record.generalize_by}_{record.hook_type}_"
          f"{record.model_name}_{corpus_tag}{run_tag}.png")
    return fig


def plot_probe_comparison(records: list, output_dir: Path, labels: list = None,
                          corpus_tag: str = "", run_tag: str = "",
                          use_depth: bool = True, filename: str = None):
    """Several ProbeRecords on shared axes.

    Defaults to fractional depth on the x-axis, because the usual reason to
    compare is across models with different layer counts.
    """
    plt.close("all")
    if not records:
        raise ValueError("No records to plot.")
    if labels is None:
        labels = [f"{r.model_name} {r.corpus_tag}".strip() for r in records]

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (r, lab) in enumerate(zip(records, labels)):
        x = _depth_axis(r.n_layers) if use_depth else np.arange(r.n_layers)
        ax.plot(x, r.accuracy, "-o", lw=2, ms=3.5,
                color=_ROLE_COLORS[i % len(_ROLE_COLORS)], label=lab)
    ax.axhline(CHANCE, color="#BDBDBD", ls=":", lw=1.2, zorder=0, label="chance")

    ax.set_xlabel("Fractional depth" if use_depth else "Layer")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.35, 1.02)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Linear probe comparison", fontsize=10)

    _save(fig, output_dir, filename or f"probe_comparison_{corpus_tag}{run_tag}.png")
    return fig
