"""Residual stream trajectory dynamics.

The residual stream evolves additively across layers:

    X_{L+1} = X_L + ΔX_L

where ΔX_L is the update written by attention + MLP at layer L.
This is formally equivalent to a discrete particle trajectory in d_model
dimensional space, enabling mechanical analogies:

    Position:     X_L          — residual stream vector at layer L
    Velocity:     ΔX_L         — update vector (displacement per layer)
    Speed:        ||ΔX_L||₂    — L2 norm of displacement
    Acceleration: ΔV_L         — change in velocity between layers
    Curvature:    cos(X_L, X_{L+1}) — cosine similarity of consecutive positions

Layer indexing note (analogous to finite difference in Fortran time-stepping):
    X         — n_layers entries:   layers 0 .. n_layers-1
    velocity  — n_layers-1 entries: transitions 0→1 .. (n-2)→(n-1)
    acceleration — n_layers-2 entries: one fewer again

Usage:
    python residual_stream_dynamics.py --corpus corpus.json
    python residual_stream_dynamics.py --corpus corpus.json --model pythia-1b
    python residual_stream_dynamics.py --corpus corpus.json --category pattern
    python residual_stream_dynamics.py --corpus corpus.json --no-plots
    python residual_stream_dynamics.py   # runs on default prompts (no corpus)
"""

import json
import argparse
import warnings
import logging
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="transformer_lens")
logging.getLogger("transformer_lens").setLevel(logging.ERROR)

# Fallback prompts for quick standalone testing (no corpus required)
DEFAULT_PROMPTS = [
    "wolf",
    "The wolf ran",
    "The wolf ran through the forest",
    "wolf wolf wolf",
    "asdfgh",
    "wol",
]

COLORS = plt.cm.tab10.colors


# ============================================================================
# MECHANICAL QUANTITIES
# All functions operate on layer vectors: tensors of shape (n_layers, d_model)
# ============================================================================

def compute_velocity(layer_vecs: torch.Tensor) -> torch.Tensor:
    """Displacement vector between consecutive layers.

    velocity[L] = X_{L+1} - X_L

    Args:
        layer_vecs: tensor of shape (n_layers, d_model)

    Returns:
        tensor of shape (n_layers-1, d_model)
    """
    return layer_vecs[1:] - layer_vecs[:-1]


def compute_speed(velocity: torch.Tensor) -> torch.Tensor:
    """L2 norm of each displacement vector — scalar speed at each transition.

    speed[L] = ||ΔX_L||₂

    Args:
        velocity: tensor of shape (n_layers-1, d_model)

    Returns:
        tensor of shape (n_layers-1,)
    """
    return torch.norm(velocity.float(), dim=1)


def compute_acceleration(velocity: torch.Tensor) -> torch.Tensor:
    """Change in velocity between consecutive layer transitions.

    acceleration[L] = ΔV_{L+1} - ΔV_L = velocity[L+1] - velocity[L]

    Args:
        velocity: tensor of shape (n_layers-1, d_model)

    Returns:
        tensor of shape (n_layers-2, d_model)
    """
    return velocity[1:] - velocity[:-1]


def compute_cosine_similarity(layer_vecs: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between consecutive position vectors.

    cos_sim[L] = cos(X_L, X_{L+1})

    Dips indicate layers making large directional changes — high curvature
    in the trajectory. Complements speed which measures magnitude of change.

    Args:
        layer_vecs: tensor of shape (n_layers, d_model)

    Returns:
        tensor of shape (n_layers-1,)
    """
    h0 = layer_vecs[:-1].float()   # (n_layers-1, d_model)
    h1 = layer_vecs[1:].float()    # (n_layers-1, d_model)
    return torch.nn.functional.cosine_similarity(h0, h1, dim=1)


# ============================================================================
# FORWARD PASS — extract layer vectors for a single prompt
# ============================================================================

def extract_layer_vectors(model, prompt: str, n_layers: int,
                           hook_pattern: str, device: str) -> torch.Tensor:
    """Run forward pass and return residual stream vectors at each layer.

    Args:
        model:        HookedTransformer
        prompt:       input string
        n_layers:     number of layers to extract
        hook_pattern: e.g. "blocks.{layer}.hook_resid_pre"
        device:       torch device string

    Returns:
        tensor of shape (n_layers, d_model) — final token position at each layer
    """
    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    vecs = []
    for layer in range(n_layers):
        hook_name = hook_pattern.format(layer=layer)
        vec = cache[hook_name][0, -1, :].float().cpu()
        vecs.append(vec)

    return torch.stack(vecs)   # (n_layers, d_model)


# ============================================================================
# ANALYSIS OVER CORPUS
# ============================================================================

def run_dynamics_analysis(model, corpus: list, n_layers: int,
                           hook_pattern: str, device: str,
                           category_filter: Optional[str] = None) -> list:
    """
    For each prompt in corpus, compute per-layer mechanical quantities
    of the residual stream trajectory at the final token position.

    Result dict contains scalar curves (lists of floats) suitable for
    JSON serialization and corpus-level aggregation.

    Velocity and acceleration direction vectors are not stored — only their
    magnitudes. Full layer vectors are kept in memory only during computation.

    Returns list of result dicts, one per prompt:
        {
            pair_id, role, category, description, prompt,
            speed:                  [float × n_layers-1],
            acceleration_magnitude: [float × n_layers-2],
            cosine_similarity:      [float × n_layers-1],
        }
    """
    filtered = corpus
    if category_filter:
        filtered = [e for e in corpus if e["category"] == category_filter]
        print(f"  Filtered to category '{category_filter}': {len(filtered)} prompts")

    results = []

    for i, entry in enumerate(filtered):
        prompt = entry["prompt"]

        layer_vecs = extract_layer_vectors(
            model, prompt, n_layers, hook_pattern, device
        )

        velocity = compute_velocity(layer_vecs)          # (n_layers-1, d_model)
        speed = compute_speed(velocity)                   # (n_layers-1,)
        acceleration = compute_acceleration(velocity)     # (n_layers-2, d_model)
        accel_mag = compute_speed(acceleration)           # (n_layers-2,) reuse norm fn
        cos_sim = compute_cosine_similarity(layer_vecs)  # (n_layers-1,)

        results.append({
            "pair_id":                entry["pair_id"],
            "role":                   entry["role"],
            "category":               entry["category"],
            "description":            entry["description"],
            "prompt":                 prompt,
            "speed":                  speed.tolist(),
            "acceleration_magnitude": accel_mag.tolist(),
            "cosine_similarity":      cos_sim.tolist(),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(filtered):
            print(f"  Processed {i+1}/{len(filtered)} prompts...")

    return results


def build_vectors_dict(model, corpus: list, n_layers: int,
                       hook_pattern: str, device: str,
                       category_filter: Optional[str] = None) -> dict:
    """
    Build { prompt_str: tensor(n_layers, d_model) } for PCA trajectory plots.
    Separate from run_dynamics_analysis() because PCA needs the full vectors,
    not just scalar curves.
    """
    filtered = corpus
    if category_filter:
        filtered = [e for e in corpus if e["category"] == category_filter]

    vectors = {}
    for entry in filtered:
        prompt = entry["prompt"]
        if prompt not in vectors:
            vectors[prompt] = extract_layer_vectors(
                model, prompt, n_layers, hook_pattern, device
            )
    return vectors


# ============================================================================
# PLOTTING
# ============================================================================

def _plot_mean_curve(ax, results: list, key: str,
                     n_transitions: int, label: str):
    """Helper: plot mean ± 1σ for base and contrast on a given axis."""
    transitions = list(range(n_transitions))

    for role, color, ls in [("base",     "#1565C0", "-"),
                              ("contrast", "#B71C1C", "--")]:
        curves = [r[key] for r in results if r["role"] == role]
        if not curves:
            continue
        arr = np.array(curves)
        mean, std = arr.mean(axis=0), arr.std(axis=0)
        ax.plot(transitions, mean, color=color, linestyle=ls,
                linewidth=2.0, label=f"{role} mean")
        ax.fill_between(transitions, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.set_xlabel("Layer transition")
    ax.set_ylabel(label)
    ax.set_xticks(transitions)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def plot_dynamics_overview(results: list, output_dir: Path,
                           model_name: str = ""):
    """
    Three-panel overview: speed, acceleration magnitude, cosine similarity.
    All on the same layer-transition x-axis.
    Primary summary plot for residual stream dynamics.
    """
    n_speed = len(results[0]["speed"])
    n_accel = len(results[0]["acceleration_magnitude"])
    n_cos   = len(results[0]["cosine_similarity"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    title = "Residual stream dynamics — all categories (mean ± 1σ)"
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=12)

    _plot_mean_curve(axes[0], results, "speed",
                     n_speed, "Speed  ||ΔX_L||₂")
    axes[0].set_title("Speed")

    _plot_mean_curve(axes[1], results, "acceleration_magnitude",
                     n_accel, "Acceleration magnitude  ||ΔV_L||₂")
    axes[1].set_title("Acceleration magnitude")

    _plot_mean_curve(axes[2], results, "cosine_similarity",
                     n_cos, "Cosine similarity  cos(X_L, X_{L+1})")
    axes[2].set_title("Trajectory curvature")
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    axes[2].axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    out_path = output_dir / f"dynamics_overview{suffix}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_dynamics_category(results: list, category: str,
                           output_dir: Path, model_name: str = ""):
    """
    Per-category three-panel dynamics plot.
    Individual pairs shown as faint lines, mean as bold.
    """
    cat_results = [r for r in results if r["category"] == category]
    if not cat_results:
        return

    by_pair = defaultdict(dict)
    for r in cat_results:
        by_pair[r["pair_id"]][r["role"]] = r

    PAIR_COLORS = list(plt.cm.tab10.colors) + list(plt.cm.Set2.colors[:5])

    keys = [
        ("speed",                  "Speed  ||ΔX_L||₂"),
        ("acceleration_magnitude", "Acceleration  ||ΔV_L||₂"),
        ("cosine_similarity",      "Cosine similarity"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title = f"Residual stream dynamics — {category}"
    if model_name:
        title += f"  [{model_name}]"
    fig.suptitle(title, fontsize=12)

    for ax, (key, ylabel) in zip(axes, keys):
        all_base, all_contrast = [], []

        for pair_idx, (pair_id, roles) in enumerate(sorted(by_pair.items())):
            color = PAIR_COLORS[pair_idx % len(PAIR_COLORS)]
            for role, ls in [("base", "-"), ("contrast", "--")]:
                if role not in roles:
                    continue
                curve = roles[role][key]
                n = len(curve)
                ax.plot(range(n), curve, color=color, linestyle=ls,
                        linewidth=1.0, alpha=0.5)
                (all_base if role == "base" else all_contrast).append(curve)

        if all_base:
            ax.plot(range(len(all_base[0])), np.mean(all_base, axis=0),
                    color="black", linestyle="-", linewidth=2.2,
                    label="mean base", zorder=5)
        if all_contrast:
            ax.plot(range(len(all_contrast[0])), np.mean(all_contrast, axis=0),
                    color="dimgray", linestyle="--", linewidth=2.2,
                    label="mean contrast", zorder=5)

        ax.set_xlabel("Layer transition")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        if key == "cosine_similarity":
            ax.set_ylim(-1.05, 1.05)
            ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
            ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    plt.tight_layout()
    suffix = f"_{model_name}" if model_name else ""
    out_path = output_dir / f"dynamics_{category}{suffix}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out_path}")


def plot_pca_trajectory(vectors: dict, n_layers: int,
                        output_path: str = "pca_trajectory.png",
                        model_name: str = ""):
    """
    Fits PCA on all prompts × layers together.
    Plots each prompt as a trajectory through 2D PCA space,
    colored by layer index (plasma colormap: early=dark, late=bright).
    """
    prompt_list = list(vectors.keys())
    all_vecs = [vectors[p].numpy() for p in prompt_list]
    stacked = np.vstack(all_vecs)   # (n_prompts * n_layers, d_model)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(stacked)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    title = "PCA trajectory of residual stream across layers"
    if model_name:
        title += f"  [{model_name}]"

    for i, prompt in enumerate(prompt_list):
        color = COLORS[i % len(COLORS)]
        traj = projected[i * n_layers: (i + 1) * n_layers]

        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.7)
        sc = ax.scatter(traj[:, 0], traj[:, 1],
                        c=list(range(n_layers)), cmap="plasma",
                        s=40, zorder=3, edgecolors=color, linewidths=0.8)
        ax.annotate("L0", traj[0], fontsize=6, color=color,
                    xytext=(3, 3), textcoords="offset points")
        ax.annotate(f"L{n_layers-1}", traj[-1], fontsize=6, color=color,
                    xytext=(3, 3), textcoords="offset points")
        ax.plot([], [], color=color, linewidth=2, label=repr(prompt))

    plt.colorbar(sc, ax=ax, label="Layer index")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title(title)
    ax.legend(fontsize=8, bbox_to_anchor=(1.15, 1), loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"✓ PCA trajectory saved to {output_path}")


# ============================================================================
# SUMMARY
# ============================================================================

def print_summary(results: list):
    """Print mean speed and cosine similarity trajectories."""
    print("\n" + "=" * 60)
    print("DYNAMICS SUMMARY")
    print("=" * 60)

    for key, label in [("speed", "Speed ||ΔX||₂"),
                        ("cosine_similarity", "Cosine similarity")]:
        print(f"\n  {label}:")
        by_role = defaultdict(list)
        for r in results:
            by_role[r["role"]].append(r[key])

        for role in ["base", "contrast"]:
            curves = by_role[role]
            if not curves:
                continue
            mean = np.mean(curves, axis=0)
            peak_layer = int(np.argmax(mean))
            print(f"    {role.upper()} ({len(curves)} prompts): "
                  f"L0={mean[0]:.3f}  peak=L{peak_layer}({mean[peak_layer]:.3f})  "
                  f"L{len(mean)-1}={mean[-1]:.3f}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Residual stream trajectory dynamics"
    )
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to corpus JSON from corpus_gen.py "
                             "(if omitted, runs on default prompts)")
    parser.add_argument("--model", type=str, default="gpt2-small",
                        help="Model name (must be in setup.py MODEL_CONFIGS)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter to a single corpus category")
    parser.add_argument("--output-dir", type=str, default="figures",
                        help="Directory for plots and results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load model
    print(f"\nLoading model '{args.model}'...")
    from setup import load_model_and_sae, MODEL_CONFIGS
    model, _, cfg = load_model_and_sae(args.model, load_sae=False, device=args.device)
    print(f"✓ Model ready on {cfg['device']}")

    model_cfg = MODEL_CONFIGS[args.model]
    hook_pattern = model_cfg.get("hook_pattern", "blocks.{layer}.hook_resid_pre")
    n_layers = model.cfg.n_layers
    print(f"  Layers: {n_layers}  Hook: {hook_pattern}")

    # ── Corpus mode ───────────────────────────────────────────────────────────
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"✗ Corpus not found: {corpus_path}")
            print("  Run: python corpus_gen.py")
            return 1

        with open(corpus_path) as f:
            corpus = json.load(f)
        print(f"\n✓ Loaded corpus: {len(corpus)} prompts ({len(corpus)//2} pairs)")

        print(f"\nComputing residual stream dynamics across {n_layers} layers...")
        results = run_dynamics_analysis(
            model, corpus, n_layers, hook_pattern,
            cfg["device"], args.category
        )

        results_path = output_dir / f"dynamics_results_{args.model}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        print_summary(results)

        if not args.no_plots:
            print(f"Generating plots in {output_dir}/...")
            plot_dynamics_overview(results, output_dir, args.model)
            categories = sorted(set(r["category"] for r in results))
            for cat in categories:
                plot_dynamics_category(results, cat, output_dir, args.model)

            # PCA requires full vectors — build separately
            print("  Building vectors for PCA...")
            vectors = build_vectors_dict(
                model, corpus, n_layers, hook_pattern,
                cfg["device"], args.category
            )
            plot_pca_trajectory(
                vectors, n_layers,
                str(output_dir / f"pca_trajectory_{args.model}.png"),
                args.model
            )

    # ── Default prompts mode (quick standalone test) ──────────────────────────
    else:
        print(f"\nNo corpus provided — running on default prompts...")
        vectors = {}

        for prompt in DEFAULT_PROMPTS:
            vecs = extract_layer_vectors(
                model, prompt, n_layers, hook_pattern, cfg["device"]
            )
            vectors[prompt] = vecs

            velocity = compute_velocity(vecs)
            speed    = compute_speed(velocity)
            cos_sim  = compute_cosine_similarity(vecs)

            print(f"  '{prompt}':")
            print(f"    speed:   {[f'{v:.2f}' for v in speed.tolist()]}")
            print(f"    cos_sim: {[f'{v:.3f}' for v in cos_sim.tolist()]}")

        if not args.no_plots:
            print(f"\nGenerating plots in {output_dir}/...")

            # Speed plot for default prompts
            fig, ax = plt.subplots(figsize=(9, 5))
            for i, (prompt, vecs) in enumerate(vectors.items()):
                velocity = compute_velocity(vecs)
                speed = compute_speed(velocity)
                n = len(speed)
                ax.plot(range(n), speed.tolist(), marker="o", markersize=4,
                        linewidth=1.8, color=COLORS[i % len(COLORS)],
                        label=repr(prompt))
            ax.set_xlabel("Layer transition")
            ax.set_ylabel("Speed  ||ΔX_L||₂")
            ax.set_title("Residual stream speed across layers")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
            plt.tight_layout()
            out = str(output_dir / "dynamics_speed_default.png")
            plt.savefig(out, dpi=130, bbox_inches="tight")
            plt.close()
            print(f"✓ {out}")

            plot_pca_trajectory(
                vectors, n_layers,
                str(output_dir / "pca_trajectory_default.png")
            )

    print(f"\nDone. Results in {output_dir}/\n")


if __name__ == "__main__":
    main()
