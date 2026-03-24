"""
renyi_spectrum.py

Computes the Rényi entropy spectrum H_q across transformer layers,
using the principal-component variance distribution of the residual stream.

For a discrete probability distribution p_i (here: normalized squared
singular values of the activation matrix), the Rényi entropy of order q is:

    H_q = (1 / (1 - q)) * log2( sum_i p_i^q )

Special cases handled analytically:
    q = 0  ->  log2(rank)          [log of number of nonzero components]
    q = 1  ->  -sum_i p_i log2(p_i) [Shannon entropy, limit of H_q as q->1]
    q = inf -> -log2(max_i p_i)     [min-entropy, dominated by largest p]

The q-spectrum H_q(layer) is a 2D surface. Its shape across q reveals
whether representations are monofractal (flat spectrum) or multifractal
(sloped spectrum with richer internal structure).

Outputs:
    renyi_spectrum.npz     -- arrays for analysis
    renyi_heatmap.png      -- H_q(q, layer) as a heatmap
    renyi_crossover.png    -- H_q vs layer for selected q values
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from transformer_lens import HookedTransformer

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_NAME   = "gpt2"
N_LAYERS     = 12          # GPT-2 small
DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR      = "."

# q values to sweep. Dense near 1 because the slope around q=1 is diagnostic.
Q_VALUES = np.array([
    0.0,
    0.25, 0.5, 0.75,
    1.0,                   # Shannon (computed via limit, not formula)
    1.25, 1.5, 1.75,
    2.0,                   # Collision entropy
    3.0, 5.0, 10.0,
    np.inf,                # Min-entropy
])

# Stability floor: singular values smaller than this fraction of the largest
# are treated as zero. Prevents 0^q numerical noise for q < 1.
EPS_FRAC = 1e-6

# ── Corpus ───────────────────────────────────────────────────────────────────
# Same contrast-pair structure as your existing corpus_gen.py.
# Each tuple is (base_prompt, contrast_prompt).

PROMPT_PAIRS = [
    ("The dog", "The cat"),
    ("Paris is the capital", "Berlin is the capital"),
    ("Water boils at", "Water freezes at"),
    ("The sun rises in the", "The sun sets in the"),
    ("One plus one equals", "Two minus one equals"),
    ("She opened the door and", "She closed the door and"),
    ("The quick brown fox", "The slow white rabbit"),
    ("In the beginning", "At the end"),
    ("Scientists discovered that", "Politicians claimed that"),
    ("The largest planet is", "The smallest planet is"),
    ("To be or not to", "To live and not to"),
    ("The patient was diagnosed with", "The patient recovered from"),
]

BASE_PROMPTS     = [p for p, _ in PROMPT_PAIRS]
CONTRAST_PROMPTS = [p for _, p in PROMPT_PAIRS]


# ── Core functions ────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str) -> HookedTransformer:
    print(f"Loading {model_name} on {device} ...")
    model = HookedTransformer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model


def get_residual_streams(
    model: HookedTransformer,
    prompts: list[str],
    n_layers: int,
    device: str,
) -> np.ndarray:
    """
    Returns array of shape (n_layers, n_prompts, d_model).

    For each prompt we take the residual stream at the *last token position*
    at hook_resid_pre for layers 0..n_layers-1. This matches the convention
    in your entropy_residual_stream.py.
    """
    hook_names = [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)]
    n_prompts  = len(prompts)
    d_model    = model.cfg.d_model

    activations = np.zeros((n_layers, n_prompts, d_model), dtype=np.float32)

    for p_idx, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt).to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_names)
        for l_idx, name in enumerate(hook_names):
            # cache[name] shape: (1, seq_len, d_model) — take last token
            vec = cache[name][0, -1, :].float().cpu().numpy()
            activations[l_idx, p_idx, :] = vec

    return activations


def pca_distribution(act_matrix: np.ndarray, eps_frac: float = EPS_FRAC) -> np.ndarray:
    """
    Given act_matrix of shape (n_prompts, d_model), compute the probability
    distribution over principal components:

        p_i = sigma_i^2 / sum_j sigma_j^2

    where sigma_i are the singular values of act_matrix (mean-centered).
    This is the same distribution used for effective rank in your existing code.

    Returns p of shape (min(n_prompts, d_model),), already normalized.
    """
    # Mean-center across prompts (removes the DC component so we measure
    # variance structure, not mean offset — analogous to removing the
    # time-mean before computing EOF variance in your climate work)
    X = act_matrix - act_matrix.mean(axis=0, keepdims=True)

    # SVD: we only need singular values, so use full_matrices=False
    _, s, _ = np.linalg.svd(X, full_matrices=False)

    # Convert to variance fractions (p_i = sigma_i^2 / sum sigma_j^2)
    s2 = s ** 2
    total = s2.sum()
    if total < 1e-12:
        # Degenerate case: all activations identical
        return np.ones(1, dtype=np.float32)

    p = s2 / total

    # Zero out components below numerical floor
    threshold = eps_frac * p.max()
    p = np.where(p >= threshold, p, 0.0)

    # Re-normalize after zeroing (keeps sum = 1)
    p = p / p.sum()

    return p.astype(np.float32)


def renyi_entropy(p: np.ndarray, q: float) -> float:
    """
    Rényi entropy H_q for distribution p (1-D array, sums to 1).

    Uses log base 2 so units are bits — consistent with your existing
    effective-rank entropy.
    """
    # Keep only non-zero entries (0^q = 0 for q>0, but log(0) is -inf)
    p = p[p > 0]

    if q == 0.0:
        # H_0 = log2(number of non-zero components) = log2(support size)
        return float(np.log2(len(p)))

    if q == 1.0:
        # Shannon entropy (limit of H_q as q -> 1)
        return float(-np.sum(p * np.log2(p)))

    if np.isinf(q):
        # Min-entropy: -log2(max p_i)
        return float(-np.log2(p.max()))

    # General case: H_q = (1/(1-q)) * log2(sum p_i^q)
    sum_pq = np.sum(p ** q)
    if sum_pq <= 0:
        return 0.0
    return float((1.0 / (1.0 - q)) * np.log2(sum_pq))


def compute_spectrum(
    activations: np.ndarray,
    q_values: np.ndarray,
) -> np.ndarray:
    """
    Compute H_q for all layers and all q values.

    activations: shape (n_layers, n_prompts, d_model)
    q_values:    shape (n_q,)

    Returns spectrum of shape (n_layers, n_q).
    """
    n_layers = activations.shape[0]
    n_q      = len(q_values)
    spectrum = np.zeros((n_layers, n_q), dtype=np.float32)

    for l in range(n_layers):
        p = pca_distribution(activations[l])          # (n_components,)
        for qi, q in enumerate(q_values):
            spectrum[l, qi] = renyi_entropy(p, q)

    return spectrum


# ── Plotting ──────────────────────────────────────────────────────────────────

def q_label(q: float) -> str:
    if q == 0.0:
        return "q=0"
    if q == 1.0:
        return "q=1 (Shannon)"
    if q == 2.0:
        return "q=2 (collision)"
    if np.isinf(q):
        return "q=∞ (min-entropy)"
    return f"q={q:g}"


def plot_heatmap(
    base_spectrum: np.ndarray,
    contrast_spectrum: np.ndarray,
    q_values: np.ndarray,
    out_path: str,
):
    """
    Side-by-side heatmaps: H_q(layer, q) for base and contrast prompts.
    x-axis: layer (0..11), y-axis: q index (displayed as q labels).
    Color: entropy in bits.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    fig.suptitle("Rényi entropy spectrum H_q across layers", fontsize=13)

    q_labels = [q_label(q) for q in q_values]

    # Use the same color scale for both panels so they're directly comparable
    vmin = min(base_spectrum.min(), contrast_spectrum.min())
    vmax = max(base_spectrum.max(), contrast_spectrum.max())

    for ax, spectrum, title in zip(
        axes,
        [base_spectrum, contrast_spectrum],
        ["Base prompts", "Contrast prompts"],
    ):
        im = ax.imshow(
            spectrum.T,                  # shape (n_q, n_layers) — q on y, layer on x
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_xticks(range(base_spectrum.shape[0]))
        ax.set_xticklabels(range(base_spectrum.shape[0]), fontsize=8)
        ax.set_yticks(range(len(q_labels)))
        ax.set_yticklabels(q_labels, fontsize=8)

    axes[0].set_ylabel("q  (order of Rényi entropy)", fontsize=10)
    fig.colorbar(im, ax=axes, label="H_q  (bits)", shrink=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap → {out_path}")


def plot_crossover(
    base_spectrum: np.ndarray,
    contrast_spectrum: np.ndarray,
    q_values: np.ndarray,
    selected_q_indices: list[int],
    out_path: str,
):
    """
    Line plots of H_q vs layer for selected q values.
    Base prompts = solid lines, contrast prompts = dashed lines.
    Each selected q gets its own color.

    This is the analog of your existing entropy_crossover plot, but now
    showing how the crossover shifts as q changes — the key multifractal
    diagnostic.
    """
    n_layers = base_spectrum.shape[0]
    layers   = np.arange(n_layers)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected_q_indices)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for color, qi in zip(colors, selected_q_indices):
        q   = q_values[qi]
        lbl = q_label(q)
        ax.plot(layers, base_spectrum[:, qi],
                color=color, lw=2.0, linestyle="-",
                label=f"Base — {lbl}")
        ax.plot(layers, contrast_spectrum[:, qi],
                color=color, lw=2.0, linestyle="--",
                label=f"Contrast — {lbl}")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("H_q  (bits)", fontsize=11)
    ax.set_title("Rényi entropy H_q vs layer for selected q", fontsize=12)
    ax.set_xticks(layers)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Shade the phase-transition region you identified (layers 4-5)
    ax.axvspan(3.5, 5.5, alpha=0.08, color="gray", label="Phase transition region")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved crossover plot → {out_path}")


def plot_multifractal_width(
    base_spectrum: np.ndarray,
    contrast_spectrum: np.ndarray,
    q_values: np.ndarray,
    out_path: str,
):
    """
    Plots the 'multifractal width' W(layer) = H_q0(layer) - H_qinf(layer)
    for base and contrast prompts.

    A monofractal system has W ~ 0 (all q give the same entropy).
    A multifractal system has W > 0. A sharp change in W across layers
    is a diagnostic of representational restructuring.

    This is analogous to measuring the width of the f(alpha) singularity
    spectrum in classical multifractal analysis of aerosols.
    """
    # Find indices for q=0 and q=inf
    q0_idx  = np.where(q_values == 0.0)[0][0]
    inf_idx = np.where(np.isinf(q_values))[0][0]

    base_width     = base_spectrum[:, q0_idx]     - base_spectrum[:, inf_idx]
    contrast_width = contrast_spectrum[:, q0_idx] - contrast_spectrum[:, inf_idx]

    layers = np.arange(base_spectrum.shape[0])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, base_width,     lw=2.5, label="Base prompts",     color="#2166ac")
    ax.plot(layers, contrast_width, lw=2.5, label="Contrast prompts", color="#d6604d",
            linestyle="--")
    ax.axvspan(3.5, 5.5, alpha=0.08, color="gray")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("W = H₀ − H∞  (bits)", fontsize=11)
    ax.set_title("Multifractal width across layers", fontsize=12)
    ax.set_xticks(layers)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved multifractal width → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    model = load_model(MODEL_NAME, DEVICE)

    print(f"\nExtracting residual streams for {len(BASE_PROMPTS)} base prompts ...")
    base_acts = get_residual_streams(model, BASE_PROMPTS, N_LAYERS, DEVICE)

    print(f"Extracting residual streams for {len(CONTRAST_PROMPTS)} contrast prompts ...")
    contrast_acts = get_residual_streams(model, CONTRAST_PROMPTS, N_LAYERS, DEVICE)

    print(f"\nComputing Rényi spectrum for q ∈ {Q_VALUES} ...")
    base_spectrum     = compute_spectrum(base_acts,     Q_VALUES)
    contrast_spectrum = compute_spectrum(contrast_acts, Q_VALUES)

    # Save raw arrays for later analysis
    npz_path = os.path.join(OUT_DIR, "renyi_spectrum.npz")
    np.savez(
        npz_path,
        base_spectrum=base_spectrum,
        contrast_spectrum=contrast_spectrum,
        q_values=Q_VALUES,
    )
    print(f"Saved arrays → {npz_path}")

    # Heatmap
    plot_heatmap(
        base_spectrum, contrast_spectrum, Q_VALUES,
        os.path.join(OUT_DIR, "renyi_heatmap.png"),
    )

    # Crossover lines for q = 0, 1 (Shannon), 2, inf
    selected = [
        np.where(Q_VALUES == 0.0)[0][0],
        np.where(Q_VALUES == 1.0)[0][0],
        np.where(Q_VALUES == 2.0)[0][0],
        np.where(np.isinf(Q_VALUES))[0][0],
    ]
    plot_crossover(
        base_spectrum, contrast_spectrum, Q_VALUES, selected,
        os.path.join(OUT_DIR, "renyi_crossover.png"),
    )

    # Multifractal width
    plot_multifractal_width(
        base_spectrum, contrast_spectrum, Q_VALUES,
        os.path.join(OUT_DIR, "renyi_multifractal_width.png"),
    )

    # Print summary table to console
    print("\n── Spectrum summary (bits) ──────────────────────────────────────")
    q_cols = [0.0, 1.0, 2.0, np.inf]
    col_idx = [np.where(Q_VALUES == q)[0][0] if not np.isinf(q)
               else np.where(np.isinf(Q_VALUES))[0][0]
               for q in q_cols]
    header = f"{'Layer':>6}  " + "  ".join(f"{'Base H'+q_label(q):>16}{'Contrast H'+q_label(q):>16}"
                                             for q in q_cols)
    print(header)
    for l in range(N_LAYERS):
        row = f"{l:>6}  "
        for qi in col_idx:
            row += f"{base_spectrum[l,qi]:>16.3f}{contrast_spectrum[l,qi]:>16.3f}"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
