# Residual Stream Dynamics

Mechanistic analysis of residual stream geometry and unembedding-matrix
subspace structure in small open-weight LLMs (GPT-2, Pythia).

## Where to start

The main artifact is the notebook:
**[`notebooks/residual_stream_dynamics.ipynb`](https://nbviewer.org/github/storyofthewolf/residual-stream-dynamics/blob/main/notebooks/residual_stream_dynamics.ipynb)**

It renders standalone via nbviewer with all figures embedded — no execution
required for reading. The notebook walks through the central findings
of the project, the W_U subspace decomposition, the two ablation stages
(post-hoc and forward-pass intervention), and the synthesis.

## What this project investigates

Motivated by my background in numerical climate modeling, I am interested
in studying the flow of information through LLMs through the lens of
physics-based intuitions, with a lean toward the geometrical representations
of the residual stream. Beginning with entropy calculations, the driving
questions emerged of, *what parts of the residual stream are doing the work
of next token prediction?* and *what information content is carried by the
tails of the residual stream energy distributions?* This project
develops a workflow for analyzing residual stream geometry and
unembedding-matrix subspace structure across the GPT-2 (small–XL) and
Pythia (160M–6.9B) model families using TransformerLens.

## Key findings

1. **Anti-correlation of entropy and logit lens certainty**: Coherent (base)
   prompts show *higher* residual stream entropy yet *lower* logit lens entropy
   than ambiguous (contrast) prompts. In terms of geometry, this suggests that
   the residual stream utilizes a larger space for coherent prompts, but
   converges on a more certain prediction after transformation through the
   unembedding matrix. This raises the question: what information is stored in
   the residual stream for base prompts that is not present for contrast prompts,
   and how does it affect next-token prediction?

2. **Cross-architecture replication**: The anti-correlation between residual
   stream and logit lens entropy holds across GPT-2 (small–XL) and Pythia
   (160M–6.9B) models of increasing parameter count, suggesting this may be a
   general property of transformer architectures.

3. **Fragility at 99% explained variance**: We decompose the unembedding matrix
   W_U via SVD and ablate the bottom-k subspace components from the residual
   stream. Next-token predictions are surprisingly sensitive: approximately 20%
   of prompts change their top-1 predicted token when ablating only the bottom
   1% of explained variance. This holds in both post-hoc and full intervention
   experiments.

4. **Differential sensitivity**: Base prompts are more stable than contrast
   prompts under ablation of the low-rank complement of W_U. This presents an
   apparent paradox — base prompts contain more information at low rank (higher
   residual stream entropy) yet are less sensitive to ablating that complement.

This is a pilot-scale methods demonstration on a small corpus (25 base /
25 contrast prompt pairs), not a finished results paper. See the notebook
for full discussion and `FutureWork.md` for ongoing directions.

## Reproducing the notebook

To keep the repository clean, `.npz` and `.png` files are not committed.
Precomputed files are available here:
- [Precomputed .npz files](https://drive.google.com/drive/folders/1dhfdz3xhMUrZ2StbQWpE7WTx7IP7Z3_l?usp=sharing)
- [Notebook figures](https://drive.google.com/drive/folders/14BkETR9IoxGrwzEcMWljHJSD8LKN9uKc?usp=sharing)

To regenerate data from scratch (or produce new data for different models,
corpora, etc.) run the scripts in `workflows/` (see "Running the workflows"
below). Workflow execution for small models can be done on a MacBook Pro with
~36 GB RAM using the TransformerLens library and standard Python libraries
(numpy, torch, matplotlib, etc.).

## Project structure

```
.
├── src/                         # Core computation modules
│   ├── math_utils.py            # Shared math: renyi_entropy, compute_wu_svd
│   ├── extraction.py            # ActivationRecord dataclass and forward pass
│   ├── entropy_compute.py       # EntropyRecord dataclass, entropy computation
│   ├── ck_spectrum_compute.py   # CkRecord dataclass, c_k computation, serialization
│   ├── ablation_compute.py      # AblationRecord dataclass and ablation computation
│   └── mechanics_compute.py     # MechanicsRecord dataclass, trajectory mechanics
├── plotting/                    # Visualization modules
│   ├── entropy_plots.py         # Entropy visualization
│   ├── ablation_plots.py        # Ablation visualization
│   ├── ck_spectrum_plots.py     # c_k spectrum visualization
│   ├── mechanics_plots.py       # Mechanics visualization
│   └── post_process_plots.py    # Curated notebook figures from stored .npz files
├── dashboard/                   # Interactive Gradio dashboard
│   ├── dashboard.py             # App entry point (no computation)
│   ├── dashboard_loader.py      # NPZ discovery, loading, caching, and index building
│   └── dashboard_viz.py         # Figure-generating functions for the dashboard
├── workflows/                   # Orchestration scripts — argument parsing, I/O only
│   ├── entropy_analysis.py      # Residual stream and logit lens entropy experiments
│   ├── ablation_analysis.py     # Post-hoc and intervention ablation experiments
│   ├── ck_analysis.py           # c_k spectrum analysis (SVD logit decomposition)
│   ├── wu_subspace_analysis.py  # SVD decomposition of residual stream @ W_U
│   ├── mechanics_analysis.py    # Residual stream trajectory mechanics
│   └── single_prompt.py         # Single-prompt entropy probe
├── utils/                       # Utilities — no computation, no torch
│   ├── model_loader.py          # Model registry, loading, and introspection
│   ├── npz_utils.py             # Loading and filtering of .npz data
│   └── npz_quicklook.py         # Quick inspection of .npz file contents
├── corpus/                      # Prompt corpus files
│   ├── base_vs_contrast_n50.json
│   └── corpus_gen.py            # Corpus generation script
├── notebooks/                   # Jupyter notebooks of results
├── data/                        # Precomputed .npz results
│   ├── entropy/                 # Entropy surface records
│   ├── wu_subspace/             # W_U subspace projection records
│   ├── ablation/                # Ablation records (posthoc and intervention)
│   ├── ck/                      # c_k spectrum records
│   └── mechanics/               # Residual stream mechanics records
├── figures/                     # Generated plots
│   ├── workflows/               # Auto-generated from workflow scripts
│   └── notebooks/               # Generated by post-processing .npz files
└── sandbox/                     # Exploratory scripts
```

## Running the workflows

Each workflow script in `workflows/` is a standalone driver that calls into the
core `*_compute.py` and `*_plots.py` modules. Run any script with `--help` for
full argument documentation.

### Single-prompt entropy probe

```bash
python workflows/single_prompt.py --model gpt2-small
```

### Corpus-driven entropy analysis

```bash
python workflows/entropy_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model gpt2-small \
    --logit-lens \
    --save-data
```

### W_U subspace analysis

```bash
python workflows/wu_subspace_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model gpt2-small \
    --also-residual --also-logit-lens \
    --save-data
```

### Ablation experiments (post-hoc and intervention)

```bash
python workflows/ablation_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model gpt2-small \
    --ev-thresholds 0.1 0.25 0.50 0.75 0.90 0.95 0.99 0.999 1.0 \
    --stage2 \
    --save-data
```

### c_k spectrum analysis

```bash
python workflows/ck_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model gpt2-small \
    --save-data
```

Computes the exact logit decomposition `c_k = σ_k · (r · v_k)` where v_k and σ_k are
the k-th right singular vector and singular value of W_U. Produces nine figures per hook
type:

1. **2×2 diagnostic** — raw projections, sorted projections, c_k spectrum, sorted |c_k|
   for a single `(layer, token)`.
2. **Last-token heatmap** — mean |c_k| at the final token, log-scale colormap, layers
   1–end, stratified by role (base / contrast / difference).
3. **All-tokens heatmap** — same but averaged over all token positions (secondary).
4. **Spectral CoM vs. layer** — center of mass `Σ_k k·c_k² / Σ_k c_k²` at the last
   token, mean ± std across prompts, base and contrast lines.
5. **Cumulative power fraction** — `F(K) = Σ_{k<K} c_k² / Σ_k c_k²` vs k for a
   user-selected set of layers (`--summary-layers`, default `[1, 3, 6, 9, 11]`).
6. **Layer-to-layer |Δc_k| heatmap** — absolute change per singular direction between
   adjacent layers (transitions 1→2, 2→3, …), stratified by role.
7. **Prompt-variance heatmap** — `Var(|c_k|)` across prompts at the last token, plus
   `log₁₀(Var_base / Var_contrast)` as a diverging colormap.
8. **Variance ratio vs. k** — `log₁₀(Var_base / Var_contrast)` as a line plot for
   selected layers, showing where base prompts are more or less variable than contrast.
9. **CoM of prompt-variance vs. layer** — spectral center of mass of the prompt-variance
   grid, base vs. contrast, as a function of layer.

Use `--summary-layers` to choose which layers appear in the cumulative power and variance
ratio plots, and `--no-skip-layer0` to include layer 0 (excluded by default).

### Residual stream mechanics

```bash
python workflows/mechanics_analysis.py \
    --model gpt2-small \
    --save-data
```

Interprets the residual stream as a discrete particle trajectory in d_model-dimensional
space. Computes five scalar curves at the final token position across layers:

1. **Speed** — ||ΔX_l||₂, the L2 norm of each layer-to-layer displacement.
2. **Acceleration magnitude** — ||ΔV_l||₂, the norm of the velocity change between layers.
3. **State cosine similarity** — cos(X_l, X_{l+1}), curvature of the trajectory.
4. **Update–state alignment** — cos(ΔX_l, X_l), whether updates amplify the current state
   or inject orthogonal content.
5. **Update coherence** — cos(ΔX_l, ΔX_{l+1}), whether consecutive updates are aligned.

All quantities are computed in pure numpy from stored ActivationRecords — no re-extraction
required if ActivationRecords are already saved. Produces two figure types per run:
an overview (mean ± 1σ across all pairs) and one per-category plot with individual pair
curves shown as faint lines behind the mean.

### Full notebook reproduction (all models)

```bash
MODEL=gpt2-small   # repeat for gpt2-medium, gpt2-large, gpt2-xl, pythia-160m, ...

python workflows/entropy_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model $MODEL --logit-lens --save-data

python workflows/wu_subspace_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model $MODEL --also-residual --also-logit-lens --save-data

python workflows/ablation_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model $MODEL \
    --ev-thresholds 0.1 0.25 0.50 0.75 0.90 0.95 0.99 0.999 1.0 \
    --stage2 --intervention-stride 1 --save-data

python workflows/ck_analysis.py \
    --corpus corpus/base_vs_contrast_n50.json \
    --model $MODEL --save-data
```

All workflows support `--load-data <path>` to skip re-extraction and go
directly to plotting from a previously saved `.npz` file.

## Hardware notes

Development was done on a MacBook Pro M3 Max (36 GB unified memory).
All eight tested models (GPT-2 small–XL, Pythia 160M–6.9B) run on this
hardware with the small pilot corpus. Larger corpora and larger models are
bounded by memory rather than compute. In particular,
`ablation_analysis.py --stage2 --intervention-stride 1` for Pythia-6.9B
consumes most of the available memory and takes approximately 4 hours on
the pilot corpus.

**macOS note**: PyPI PyTorch wheels link against the Accelerate framework
and are unstable on macOS 15 (Sequoia). Install PyTorch via conda
(which uses OpenBLAS) to avoid this:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

## Interactive dashboard

A Gradio dashboard is included for rapid visual exploration of precomputed `.npz`
results — no model inference, no torch.

```bash
pip install gradio
python dashboard/dashboard.py --data-root data/
```

Opens at `http://localhost:7860` with four tabs:

| Tab | What it shows |
|-----|---------------|
| **Entropy** | Mean ± SEM entropy vs. layer, base vs. contrast, for any model / norm key / α |
| **WU Subspace** | Same for the r‖ and r⊥ subspace projections, sweepable over rank k |
| **Ablation** | Posthoc curves (top-1 preservation, KL divergence) and intervention heatmap (k × layer) |
| **C_k Spectra** | Mean \|c_k\| ± SEM at a selected layer, with the σ_k spectrum below |
| **Mechanics** | Speed, acceleration, and cosine similarity curves, base vs. contrast |

Each tab has an **Update Plot** button; model dropdowns chain-update their dependent
selectors (norm key, α, k) automatically. Run with `--share` for a temporary public URL.

The dashboard expects data in the layout produced by the workflows:
```
data/
  entropy/       ← entropy_analysis.py --save-data
  wu_subspace/   ← wu_subspace_analysis.py --save-data
  ablation/      ← ablation_analysis.py --save-data
  ck/            ← ck_analysis.py --save-data
  mechanics/     ← mechanics_analysis.py --save-data
```

## Status and roadmap

Current state: pilot-scale workflow demonstration with results summarized in
the main notebook. The pipeline covers entropy surfaces, W_U subspace projections,
post-hoc and intervention ablation, c_k spectrum analysis, and residual stream
trajectory mechanics across GPT-2 (small–XL) and Pythia (160M–6.9B). An interactive
Gradio dashboard (five tabs) provides rapid visual exploration of all precomputed
results without re-running any model inference.

Active and planned directions are tracked in `FutureWork.md`, including:

- Corpus expansion beyond the n=50 pilot
- Deeper interpretation of the mechanics curves in relation to the entropy findings

This repository will continue to evolve as the project develops.

---

*Eric T. Wolf — University of Colorado, Laboratory for Atmospheric and Space Physics*
