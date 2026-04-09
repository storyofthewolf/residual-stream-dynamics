# Future Work

*Last updated: April 8th, 2026*

This document tracks planned and in-progress research and engineering directions
for the residual-stream-dynamics project. Items are roughly prioritized within
each section but the ordering is informal.  This is a working document.

---

## Science directions

### Compute per-direction logit influence spectrum (c_k)

For each prompt and each layer, compute the per-direction logit influence:

> c_k = σ_k · (r · v_k)

where σ_k is the k-th singular value of W_U and v_k is the k-th right singular
vector. This gives a magnitude spectrum of how much each W_U
singular direction contributes to the logits for a given residual stream state.

The flat W_U spectrum observed in the notebook tells us what directions *could*
matter; the c_k spectrum tells us what directions *do* matter, prompt by prompt.

This calculation applied over all layers of the network, will help us understand 
how energy flows through the system.  Our working hypothesis is that the models
first accumulate energy in all directions democractically, but concentrate energy 
back into high-σ directions for readout at the final layer.  This inherent energy
redistribution should show different shapes depending on the nature of the prompt.

To efficiently perform these experiments, will need to output a data cube of 
the residual stream at each layer, the W_U SVD.  Currently the raw residual 
stream is processed but not saved.


### Dynamical systems framework on residual stream trajectories

Motivated by the physical science background, consider the residual stream 
as the state of a discrete-time dynamical system, with the layer index as time 
and the layer updates akin to changing position, velocity, and acceleration.
Several quantities then become well-defined:

- **Layer-wise Jacobian norms** along base vs. contrast trajectories. Compute
  J_l = ∂f_l/∂r at each layer for the actual residual stream of base and
  contrast prompts; the eigenvalue spectrum reveals local amplification,
  damping, and rotation. Whether base and contrast prompts pass through
  systematically different Jacobian regimes is a sharp empirical question.
- **Finite-time Lyapunov exponents** from perturbed-prompt pairs. Track the
  divergence of nearby residual stream trajectories layer by layer. Different
  Lyapunov spectra for base vs. contrast prompts would mean the model treats
  them as living in dynamically distinct regions of state space.
- **Trajectory visualization** in low-dimensional projections (PCA, c_k basis,
  or top singular directions) to make the dynamics visible
  as a geometric pattern rather than only as a statistical claim.

These methods are standard tools in atmospheric predictability analysis and
transfer naturally to residual stream analysis. 

### Other science directions

- **Von Neumann entropy** of residual stream covariance matrices, as a
  quantum-information-flavored complement to the Shannon/Rényi entropy
  measurements already in the notebook.
- **Prompt entropy** — entropy of the input distribution itself, as a
  control variable for interpreting residual stream entropy differences.
- **Rényi spectrum and multifractal analysis** of residual stream activations.
  Multifractal scaling exponents τ(q) and the corresponding Rényi dimension
  spectrum D_q would distinguish monofractal from multifractal geometric
  structure, a genuine differentiator if base and contrast prompts have
  different multifractal signatures.
- **Randomization controls** — replace residual stream content with
  shuffled or randomized versions and measure how predictions degrade.
  A baseline against which the structured-content claims can be compared.
- **Logit gap analysis** per layer — the pre-ablation margin between top-1
  and top-2 logits, stratified by base vs. contrast. May explain why some
  prompts are more sensitive to ablation than others.
- **Reverse-truncation ablation** — keep the *bottom* k W_U directions
  instead of the top k, and compare preservation curves. Tests whether
  signal is concentrated or distributed across the spectrum.

---

## Engineering and infrastructure

### Data pipeline

- **Save additional metadata to `.npz` records.** Add `d_model` and `n_layers`
  as scalar metadata fields in `save_ablation_records` / `load_ablation_records`.
  This would let `plot_top1_intervention_tripanel` use `x_mode='relative'`
  without the notebook needing the `D_MODEL` lookup dictionary.
- **Save `ev_fractions` array to ablation `.npz`** when `--ev-thresholds` is
  used (parallel to the existing `ks` array). Would enable `y_mode='ev'` in
  the intervention tripanel plot without requiring a separate
  `wu_subspace_analysis` run to recover the explained-variance mapping.
- **Save `explained variance vs. k-subspace rank` arrays**, currently plotted
  but not persisted. Recomputable but cheap to store.
- **Save raw residual stream and logit lens data cubes** for post-processing,
  to enable analyses like the c_k spectrum experiment without re-running
  the workflow scripts. Expense to recompute.
- **Corpus expansion.** The current n=50 pilot corpus is sufficient for
  workflow demonstration but produces wide confidence intervals on most
  measurements. Target: 100–200 prompt pairs per category, with uniform
  k-grids across models for cleaner cross-architecture comparison. Expand 
  to different types of corpus patterns.

### Refactoring

These are deferred clean-up items, listed for completeness. The current code
works; these are quality-of-life improvements for when the project leaves
pilot scale.

- **Unify duplicated SVD utilities.** `wu_explained_variance()` and
  `compute_wu_svd()` are duplicated in `ablation_compute.py` and
  `entropy_compute.py` with subtly different argument structures. Consolidate
  into a shared `svd_utils.py` module.  The two callers have slightly different 
  needs but should be combined.
- **Standardize plot file-path arguments.** `entropy_plots.py` functions take
  `output_dir` plus an optional `filename` and merge them internally;
  `ablation_plots.py` functions take a single pre-merged `save_path` from
  the workflow scripts. Pick one convention and apply it consistently.
- **Separate exploratory diagnostic plots from curated presentation plots.**
  `entropy_plots.py` and `ablation_plots.py` produce multi-panel diagnostic
  figures during workflow runs, while `post_process_plots.py` ingests `.npz`
  files to produce bespoke presentation figures. Some low-level helpers
  (`_mean_and_ci`, `_diff_and_ci`, `_save`, color constants, `_fdr_bh`)
  are duplicated across both. Eventually move shared helpers into a common
  plotting utility module.
- **More robust data and plot naming conventions** if parameter sweeps
  proliferate further. `entropy_plots.py` functions currently take different
  argument lists for corpus-driven vs. single-prompt usage. The
  `{corpus_tag}_{run_tag}` suffix pattern is a stop-gap; a cleaner solution
  would be a single naming scheme defined in one place.

---

## Notes on prioritization

1. Save residual stream and logit lens data cubes (one-time pipeline change)
2. Run the c_k spectrum experiment on existing data
3. Expand corpus to 100+ pairs per category.  Expensive experiments with gpt2-xl and pythia-2.8b and 6.9b need to be planned around my daily work schedule since they occupy my memory and significantly slow down my computer.
4. Begin Jacobian experiments on a small subset of prompts as a
   proof-of-concept for the dynamical systems direction
