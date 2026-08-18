# Future Work

*Last updated: August 17th, 2026*

This document tracks planned and in-progress research and engineering directions
for the residual-stream-dynamics project. Items are roughly prioritized within
each section but the ordering is informal. This is a working document.

---

## Current priorities

Compute now runs on Google Colab free tier (T4; the card reports ~14.6 GB
usable, not the nominal 16), so the practical ceiling is **pythia-2.8b** — and
that is the one model where an OOM is plausible. This is not a limitation for the science: the claims are
about residual stream geometry, not about model capability, and the existing
gpt2-xl / pythia-6.9b results already establish that the effect is not a
small-model artifact. Model scale is a settled question; the open questions are
statistical power and mechanism.

1. **Run the c_k spectrum experiment.** The infrastructure has been built and
   validated but has never produced a stored corpus result — `data/ck/` is
   empty. Use `--last-token-only` (8 of the 9 c_k figures need only the final
   token; it cuts the `.npz` ~9x). This is the build-then-cash-out hypothesis
   and it is the most mechanistic story the project has.
2. **Re-run the corpus workflows at n=108.** The expanded corpus
   (`corpus/base_vs_contrast_n216.json`) roughly halves the error bars. All
   existing results are at n=25 pairs.
3. **Stratify by `contrast_type`** — the new confound control (see below).
   This is the analysis the expanded corpus was designed to enable and it does
   not yet exist.
4. **Fill in mechanics coverage.** Mechanics exists for gpt2-small only, while
   ablation / entropy / wu_subspace cover eight models. Mechanics is pure numpy
   over stored activations — the cheapest analysis in the project.
5. **Begin Jacobian experiments** on a small subset of prompts as a
   proof-of-concept for the dynamical systems direction.

---

## Science directions

### Stratify results by contrast_type — NEW, HIGH PRIORITY

The original n=25 corpus built almost every contrast by appending a
low-frequency abstract noun ("philosophy", "democracy", "calculus"). This
confounds two variables: *the structure is broken* AND *an unusual token
appears*. A referee can attribute the entire r⊥ effect to lexical frequency
rather than to structure, and at n=5 per category that objection cannot be
answered.

`corpus_gen.py` now emits a `contrast_type` field with four levels:

| type | how the contrast breaks the base | controls for |
|---|---|---|
| `abstract` | low-frequency abstract noun (original design) | — (baseline) |
| `concrete` | high-frequency concrete noun | lexical frequency |
| `in_domain` | same semantic class, wrong position (`one two three seven`) | token identity *and* frequency |
| `swap` | same tokens, order destroyed | token identity exactly |

**The key comparison is `abstract` vs. `in_domain`.** If the r⊥ effect survives
in `in_domain` pairs — where the intruding token is perfectly ordinary and only
its *position* is wrong — then the effect is structural, not a frequency
artifact. If it vanishes, the original finding was largely lexical and the paper
needs rewriting. Either outcome is publishable; the current design cannot
distinguish them.

`swap` is the tightest control of all (token multiset held exactly fixed) but is
undefined for the `repetition` category, where shuffling produces an identical
string. Expect n=0 in that cell.

`contrast_type` is written to the corpus JSON but is **not** consumed by
`extraction.py`, which reads only `pair_id` / `role` / `category`. Stratified
analysis currently requires joining records back to the corpus file on `prompt`
or `pair_id`. Threading `contrast_type` through the Record types is a small
pipeline change (see Engineering below) and would make this a first-class
grouping variable.

### Per-direction logit influence spectrum (c_k) — BUILT, NOT YET RUN

For each prompt and layer, the per-direction logit influence:

> c_k = σ_k · (r · v_k)

where σ_k is the k-th singular value of W_U and v_k the k-th right singular
vector. The flat W_U spectrum tells us what directions *could* matter; the c_k
spectrum tells us what directions *do* matter, prompt by prompt.

Working hypothesis: models first accumulate energy across all directions
democratically, then concentrate it back into high-σ directions for readout at
the final layer. This redistribution should take different shapes depending on
the nature of the prompt.

`ck_spectrum_compute.py`, `ck_spectrum_plots.py` (nine figures including the
prompt-variance diagnostics), and `workflows/ck_analysis.py` are all complete
and tested. Preliminary single-model output does show energy concentrating into
low-k directions with depth, with the base−contrast difference localized in the
lowest-k directions at the deepest layers — consistent with the hypothesis, but
this needs a proper corpus run before it is a result.

**Not yet done:** a corpus run saved to `data/ck/` for any model.

### Dynamical systems framework on residual stream trajectories

Motivated by the physical science background, consider the residual stream as
the state of a discrete-time dynamical system, with layer index as time and
layer updates akin to changing position, velocity, and acceleration.

The scalar mechanics (speed, acceleration magnitude, three cosine similarity
curves) are **implemented** in `mechanics_compute.py` and run for gpt2-small.
The harder quantities remain open:

- **Layer-wise Jacobian norms** along base vs. contrast trajectories. Compute
  J_l = ∂f_l/∂r at each layer for the actual residual stream; the eigenvalue
  spectrum reveals local amplification, damping, and rotation. Whether base and
  contrast prompts pass through systematically different Jacobian regimes is a
  sharp empirical question.
- **Finite-time Lyapunov exponents** from perturbed-prompt pairs. Track the
  divergence of nearby residual stream trajectories layer by layer. Different
  Lyapunov spectra for base vs. contrast prompts would mean the model treats
  them as living in dynamically distinct regions of state space.
- **Trajectory visualization** in low-dimensional projections (PCA, c_k basis,
  or top singular directions), to make the dynamics visible as a geometric
  pattern rather than only as a statistical claim.

These are standard tools in atmospheric predictability analysis and transfer
naturally. Note on feasibility: a full `[d_model, d_model]` Jacobian per layer
is memory-hungry and fits Colab free tier badly. Start with
Jacobian-vector products (`torch.autograd.functional.jvp`) on a handful of
prompts rather than materializing the full matrix.

### Other science directions

- **Von Neumann entropy** of residual stream covariance matrices, as a
  quantum-information-flavored complement to the Shannon/Rényi measurements.
- **Prompt entropy** — entropy of the input distribution itself, as a control
  variable for interpreting residual stream entropy differences.
- **Rényi spectrum and multifractal analysis** of residual stream activations.
  Multifractal scaling exponents τ(q) and the Rényi dimension spectrum D_q
  would distinguish monofractal from multifractal geometric structure — a
  genuine differentiator if base and contrast prompts have different
  multifractal signatures.
- **Randomization controls** — replace residual stream content with shuffled or
  randomized versions and measure how predictions degrade. Note that the
  `swap` contrast_type now provides a *prompt-level* version of this control;
  this item is the activation-level counterpart.
- **Logit gap analysis** per layer — the pre-ablation margin between top-1 and
  top-2 logits, stratified by base vs. contrast. May explain why some prompts
  are more sensitive to ablation than others.
- **Reverse-truncation ablation** — keep the *bottom* k W_U directions instead
  of the top k, and compare preservation curves. Tests whether signal is
  concentrated or distributed across the spectrum.

---

## Engineering and infrastructure

### Data pipeline

- **Thread `contrast_type` through the Record types.** `extraction.py` reads
  only `pair_id` / `role` / `category` from corpus entries. Adding
  `contrast_type` alongside them (and to each `save_*`/`load_*` pair, with the
  usual backward-compatible default) would make the confound-control analysis a
  one-line filter instead of a join against the corpus JSON.
- **Save additional metadata to `.npz` records.** Add `d_model` and `n_layers`
  as scalar metadata in `save_ablation_records` / `load_ablation_records`, so
  `plot_top1_intervention_tripanel` can use `x_mode='relative'` without the
  notebook needing a `D_MODEL` lookup dict.
- **Save `ev_fractions` array to ablation `.npz`** when `--ev-thresholds` is
  used (parallel to the existing `ks` array). Would enable `y_mode='ev'` in the
  intervention tripanel without a separate `wu_subspace_analysis` run to
  recover the explained-variance mapping.
- **Save `explained variance vs. k-subspace rank` arrays**, currently plotted
  but not persisted. Recomputable but cheap to store.
- **Save raw residual stream and logit lens data cubes** for post-processing.
  Partially addressed: `--save-data` now persists ActivationRecords from the
  entropy, ablation, ck, and single-prompt workflows, so c_k-style analyses can
  be re-run without new forward passes.

### Corpus

- **Corpus expanded to 108 pairs / 216 prompts** (`base_vs_contrast_n216.json`),
  up from 25 pairs. Roughly halves confidence intervals — verified on
  gpt2-small ablation, where base KL standard error at k=200 went from ±0.64 to
  ±0.31 while the base-vs-contrast separation held.
  `corpus_gen.py --legacy` regenerates the original 25-pair file byte-for-byte
  on every original field, so pre-expansion results stay comparable.
- **Per-category power is still thin.** At 108 pairs across 5 categories x 4
  contrast_types, some cells hold only 3-5 pairs. Aggregate and
  per-contrast_type claims are well powered; per-(category x contrast_type)
  claims are not. Target 200+ pairs if the cross-tabulated breakdown becomes
  load-bearing.
- **Category balance is uneven** (pattern 25, syntactic 24, predictability 23,
  arithmetic 21, repetition 15). Acceptable, but worth evening out on the next
  expansion.

### Refactoring

Deferred clean-up items. The current code works; these are quality-of-life
improvements for when the project leaves pilot scale.

- ~~**Unify duplicated SVD utilities.**~~ **DONE.** `compute_wu_svd()` and
  `renyi_entropy()` now live in `src/math_utils.py` and are re-exported by
  `entropy_compute` and `ablation_compute` for backward compatibility.
  `math_utils` also owns the device policy (`svd_device`, `compute_device`).
- **Standardize plot file-path arguments.** `entropy_plots.py` takes
  `output_dir` plus optional `filename` and merges internally;
  `ablation_plots.py` takes a single pre-merged `save_path`. Pick one.
- **Separate exploratory diagnostic plots from curated presentation plots.**
  Low-level helpers (`_mean_and_ci`, `_diff_and_ci`, `_save`, color constants,
  `_fdr_bh`) are duplicated between the workflow plot modules and
  `post_process_plots.py`. Move shared helpers into a common plotting utility.
- **More robust data and plot naming conventions** if parameter sweeps
  proliferate. The `{corpus_tag}_{run_tag}` suffix pattern is a stop-gap; a
  single naming scheme defined in one place would be cleaner.

### Compute environment

- **Colab support is in place** (`colab/residual_stream_dynamics_colab.ipynb`,
  edited directly). `data/` and `figures/` symlink into Drive so results
  survive runtime recycling.
- **pythia-6.9b does not fit** in 16GB and is not runnable on the free tier.
  Existing 6.9b results in `data/` were produced locally and remain valid.
- **Watch system RAM, not just VRAM.** Colab free tier has ~12GB of host RAM,
  and `extract_activations()` returns numpy arrays on the host. For pythia-2.8b
  across 216 prompts x 3 hook types, RAM will bind before VRAM. Chunk the
  corpus if this bites.
