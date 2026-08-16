# residual-stream-dynamics

## Project purpose

This project investigates how information in a transformer's residual stream evolves
across layers, testing whether base prompts (those requiring structured reasoning) and
contrast prompts (those that do not) differ in their geometric and semantic entropy
profiles. The central finding is an anti-correlation between residual stream entropy
and logit lens entropy across layers. The ablation experiments convert this correlational
finding into an interventional claim: systematically removing the component of the
residual stream orthogonal to the W_U prediction subspace (r⊥) degrades token
predictions more for base prompts than for contrast prompts, suggesting that r⊥ carries
computationally meaningful content that differs systematically between prompt types.

---

## Pipeline architecture

```
src/math_utils.py             SHARED MATH — stateless mathematical utilities used by more than one
                              compute module. No model access, no file I/O, no plotting.
                              renyi_entropy(probs, alpha) — Rényi entropy in bits; Shannon at alpha=1.
                              compute_wu_svd(W_U) — SVD of W_U returning Vh; forces .cpu() for MPS
                              stability. Both re-exported by entropy_compute and ablation_compute.

src/extraction.py             EXTRACTION — runs forward passes; produces ActivationRecord dicts.
                              One forward pass per prompt; all hook types extracted simultaneously.
                              Exports BOS_TOKENS (set) and save/load_activation_records (batch).

src/entropy_compute.py        COMPUTATION — entropy and metric computation over ActivationRecords.
                              Produces EntropyRecords. No forward passes; no live model required
                              (except W_U / ln_final passed in for logit lens and subspace paths).
                              Key functions: wu_explained_variance(W_U, k_values) [2-arg].
                              Re-exports compute_wu_svd from math_utils, and CkRecord,
                              compute_wu_svd_full, compute_ck_spectrum, save_ck_records,
                              load_ck_records from ck_spectrum_compute.py for backward compatibility.

src/ck_spectrum_compute.py    COMPUTATION — c_k spectrum computation over ActivationRecords.
                              Produces CkRecords. Canonical home for CkRecord dataclass,
                              compute_wu_svd_full() [S, Vh], compute_ck_spectrum(),
                              save_ck_records / load_ck_records.

src/ablation_compute.py       COMPUTATION — ablation experiments over ActivationRecords.
                              Produces AblationRecords. Stage 1 (posthoc) needs no forward pass;
                              Stage 2 (intervention) requires a live model passed in.
                              Re-exports compute_wu_svd from math_utils.

src/mechanics_compute.py      COMPUTATION — residual stream trajectory mechanics over ActivationRecords.
                              Produces MechanicsRecords. Pure numpy; no model access; no forward pass.
                              Interprets the residual stream as a discrete particle trajectory:
                              speed (||ΔX_l||₂), acceleration magnitude (||ΔV_l||₂), and three
                              cosine similarity curves (state-state, update-state, update-update).
                              save_mechanics_records / load_mechanics_records in this module.

workflows/                    ORCHESTRATION — argument parsing, corpus iteration, save/load,
                              calls into compute modules, calls into plot modules. No computation
                              or plotting logic belongs here.

  entropy_analysis.py         — entropy corpus workflow; saves EntropyRecords + ActivationRecords
  ablation_analysis.py        — ablation corpus workflow; saves AblationRecords + ActivationRecords
  ck_analysis.py              — c_k spectrum workflow; saves CkRecords + ActivationRecords
  wu_subspace_analysis.py     — W_U subspace entropy workflow
  mechanics_analysis.py       — mechanics corpus workflow; saves MechanicsRecords to data/mechanics/
  single_prompt.py            — exploratory single-prompt workflow; saves ActivationRecords

plotting/entropy_plots.py     VISUALIZATION — workflow-layer figures produced by entropy_analysis.py.
plotting/ablation_plots.py    VISUALIZATION — workflow-layer figures produced by ablation_analysis.py.
plotting/ck_spectrum_plots.py VISUALIZATION — figures produced by ck_analysis.py. All _plot_*
                              functions for CkRecords live here; none belong in the workflow.
plotting/mechanics_plots.py   VISUALIZATION — figures produced by mechanics_analysis.py.
                              plot_mechanics_overview() and plot_mechanics_category().
plotting/post_process_plots.py VISUALIZATION — curated notebook figures. Accepts pre-filtered
                              profiles (list of np.ndarray); does not run extraction or computation.

dashboard/dashboard_loader.py DASHBOARD DATA — NPZ discovery, loading, caching, and index building
                              for the interactive dashboard. No computation; no plotting; no torch.
                              Loads mechanics/ subdirectory alongside entropy/, wu_subspace/,
                              ablation/, ck/. query_mechanics(model, role) returns all five
                              trimmed curve lists. available_mechanics_models() for discovery.
dashboard/dashboard_viz.py    DASHBOARD VISUALIZATION — matplotlib figure functions for the dashboard.
                              All functions take numpy arrays and return Figure. No file I/O.
                              Includes plot_mechanics_curves() for the Mechanics tab.
dashboard/dashboard.py        DASHBOARD APP — Gradio Blocks app. Five tabs: Entropy, WU Subspace,
                              Ablation, C_k Spectra, Mechanics. Callbacks call loader → viz →
                              return figure. Nothing else. --data-root must exist or the app
                              exits with a clear error. --port defaults to None (auto-select).
                              Entry point: python dashboard/dashboard.py --data-root data/

utils/npz_utils.py            DATA ACCESS — load and filter .npz files. No computation; no plotting.
utils/model_loader.py         MODEL LOADING — TransformerLens model loader and MODEL_CONFIGS registry.
corpus/corpus_gen.py          CORPUS — generates base/contrast prompt pairs as JSON.
                              108 pairs / 216 prompts across 5 categories x 4 contrast_types.
                              --legacy regenerates the original 25-pair corpus byte-for-byte
                              on every original field. --stats prints the design matrix.
                              validate_corpus() runs on every invocation and fails closed on
                              duplicate descriptions (which would make --legacy ambiguous).
```

---

## Key data structures

### ActivationRecord  (extraction.py)
Stores raw activations for ONE hook type across all layers and token positions.
- `activations`: `np.ndarray` shape `[n_layers, seq_len, d_model]`
- `hook_type`: short name e.g. `"resid_post"` (key into `HOOK_TYPES` registry)
- `hook_pattern`: full TransformerLens template e.g. `"blocks.{layer}.hook_resid_post"`
- `has_resid_mid`: model capability flag (True for GPT-2, Gemma-2, Llama; False for Pythia)
- `pair_id / role / category`: corpus metadata; `None` for single-prompt runs
- One record per hook type; multiple hook types from the same forward pass are
  returned as `dict[str, ActivationRecord]` by `extract_activations()`.
- Batch serialization: `save_activation_records(records, path)` / `load_activation_records(path)`
  in `extraction.py`. NaN-pads to `[n, max_layers, max_seq_len, d_model]`.

### EntropyRecord  (entropy_compute.py)
Stores a 2D entropy surface for one `(hook_type, norm_key, alpha)` combination.
- `surface`: `np.ndarray` shape `[n_layers, seq_len]`
- `norm_key`: `"energy"`, `"abs"`, `"softmax"`, or `"logit_lens"`
- `alpha`: Rényi order parameter (1.0 = Shannon)
- `str_tokens`: list of token strings; persisted in `.npz` (backward-compat: old files restore `[]`)
- One ActivationRecord → many EntropyRecords (one per norm_key × alpha)

### AblationRecord  (ablation_compute.py)
Stores ablation results for one `(prompt, k, ablation_type)` combination.
- Posthoc: metric arrays have shape `[n_layers]` (one value per layer)
- Intervention: metric arrays have shape `[1]` (final-layer effect only)
- `k`: subspace rank of the W_U SVD projection
- `ablation_type`: `"posthoc"` or `"intervention"`
- `intervention_layer`: `None` for posthoc; layer index for intervention
- `entropy_full` / `entropy_ablated`: per-layer entropy arrays, now serialized in `.npz`
  (backward-compat: old files without these fields restore `np.nan` arrays)

### MechanicsRecord  (mechanics_compute.py)
Stores five scalar mechanical curves for one prompt at the final token position.
- `speed`: `np.ndarray` shape `[n_layers-1]` — ||ΔX_l||₂
- `acceleration_magnitude`: `np.ndarray` shape `[n_layers-2]` — ||ΔV_l||₂
- `cosine_sim_state`: `np.ndarray` shape `[n_layers-1]` — cos(X_l, X_{l+1})
- `cosine_sim_update_state`: `np.ndarray` shape `[n_layers-1]` — cos(ΔX_l, X_l)
- `cosine_sim_update_update`: `np.ndarray` shape `[n_layers-2]` — cos(ΔX_l, ΔX_{l+1})
- Standard metadata: `prompt`, `model_name`, `pair_id`, `role`, `category`
- All arrays are pure numpy float64. No model access required — computed entirely
  from `record.activations[:, token_pos, :]`.
- Batch serialization: `save_mechanics_records` / `load_mechanics_records` in
  `mechanics_compute.py`. Saves to `data/mechanics/` for dashboard discovery.

### CkRecord  (ck_spectrum_compute.py)
Stores the c_k spectrum for one prompt: `c_k = σ_k · (r · v_k)`, an exact decomposition
of logits via the SVD of W_U. Produced by `compute_ck_spectrum()`.
- `ck_spectrum`: `np.ndarray` shape `[n_layers, seq_len, d_model]`, or
  `[n_layers, 1, d_model]` when computed with `last_token_only=True`
- `singular_values`: `np.ndarray` shape `[d_model]`, descending σ_k
- `last_token_only`: bool — True when only the final token position was
  retained (`--last-token-only`). Eight of the nine functions in
  `ck_spectrum_plots.py` slice `[:, -1, :]` and are unaffected;
  `plot_heatmap_alltokens()` and `CkRecord.layer_spectrum()` raise on such
  records rather than silently averaging over one token. Serialized in the
  `.npz`; old files without the key load as `False`.
- `str_tokens`: list of token strings; persisted in `.npz`
- Standard metadata: `prompt`, `model_name`, `hook_type`, `n_layers`, `seq_len`, `d_model`,
  `pair_id`, `role`, `category`
- Batch serialization: `save_ck_records` / `load_ck_records` in `ck_spectrum_compute.py`
- `entropy_compute.py` re-exports `CkRecord` and all its functions for backward compatibility;
  new code should import directly from `ck_spectrum_compute`.

---

## Naming conventions and module organization

- **Hook types** always use short names (`"resid_post"`) as dict keys and as the
  `hook_type` field on records. Full TransformerLens patterns (`"blocks.{layer}.hook_resid_post"`)
  live in `HOOK_TYPES` in `src/extraction.py` — nowhere else.
- **Norm keys** always use `"energy"`, `"abs"`, `"softmax"`, `"logit_lens"` as strings.
  The single source of truth is `NORM_METHODS` in `src/entropy_compute.py`.
- **Dataclass convention**: `ActivationRecord`, `EntropyRecord`, `AblationRecord`, `CkRecord`
  are the canonical pipeline data structures. Each record type lives in its own compute module
  (`src/`) alongside its compute functions and serialization pair. Add new analysis types by
  creating a new `src/*_compute.py` / `plotting/*_plots.py` pair — do not add new Record
  classes to existing modules.
- **Module pair pattern**: every analysis type has a matched pair — `src/*_compute.py`
  (dataclass, compute functions, save/load) and `plotting/*_plots.py` (all visualization,
  no file I/O). The workflow script in `workflows/` is the only caller of both.
- **Serialization**: every Record type has a `save_*/load_*` pair in its compute module.
  Use `.npz` for all persistence. NaN-padding is used for variable-length arrays.
- **Corpus metadata** (`pair_id`, `role`, `category`) flows through all Record types
  unchanged from ActivationRecord. It is `None` for single-prompt exploratory runs.
- **`contrast_type` is corpus-only.** The corpus JSON carries a fourth field
  (`abstract` / `concrete` / `in_domain` / `swap`) controlling the abstract-noun
  confound, but `extraction.py` reads only `pair_id` / `role` / `category`, so it
  does not reach the Record types. Stratified analysis joins back to the corpus
  file on `prompt` or `pair_id`. Threading it through is a listed FutureWork item —
  do not add it to one Record type in isolation.

---

## What belongs in notebooks vs. modules

**Strict rule**: Notebooks contain narrative, figure display, and parameter choices.
They do not contain computation, filtering logic, or matplotlib calls beyond
`plt.show()` / saving.

| Belongs in notebooks | Belongs in modules |
|---|---|
| `load_entropy_npz(...)` calls | Loading and filtering logic (`npz_utils.py`) |
| `get_final_token_profiles(...)` calls | Statistical helpers (`_mean_and_ci`, etc.) |
| `plot_*(...)` calls | Plotting functions (`plotting/post_process_plots.py`) |
| Parameter choices (k, alpha, model) | Compute loops and entropy math |
| `fig.show()` / `plt.savefig()` | `save_path` handled inside plot functions |

**No inline matplotlib in notebooks** — all figure construction belongs in
`plotting/post_process_plots.py`. Notebooks pass pre-filtered profile lists to plot functions
and display the returned `fig`.

**No logic in workflow scripts** — `workflows/` scripts parse arguments, call
compute functions, and call plot functions. Statistical or computational logic that
is not argument-parsing or I/O belongs in a compute module.

---

## Known architectural constraints

- **k range**: The W_U subspace rank k must satisfy `1 ≤ k ≤ d_model`. `k = d_model`
  is explicitly allowed (it is the no-op ablation sentinel — `Q_k @ Q_k.T` becomes the
  identity). The entropy subspace functions require `k < d_model` to produce a meaningful
  orthogonal complement.
- **Post-LN convention for ablation**: `compute_posthoc_ablation()` projects the pre-LN
  residual stream (`r`), then applies `ln_final` to the projected vector. The alternative
  (project post-LN, then multiply by W_U) is noted in a comment but currently inactive.
  Do not change this without updating the paper and rerunning the full ablation sweep.
- **Final token convention**: All ablation metrics operate on the final token position
  (`activations[layer, -1, :]`). This is a deliberate design choice for next-token
  prediction analysis. Entropy surfaces are computed over all token positions.
- **BOS token**: BOS detection is centralized in `BOS_TOKENS = {"<|endoftext|>", "<s>", "<bos>"}`
  exported from `src/extraction.py`. `ActivationRecord.token_slice()` and `plotting/entropy_plots._bos_slice()`
  both import this set. GPT-2/Pythia, Llama, and Gemma are all covered. Do not hardcode BOS
  strings elsewhere — add new model families to `BOS_TOKENS` in `src/extraction.py` only.
- **MPS stability**: `torch.linalg.svd` on large matrices is unstable on MPS. The
  canonical `compute_wu_svd()` pins to `.cpu()` before decomposition on MPS.
  Do not remove this. It is now expressed through `math_utils.svd_device()`
  rather than an unconditional `.cpu()` call, so CUDA (where svd is stable and
  much faster) keeps the tensor on device. On CPU and MPS the behavior is
  unchanged from the original unconditional version.
- **Device policy lives in `math_utils`**: `svd_device(t)` and `compute_device(t)`
  are the single source of truth for where linear algebra runs. Both are
  identity on CPU/CUDA and `.cpu()` on MPS. Use them instead of hardcoding
  `.cpu()` in a compute module.
- **Not every loop belongs on the GPU.** `compute_wu_subspace_entropy()` stays
  on CPU deliberately: its inner loop is (layer × token × k × alpha) and each
  iteration ends in `renyi_entropy(...).item()`, a device sync that costs more
  than the small `[d_model, k]` matmul it guards. The logit-lens and c_k paths
  are the opposite case — one large matmul per sync — and do run on device.
- **Single forward pass**: `extract_activations()` runs exactly one forward pass
  regardless of how many hook types are requested. All hook types share the same
  forward pass. Never call `run_with_cache` inside a compute module.

---

## Hardware and framework context

- **TransformerLens** (`HookedTransformer`): all forward passes, hook registration,
  and model weight access use TransformerLens. `model.W_U`, `model.ln_final`,
  `model.cfg`, and `model.run_with_hooks()` are the primary interfaces.
- **Device**: workflows default `--device None` (auto-detect: CUDA if available,
  else CPU; MPS is downgraded to CPU in `model_loader.py`). MPS is usable for
  forward passes but not for `torch.linalg.svd` on large matrices, so SVD pins
  to CPU there via `math_utils.svd_device()`.
- **Dtype**: `--dtype` defaults to float32. On CUDA, models flagged
  `large_on_16gb` in `MODEL_CONFIGS` (pythia-2.8b, pythia-6.9b, gpt2-xl,
  llama-3.2-3b) auto-select float16 so they fit a 16GB card. fp16 is forced
  back to float32 on CPU and MPS. `W_U` is upcast with `.float()` before SVD
  regardless, so fp16 weights never reach the decomposition.
- **Google Colab**: `colab/residual_stream_dynamics_colab.ipynb` runs the
  corpus workflows on a free-tier T4. Regenerate it with
  `python colab/build_notebook.py` — do not hand-edit the `.ipynb`.
  pythia-6.9b does not fit in 16GB and is not runnable on the free tier.
- **Model zoo**: GPT-2 (small/medium/large/XL) and Pythia (160m, 1b, 2.8b, 6.9b)
  are the primary test models. `utils/model_loader.py` contains `MODEL_CONFIGS` for each.
  Gemma-2 and Llama support is partial (`has_resid_mid` detection works; BOS token
  handling in `token_slice()` requires model-specific attention).
- **Data format**: all persistent results are `.npz` (NumPy compressed). Variable-length
  arrays (different seq_len per prompt, different n_layers across models) are NaN-padded
  to a common shape on save and trimmed back on load.

---

## What NOT to do

- **No inline matplotlib in notebooks.** Put all figure construction in
  `plotting/post_process_plots.py`. Notebooks call plot functions and display `fig`.
- **No logic in workflow scripts** (`workflows/`) that belongs in compute modules.
  Argument parsing, path resolution, and I/O are the only things that belong in workflows.
- **No premature abstraction.** Do not create a base `Record` class or a unified
  `compute_entropy_or_ablation` dispatcher. The four Record types are intentionally
  separate; their compute functions are intentionally not polymorphic.
- **Do not add new hook types to HOOK_TYPES** without verifying the pattern works
  across all models in MODEL_CONFIGS. MLP-internal hooks (`mlp_pre`, `mlp_post`) have
  d_model = 4× the residual stream width — this propagates through d_model fields on
  the record and must not be passed to logit lens or ablation functions.
- **Do not run forward passes inside compute modules.** Only `src/extraction.py` and the
  intervention path of `src/ablation_compute.py` (which requires a live model explicitly
  passed in) are allowed to call `model.run_with_cache` or `model.run_with_hooks`.
- **Do not save figures using `plt.savefig` directly in compute or workflow scripts.**
  All saving goes through the `_save(fig, save_path)` helper in plot modules, which
  creates parent directories and uses consistent DPI (150).
- **Do not use `--intervention-all-layers`** — this flag has been removed. Use
  `--intervention-stride 1` to achieve the same effect.
- **Do not call `wu_explained_variance` with 3 arguments.** The signature is
  `wu_explained_variance(W_U, k_values)` — the old `Vh` parameter was removed because
  it was silently ignored. The function recomputes SVD internally.
- **Do not use relative `sys.path` inserts** (e.g. `sys.path.insert(0, 'utils')`).
  Always derive paths from `Path(__file__).resolve()` so scripts work from any working
  directory.
- **No torch, no TransformerLens, no model loading in the dashboard modules.**
  `dashboard/dashboard_loader.py`, `dashboard/dashboard_viz.py`, and `dashboard/dashboard.py`
  are pure numpy + matplotlib + Gradio. Any computation that requires a live model belongs
  in a compute module, not here.
- **No inline matplotlib in `dashboard/dashboard.py`.** All figure construction belongs in
  `dashboard/dashboard_viz.py`. Dashboard callbacks do exactly: call loader → call viz → return figure.
- **No plot-triggering `.change()` callbacks in `dashboard/dashboard.py`.** Plots fire only from
  `.click()` on an explicit "Update Plot" button. Dropdown `.change()` callbacks are
  reserved for chaining dependent selectors (norm key, alpha, k) only.

---

## Next steps

### Clean up the project root directory — COMPLETED (2026-05-02)
Root reorganization is done. All Python files are now in named subdirectories:
- `src/` — extraction + compute modules
- `plotting/` — all visualization modules (including post_process_plots.py)
- `dashboard/` — dashboard app, loader, and viz
- `utils/` — model_loader.py (renamed from setup.py) + npz_utils.py
- `corpus/` — corpus_gen.py + data
- `workflows/` — CLI scripts (updated sys.path + imports)
- `notebooks/` — notebook only
- `sandbox/` — exploratory scripts (imports updated)
- `tests/` — deleted (test_wu_subspace.py referenced a module that no longer exists)

---

## Session changelog (2026-05-02)

### Repo reorganization — all Python moved out of project root

All 15 Python files moved from the project root into named subdirectories. Zero Python
files remain at the root. Import strategy uses `sys.path.insert` (the project's existing
convention) so no module names changed — only file locations.

- `src/` — `extraction.py`, `entropy_compute.py`, `ck_spectrum_compute.py`,
  `ablation_compute.py`, `mechanics_compute.py`
- `plotting/` — `entropy_plots.py`, `ck_spectrum_plots.py`, `ablation_plots.py`,
  `mechanics_plots.py`, `post_process_plots.py`
- `dashboard/` — `dashboard.py`, `dashboard_loader.py`, `dashboard_viz.py`
- `utils/` — `model_loader.py` (renamed from `setup.py`), `npz_utils.py`, `npz_quicklook.py`
- `corpus/` — `corpus_gen.py` + data files
- `tests/` — deleted (`test_wu_subspace.py` imported a module that no longer exists)

All 6 workflow scripts updated: `sys.path` now inserts `src/`, `utils/`, and `plotting/`
explicitly. All sandbox scripts updated: `from setup import` → `from model_loader import`.
Notebook sys.path cell updated to point at `../plotting` and `../utils`.

### New: `src/math_utils.py` — shared mathematical utilities

Created `src/math_utils.py` as a home for stateless mathematical functions used by more
than one compute module. No model access, no file I/O, no plotting.

**`renyi_entropy(probs, alpha)`** — moved from `src/entropy_compute.py`. Rényi entropy
in bits; Shannon limit at `alpha=1.0`. Previously duplicated implicitly via import.

**`compute_wu_svd(W_U)`** — consolidated from two divergent definitions:
- `src/entropy_compute.py` had a version that did NOT force `.cpu()` (MPS-unsafe)
- `src/ablation_compute.py` had the canonical version with `.cpu()`
The canonical (`.cpu()`) version now lives only in `src/math_utils.py`. Both compute
modules re-export it via `from math_utils import compute_wu_svd` for backward compatibility.

### Other quality-of-life improvements

- `requirements.txt` created at project root (torch, transformer_lens, sae_lens, numpy,
  matplotlib, gradio, huggingface_hub with minimum version pins).
- `--help` text in all 6 workflows updated: `"must be in setup.py MODEL_CONFIGS"` →
  `"must be in utils/model_loader.py MODEL_CONFIGS"`.
- Error handling added to all 6 workflow `main()` functions: bad corpus path, bad JSON,
  unknown model name, and bad `--load-data` path all now print a clean message and
  `return 1` instead of raising a raw traceback.
- Duplicate `from pathlib import Path` imports removed from `plotting/entropy_plots.py`,
  `plotting/mechanics_plots.py`, and `plotting/post_process_plots.py`.

---

## Session changelog (2026-05-01)

### New: mechanics analysis group

**`mechanics_compute.py`** (new)
`MechanicsRecord` dataclass and five pure-numpy compute functions that interpret the
residual stream as a discrete particle trajectory: `_compute_velocity`, `_compute_speed`,
`_compute_acceleration_magnitude`, `_compute_cosine_sim_state`,
`_compute_cosine_sim_update_state`, `_compute_cosine_sim_update_update`. Public entry
point: `compute_mechanics(record, token_pos=-1) -> MechanicsRecord`. No torch, no model
access. `save_mechanics_records` / `load_mechanics_records` follow the NaN-padding
conventions established in `entropy_compute.py`.

**`mechanics_plots.py`** (new)
Two corpus-level plot functions: `plot_mechanics_overview` (five-panel mean ± 1σ across
all pairs, base vs. contrast) and `plot_mechanics_category` (same five panels for one
category, individual pair curves as faint lines behind the bold mean). Style mirrors
`entropy_plots.plot_overall_mean` and `plot_category`. `_save()` helper creates parent
directories and uses dpi=150 consistently.

**`workflows/mechanics_analysis.py`** (new)
CLI workflow following `entropy_analysis.py` conventions exactly. Args: `--corpus`
(default: `corpus/base_vs_contrast_n50.json`), `--model`, `--category`, `--output-dir-plots`,
`--output-dir-data`, `--save-data`, `--no-plots`, `--run-tag`, `--device`. Saves
MechanicsRecords to `data/mechanics/` (the subdirectory the dashboard loader expects).
Does not save ActivationRecords — mechanics are already downstream scalars.

### Dashboard: Mechanics tab added

**`dashboard_loader.py`**
Added `_MECHANICS_REQUIRED_KEYS`, `_load_mechanics_file()`, `"mechanics"` entry in
`_SUBDIRS` (discovers `data/mechanics/`), `query_mechanics(model, role)` returning all
five trimmed curve lists, and `available_mechanics_models()`.

**`dashboard_viz.py`**
Added `plot_mechanics_curves(base, contrast, model_name)` — five-panel figure (speed,
acceleration magnitude, three cosine similarities), mean ± SEM per role. Cosine panels
include y-limits ±1.05 and a zero reference line.

**`dashboard.py`**
Added Tab 5 "Mechanics" with model dropdown and "Update Plot" button. Both roles are
always plotted together (no role selector) since the comparison is the point. Includes
"no data" fallback message with the generation command. Added `--data-root` existence
check with a clear error and `sys.exit(1)` before loading. Changed `--port` default
from 7860 to `None` so Gradio auto-selects a free port.

### Quality-of-life improvements

**Default `--corpus` in all workflows**
All five `workflows/*_analysis.py` scripts now default `--corpus` to
`corpus/base_vs_contrast_n50.json` (resolved via `_PROJECT_ROOT`) instead of
requiring it. `required=True` removed.

**Corpus filename printed after loading**
All five workflow scripts now print `  Corpus file:  <filename>` on the line after
`Loaded corpus: N prompts (M pairs)` to make it easy to confirm which corpus is active.

---

## Session changelog (2026-04-30)

### Refactor: CkRecord extracted into its own module pair

**`ck_spectrum_compute.py`** (new)
`CkRecord`, `compute_wu_svd_full()`, `compute_ck_spectrum()`, `save_ck_records()`, and
`load_ck_records()` were moved out of `entropy_compute.py` into a dedicated module,
following the established `*_compute.py` / `*_plots.py` pattern for each record type.
`entropy_compute.py` re-exports all five names for backward compatibility — existing
`from entropy_compute import CkRecord` calls continue to work unchanged.

**`ck_spectrum_plots.py`** (new)
All `_plot_*` functions for CkRecords were moved out of `workflows/ck_analysis.py` into
a dedicated plot module. Public names drop the leading underscore:
`plot_single_prompt_diagnostic`, `plot_heatmap_lasttoken`, `plot_heatmap_alltokens`,
`plot_com_vs_layer`, `plot_cumpower_vs_k`, `plot_delta_ck_heatmap`,
`plot_heatmap_variance_lasttoken`, `plot_variance_ratio_vs_k`, `plot_com_variance_vs_layer`.
Internal helpers (`_build_heatmap_grids`, `_render_heatmap_figure`, `_compute_*`) remain
private. The `_save()` helper in `ck_spectrum_plots.py` creates parent directories before
saving, consistent with other plot modules.

**`workflows/ck_analysis.py`** (reduced)
Now imports from `ck_spectrum_compute` and `ck_spectrum_plots` and contains only
`_safe_model_name`, `_run_ck_corpus` (corpus iteration helper), and `main`.

### New: interactive Gradio dashboard

**`dashboard_loader.py`** (new)
`DashboardLoader` class discovers all `.npz` files by subdirectory on construction,
concatenates multiple files of the same type along axis 0 (NaN-padding surfaces to the
global max shape to handle different `max_layers` / `max_seq_len` across model families),
and builds an in-memory index. All queries are vectorized boolean masking — no Python
loops in the hot path. Key methods: `query_entropy`, `query_wu_subspace`,
`query_ablation_posthoc`, `query_ablation_intervention`, `query_ck`.

**`dashboard_viz.py`** (new)
Standalone matplotlib figure functions: `plot_entropy_curves`, `plot_wu_subspace_curves`,
`plot_ablation_posthoc`, `plot_ablation_heatmap`, `plot_ck_spectrum`. All functions take
numpy arrays and return `matplotlib.Figure`. Each calls `plt.close('all')` as its first
executable line. All handle the zero-records case with an informative empty figure.

**`dashboard.py`** (new)
Gradio `Blocks` app with four tabs (Entropy, WU Subspace, Ablation, C_k Spectra).
Each tab has an "Update Plot" button; plots fire only from `.click()`, never from
`.change()`. Model dropdowns chain-update dependent selectors via `.change()`.
Run with `python dashboard/dashboard.py --data-root data/`.

---

## Session changelog (2026-04-28)

### Bug fixes

**Bug #1 — `corpus_tag` NameError in `ablation_analysis.py` fast path**
(`workflows/ablation_analysis.py`)
The `--load-data` branch referenced `corpus_tag` before it was defined. Fixed by adding
`corpus_tag = Path(args.load_data).stem` at the top of that branch.

**Bug #2 — `entropy_full` / `entropy_ablated` not serialized in AblationRecords**
(`ablation_compute.py`)
`save_ablation_records` did not persist the per-layer entropy arrays. Added
`ent_full_padded` / `ent_abl_padded` NaN-padded arrays to the `.npz`. Load is
backward-compatible: old files without these fields restore `np.nan` arrays of the
correct length.

**Bug #3 — `wu_explained_variance` dead `Vh` parameter**
(`ablation_compute.py`, `workflows/ablation_analysis.py`)
The function accepted `Vh` as a third argument but never used it (recomputed SVD
internally). Removed the parameter; updated all call sites to 2-arg form
`wu_explained_variance(W_U, k_values)`.

**Bug #4 — BOS detection hardcoded to `'<|endoftext|>'`**
(`extraction.py`, `entropy_plots.py`)
`token_slice()` used a string literal that breaks for Llama (`<s>`) and Gemma (`<bos>`).
Replaced with `BOS_TOKENS = {"<|endoftext|>", "<s>", "<bos>"}` exported from
`extraction.py` as the single source of truth. Both `token_slice()` and
`entropy_plots._bos_slice()` now import and use this set.

**Bug #5 — `str_tokens` silently dropped on save/load**
(`entropy_compute.py`)
`save_entropy_records` did not persist `str_tokens`; `load_entropy_records` always
restored `[]`. Fixed: `str_tokens` is now serialized as an object array in the `.npz`
for both `EntropyRecord` and `CkRecord`. Load is backward-compatible: old files without
the key restore `[]`.

**Bug #6 — Fragile relative `sys.path` in `post_process_plots.py`**
(`post_process_plots.py`)
`sys.path.insert(0, 'utils')` broke when the script was imported from any directory
other than the project root. Replaced with
`sys.path.insert(0, str(Path(__file__).resolve().parent / "utils"))`.

### New features

**Batch ActivationRecord serialization**
(`extraction.py`, `workflows/entropy_analysis.py`, `workflows/ablation_analysis.py`,
`workflows/single_prompt.py`)
Added `save_activation_records(records, path)` / `load_activation_records(path)` (plural,
batch) to `extraction.py`. NaN-pads activations to `[n, max_layers, max_seq_len, d_model]`.
Removed the old singular `save_activation_record` / `load_activation_record` which were
defined but never called. All three workflow scripts now save ActivationRecords under
`--save-data` alongside their primary record type.

**c_k spectrum analysis**
(`entropy_compute.py`, `workflows/ck_analysis.py`)
Added to `entropy_compute.py`:
- `compute_wu_svd_full(W_U)` → `(S, Vh)`: full SVD returning both singular values and
  right singular vectors. Distinct from `compute_wu_svd()` which returns only `Vh`.
- `CkRecord` dataclass: stores `ck_spectrum [n_layers, seq_len, d_model]` and
  `singular_values [d_model]` plus standard metadata.
- `compute_ck_spectrum(record, S, Vh)` → `CkRecord`: vectorized computation of
  `c_k = σ_k · (v_k · r)` per layer, exact logit decomposition.
- `save_ck_records` / `load_ck_records`: NaN-padded batch serialization.

Created `workflows/ck_analysis.py`: full CLI workflow mirroring `ablation_analysis.py`
conventions. Produces two figure types:
1. 2×2 single-prompt diagnostic (raw projections, sorted projections, c_k spectrum,
   sorted |c_k|) for a selected `(layer, token)`.
2. Layer-evolution heatmap: mean |c_k| over tokens and prompts, stratified by role
   (base, contrast, base−contrast difference). Primary diagnostic for the
   build-then-cash-out hypothesis.
Supports `--save-data`, `--load-data` fast path, `--category`, `--run-tag`.

**Extended `ck_analysis.py` with summary statistics and improved heatmaps**
(`workflows/ck_analysis.py`)
Replaced the single layer-evolution heatmap with five new figures. All new computations
are self-contained in `ck_analysis.py`; `entropy_compute.py` is unchanged.

1. **Last-token heatmap** (`_heatmap_lasttoken`): restricts to `token = seq_len - 1`
   before averaging over prompts. Drops layer 0. Uses `LogNorm` colormap for Base and
   Contrast panels; symmetric linear scale for the Base−Contrast difference panel.
2. **All-tokens heatmap** (`_heatmap_alltokens`): same as above but averages over all
   token positions. Secondary diagnostic.
3. **Spectral CoM vs. layer** (`_com_vs_layer`): `CoM = Σ_k k·c_k² / Σ_k c_k²` at the
   last token, plotted as mean ± std across prompts vs. layer, base and contrast lines.
4. **Cumulative power fraction** (`_cumpower_vs_k`): `F(K) = Σ_{k<K} c_k² / Σ_k c_k²`
   at the last token, using squared c_k (exact power decomposition). One subplot per
   layer in `--summary-layers` (default `[1, 3, 6, 9, 11]`).
5. **Layer-to-layer |Δc_k| heatmap** (`_delta_ck_heatmap`): `|c_k(l) − c_k(l−1)|` at
   the last token. With `--skip-layer0` (default), the 0→1 transition is excluded; y-tick
   labels read `"1→2"`, `"2→3"`, etc.

New CLI flags: `--summary-layers` (list of ints), `--skip-layer0` / `--no-skip-layer0`.

**Prompt-variance diagnostics added to `ck_analysis.py`**
(`workflows/ck_analysis.py`)
Three new figures that measure inter-prompt scatter in the c_k spectrum rather than the
mean. All computed at the last token only; variance is taken across the prompt dimension
after stacking per-prompt `|c_k|` arrays. No new CLI flags — reuses `--summary-layers`
and `--skip-layer0`.

1. **Variance heatmap** (`_heatmap_variance_lasttoken`): Var_base and Var_contrast panels
   share an identical `LogNorm` for direct comparison. Third panel plots
   `log10(Var_base / (Var_contrast + 1e-8))`, clipped to `[−2, 2]`, on a diverging
   `RdBu_r` colormap. The `+1e-8` additive guard protects the denominator in directions
   where contrast prompts have near-zero variance; `var_base` is left unmodified.
2. **Variance ratio vs. k** (`_variance_ratio_vs_k`): `log10` ratio as a line plot, one
   line per `--summary-layers` layer on shared axes, with a `y=0` dashed reference.
   Unclipped — full dynamic range visible in the line plot.
3. **CoM of prompt-variance vs. layer** (`_com_variance_vs_layer`):
   `CoM_var = Σ_k k·Var(|c_k|) / Σ_k Var(|c_k|)`, computed from the already-aggregated
   variance grid (no per-prompt scatter to shade — lines only, no fill).

**Workflow output paths anchored to project root**
(`workflows/single_prompt.py`, `workflows/entropy_analysis.py`,
`workflows/ablation_analysis.py`, `workflows/ck_analysis.py`,
`workflows/wu_subspace_analysis.py`)
All five workflow scripts now compute `_PROJECT_ROOT = Path(__file__).resolve().parent.parent`
at module load time and use it as the base for default `--output-dir-plots` and
`--output-dir-data` paths. Figures and data land in `/project_root/figures/` and
`/project_root/data/` regardless of where the script is invoked from.

**Removed `--intervention-all-layers` flag**
(`workflows/ablation_analysis.py`)
Replaced throughout with `--intervention-stride 1` to achieve equivalent behavior.
The flag was removed from argparse, docstrings, and all example invocations.

**Legend fix in `plot_entropy_vs_layer`**
(`ablation_plots.py`)
Lines were plotted without labels, so the legend was empty. Added
`label=f"{label} (full)"` and `label=f"{label} (ablated)"` to the two `ax.plot` calls.
