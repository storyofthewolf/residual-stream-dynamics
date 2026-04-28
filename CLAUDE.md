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
extraction.py          EXTRACTION — runs forward passes; produces ActivationRecord dicts.
                       One forward pass per prompt; all hook types extracted simultaneously.

entropy_compute.py     COMPUTATION — entropy and metric computation over ActivationRecords.
                       Produces EntropyRecords. No forward passes; no live model required
                       (except W_U / ln_final passed in for logit lens and subspace paths).

ablation_compute.py    COMPUTATION — SVD utilities and ablation experiments over
                       ActivationRecords. Produces AblationRecords. Stage 1 (posthoc) needs
                       no forward pass; Stage 2 (intervention) requires a live model passed in.

workflows/             ORCHESTRATION — argument parsing, corpus iteration, save/load,
                       calls into compute modules, calls into plot modules. No computation
                       or plotting logic belongs here.

entropy_plots.py       VISUALIZATION — workflow-layer figures produced by entropy_analysis.py.
ablation_plots.py      VISUALIZATION — workflow-layer figures produced by ablation_analysis.py.
post_process_plots.py  VISUALIZATION — curated notebook figures. Accepts pre-filtered profiles
                       (list of np.ndarray); does not run extraction or computation.

utils/npz_utils.py     DATA ACCESS — load and filter .npz files. No computation; no plotting.
setup.py               MODEL LOADING — TransformerLens model loader and MODEL_CONFIGS registry.
corpus_gen.py          CORPUS — generates base/contrast prompt pairs as JSON.
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

### EntropyRecord  (entropy_compute.py)
Stores a 2D entropy surface for one `(hook_type, norm_key, alpha)` combination.
- `surface`: `np.ndarray` shape `[n_layers, seq_len]`
- `norm_key`: `"energy"`, `"abs"`, `"softmax"`, or `"logit_lens"`
- `alpha`: Rényi order parameter (1.0 = Shannon)
- One ActivationRecord → many EntropyRecords (one per norm_key × alpha)

### AblationRecord  (ablation_compute.py)
Stores ablation results for one `(prompt, k, ablation_type)` combination.
- Posthoc: metric arrays have shape `[n_layers]` (one value per layer)
- Intervention: metric arrays have shape `[1]` (final-layer effect only)
- `k`: subspace rank of the W_U SVD projection
- `ablation_type`: `"posthoc"` or `"intervention"`
- `intervention_layer`: `None` for posthoc; layer index for intervention

---

## Naming conventions and module organization

- **Hook types** always use short names (`"resid_post"`) as dict keys and as the
  `hook_type` field on records. Full TransformerLens patterns (`"blocks.{layer}.hook_resid_post"`)
  live in `HOOK_TYPES` in `extraction.py` — nowhere else.
- **Norm keys** always use `"energy"`, `"abs"`, `"softmax"`, `"logit_lens"` as strings.
  The single source of truth is `NORM_METHODS` in `entropy_compute.py`.
- **Dataclass convention**: `ActivationRecord`, `EntropyRecord`, `AblationRecord` are the
  canonical pipeline data structures. Add new analysis types by adding new `*Record` classes
  and compute functions to the appropriate compute module.
- **Serialization**: every Record type has a `save_*/load_*` pair in its compute module.
  Use `.npz` for all persistence. NaN-padding is used for variable-length arrays.
- **Corpus metadata** (`pair_id`, `role`, `category`) flows through all Record types
  unchanged from ActivationRecord. It is `None` for single-prompt exploratory runs.

---

## What belongs in notebooks vs. modules

**Strict rule**: Notebooks contain narrative, figure display, and parameter choices.
They do not contain computation, filtering logic, or matplotlib calls beyond
`plt.show()` / saving.

| Belongs in notebooks | Belongs in modules |
|---|---|
| `load_entropy_npz(...)` calls | Loading and filtering logic (`npz_utils.py`) |
| `get_final_token_profiles(...)` calls | Statistical helpers (`_mean_and_ci`, etc.) |
| `plot_*(...)` calls | Plotting functions (`post_process_plots.py`) |
| Parameter choices (k, alpha, model) | Compute loops and entropy math |
| `fig.show()` / `plt.savefig()` | `save_path` handled inside plot functions |

**No inline matplotlib in notebooks** — all figure construction belongs in
`post_process_plots.py`. Notebooks pass pre-filtered profile lists to plot functions
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
- **BOS token**: `ActivationRecord.token_slice()` checks for `'<|endoftext|>'` to
  identify BOS. This is correct for GPT-2 and Pythia. For models with different BOS
  tokens (Llama: `<s>`, Gemma: `<bos>`), pass `skip_bos=False` or extend the check.
- **MPS stability**: `torch.linalg.svd` on large matrices is unstable on MPS. The
  canonical `compute_wu_svd()` forces `.cpu()` before decomposition. Do not remove this.
- **Single forward pass**: `extract_activations()` runs exactly one forward pass
  regardless of how many hook types are requested. All hook types share the same
  forward pass. Never call `run_with_cache` inside a compute module.

---

## Hardware and framework context

- **TransformerLens** (`HookedTransformer`): all forward passes, hook registration,
  and model weight access use TransformerLens. `model.W_U`, `model.ln_final`,
  `model.cfg`, and `model.run_with_hooks()` are the primary interfaces.
- **Device**: default `cpu`; MPS is usable for forward passes but not for
  `torch.linalg.svd` on large matrices. All SVD computations force `.cpu()`.
- **Model zoo**: GPT-2 (small/medium/large/XL) and Pythia (160m, 1b, 2.8b, 6.9b)
  are the primary test models. `setup.py` contains `MODEL_CONFIGS` for each.
  Gemma-2 and Llama support is partial (`has_resid_mid` detection works; BOS token
  handling in `token_slice()` requires model-specific attention).
- **Data format**: all persistent results are `.npz` (NumPy compressed). Variable-length
  arrays (different seq_len per prompt, different n_layers across models) are NaN-padded
  to a common shape on save and trimmed back on load.

---

## What NOT to do

- **No inline matplotlib in notebooks.** Put all figure construction in
  `post_process_plots.py`. Notebooks call plot functions and display `fig`.
- **No logic in workflow scripts** (`workflows/`) that belongs in compute modules.
  Argument parsing, path resolution, and I/O are the only things that belong in workflows.
- **No premature abstraction.** Do not create a base `Record` class or a unified
  `compute_entropy_or_ablation` dispatcher. The three Record types are intentionally
  separate; their compute functions are intentionally not polymorphic.
- **Do not add new hook types to HOOK_TYPES** without verifying the pattern works
  across all models in MODEL_CONFIGS. MLP-internal hooks (`mlp_pre`, `mlp_post`) have
  d_model = 4× the residual stream width — this propagates through d_model fields on
  the record and must not be passed to logit lens or ablation functions.
- **Do not run forward passes inside compute modules.** Only `extraction.py` and the
  intervention path of `ablation_compute.py` (which requires a live model explicitly
  passed in) are allowed to call `model.run_with_cache` or `model.run_with_hooks`.
- **Do not save figures using `plt.savefig` directly in compute or workflow scripts.**
  All saving goes through the `_save(fig, save_path)` helper in plot modules, which
  creates parent directories and uses consistent DPI (150).
