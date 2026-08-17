# Session Log

Decisions, rejected alternatives, and open issues that the commit history does
not capture on its own. Newest entry at the bottom.

---

## 2026-08-16

**Commits:** `2816344`, `f54c06d` (branch `colab-support`, not pushed)

**Decisions**

- **Made the `.cpu()` pinning conditional rather than removing it.** CLAUDE.md
  says "Do not remove this" about the MPS SVD guard. The guard was correct; what
  was wrong was applying it unconditionally, which stranded the hot paths on the
  host even with a GPU present. `math_utils.svd_device()` / `compute_device()`
  are now the single source of truth: identity on CPU/CUDA, `.cpu()` on MPS. On
  CPU and MPS the numerics are unchanged.

- **Left `compute_wu_subspace_entropy()` on CPU deliberately, even under CUDA.**
  Its inner loop is (layer x token x k x alpha) and each iteration ends in
  `renyi_entropy(...).item()` — a device sync guarding a small `[d_model, k]`
  matmul. On GPU the sync would cost more than the matmul saves. The logit-lens
  and c_k paths are the opposite shape (one large matmul per sync) and did move.
  Commented in the source so it does not read as an oversight.

- **Verified the device change against committed results rather than trusting
  it.** Re-ran posthoc ablation on gpt2-small with the same explained-variance k
  values as the stored `.npz`: 450/450 records matched, max |ΔKL| 4.1e-4, max
  |ΔH| 1.7e-4, `entropy_full` bit-identical, zero top-1 flips. Those deltas are
  float32 matmul-reassociation noise, inside the 1e-3 tolerance
  `validate_ablation()` already documents.

- **Corpus expansion designed to break a confound, not just add n.** Nearly
  every original contrast appended a low-frequency abstract noun, so "contrast"
  meant both *structure broken* and *unusual token present* — the r⊥ effect was
  attackable as a lexical-frequency artifact. Four `contrast_type` levels
  (abstract / concrete / in_domain / swap) now span every category, with
  abstract-vs-in_domain as the key comparison. Scaling the original design to
  n=108 would have scaled the confound with it.

- **`--legacy` regenerates the original corpus byte-for-byte** on every original
  field including `pair_id` ordering, verified by direct comparison against the
  committed n=50 file. Without this the expansion would have silently
  invalidated every stored result.

- **`--last-token-only` guards rather than degrades.** Eight of nine c_k plots
  slice `[:, -1, :]`; the ninth would have silently averaged over a single token
  under a title claiming otherwise. It now raises, as does
  `CkRecord.layer_spectrum()`, and `--token N` fails at parse time rather than
  mid-corpus.

**Considered and rejected**

- **Rewriting the ablation inner loop into batched matmuls.** Offered as a third
  scope option and declined in favor of the smaller diff. It remains the largest
  available speedup but touches the numerics of the primary result, so it would
  need validation against the stored `.npz` files. Worth revisiting only if
  ablation runtime becomes the bottleneck.

- **Threading `contrast_type` through the Record types.** Would make stratified
  analysis a one-line filter instead of a join against the corpus JSON, but it
  touches every `save_*`/`load_*` pair and every Record dataclass. Deliberately
  left as a FutureWork item rather than done piecemeal in one record type.

- **Storing c_k for all tokens on Colab.** Initially justified `--last-token-only`
  on Drive-quota grounds with a bad estimate (~700MB for gpt2-small; actually
  ~60MB — the estimate assumed long prompts, but corpus prompts are 4-8 tokens).
  Storage was never the binding constraint. The flag was kept anyway for a real
  reason: 18M -> 1.9M measured, mostly from not NaN-padding every record to the
  corpus-wide `max_seq_len`, which matters for dashboard and notebook load time.

- **Driving Colab from this session.** Not possible. Colab has no public API for
  runtime creation or notebook execution, free-tier runtime allocation is gated
  behind an interactive browser session, and `drive.mount()` requires an OAuth
  click. The README now documents the manual sequence instead.

**Open issues**

- **The CUDA paths have never executed.** Everything was verified on CPU and by
  confirming the CPU/MPS branches are unchanged. Cell 5 of the Colab notebook is
  the smoke test; run it before any long job.

- **The expanded corpus has produced no stored results.** All of `data/` is from
  the 25-pair corpus. The n=108 ablation run done this session was a spot check
  and was not saved.

- **`data/ck/` is empty.** The c_k workflow has never produced a corpus result
  for any model, despite being complete since April. A single-model heatmap
  rendered during testing does look consistent with the build-then-cash-out
  hypothesis — energy concentrating into low-k directions with depth, with the
  base−contrast difference localized in the lowest-k directions at the deepest
  layers — but that is one model on the old corpus and is not a result.

- **Per-(category x contrast_type) cells hold only 3-5 pairs.** Aggregate and
  per-contrast_type comparisons are well powered at n=108; the full
  cross-tabulation is not. Category balance is also uneven (pattern 25,
  syntactic 24, predictability 23, arithmetic 21, repetition 15).

- **`repetition` x `swap` is empty by construction** — shuffling
  `the the the the` produces an identical string. Expected, not a gap to fill.

- **Colab free tier binds on host RAM before VRAM** for large models over the
  full corpus (~12GB RAM; `extract_activations()` returns numpy on the host).
  Chunk the corpus if pythia-2.8b over 216 prompts x 3 hooks fails.

**Notes**

- Found and fixed a latent bug while porting: `ablation_analysis.py` pinned
  `W_U` to `.cpu()` while leaving `ln_final` on the model device — unique among
  the six workflows. Harmless on CPU, would have crashed on CUDA in the ablation
  path specifically.

- The notebook builder had a rendering bug: `md()`/`code()` split on `\n`
  without `keepends`, so nbformat source lists had no line terminators and
  markdown cells collapsed into one run-on paragraph. `nbformat.validate()`
  passes on this — it checks structure, not line terminators — so the first
  validation gave false confidence. `build_notebook.py` now asserts trailing
  newlines explicitly.

- `corpus_gen.py` gained a `validate_corpus()` that fails closed on duplicate
  descriptions, identical base/contrast pairs, missing legacy descriptions, and
  unknown contrast_types. It caught 8 duplicates during authoring, one of which
  collided with a legacy entry and would have made `--legacy` ambiguous.

- DEVELOPER_NOTES.md does not exist in this repo. Granular flag and function
  listings currently live in README.md and CLAUDE.md.
