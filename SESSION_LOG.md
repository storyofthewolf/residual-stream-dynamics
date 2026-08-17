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
  validation gave false confidence. `build_notebook.py` gained an explicit
  trailing-newline assertion (the generator was removed the next day — see the
  2026-08-17 entry — which eliminates this failure mode entirely).

- `corpus_gen.py` gained a `validate_corpus()` that fails closed on duplicate
  descriptions, identical base/contrast pairs, missing legacy descriptions, and
  unknown contrast_types. It caught 8 duplicates during authoring, one of which
  collided with a legacy entry and would have made `--legacy` ambiguous.

- DEVELOPER_NOTES.md does not exist in this repo. Granular flag and function
  listings currently live in README.md and CLAUDE.md.

---

## 2026-08-17

**Commits:** `8388519`, `31078f5`, `72889d7`, `a760c1b`, merged to `main` as
`619d932` (PR #1). This closes out the Colab work started 2026-08-16.

**Decisions**

- **Removed `colab/build_notebook.py`; the `.ipynb` is now the source of
  truth.** The generator was introduced to keep notebook diffs readable — JSON
  `.ipynb` produces noisy diffs and churns execution counts. That benefit did
  not survive contact with how the notebook is actually used.

  The deciding argument: the Colab notebook is *edited in Colab*, which writes
  `.ipynb`. A generator makes every Colab-side edit something that must be
  manually back-ported into Python or silently lost on the next build. That
  already happened once — a hand-edited `BRANCH = "colab-support"` was at risk
  of reverting to `"main"`, and `DEFAULT_BRANCH` was added purely to paper over
  a problem the generator itself created. The `_lines()` newline bug was also a
  pure artifact of synthesizing notebook JSON by hand; editing the `.ipynb`
  directly cannot produce that class of error.

  Treating a file as a build artifact only works when nothing else writes to
  it. Colab does. Do not reintroduce a generator.

- **Fixed notebook paths and requirements pins** (see commit `8388519`). The
  requirements bug was the more serious of the two: `transformer_lens>=6.37.6`
  and `sae_lens>=1.26.4` were transposed, and since transformer_lens has no 6.x
  release, `pip install -r requirements.txt` could not resolve at all. Anyone
  cloning the repo hit a hard failure. Found by checking the pins against the
  installed environment rather than reading them.

**Notes**

- Trade-off accepted with the generator removal: notebook diffs are noisier
  now. If that becomes annoying, an `nbstripout` filter on `.ipynb` is the
  standard fix and does not reintroduce the back-porting problem.

- Notebook kernel for local work is the Anaconda `base` env
  (`/opt/anaconda3/bin/python`, Python 3.11.11) — the only registered
  kernelspec, and what the notebook metadata already names.

**Later the same day — Colab guards, GPU confirmation, merge**

**Decisions**

- **Made the Colab notebook fail loudly outside Colab.** Running it in local
  Jupyter produced `nvidia-smi: command not found`, then `cuda available:
  False`, then a `FileNotFoundError` on `/content/residual-stream-dynamics`
  three cells later. All correct behavior — `/content` and `nvidia-smi` are
  Colab-only — but nothing said so, and the traceback pointed at a path that
  means nothing on a laptop. Added step 0, which raises immediately when
  `import google.colab` fails and points at the workflow scripts for local work.

- **The no-GPU case now raises instead of warning.** It previously printed
  "NO GPU — set Runtime..." and continued. A user could scroll past that and
  start a multi-hour job that silently ran on CPU. Failing closed is the right
  default when the whole point of the notebook is GPU access.

- **Set `BRANCH` back to `"main"` before merging**, so a fresh clone from the
  default branch pulls the default branch. Left as its own commit ahead of the
  merge rather than folded in, to keep the reason legible.

- **Merged via PR with a merge commit, not a squash.** The seven commits are
  individually meaningful and the doc/session-log history is worth keeping
  distinct in the graph.

**Notes**

- **The CUDA path is confirmed working.** A Colab T4 (driver 580.82.07, CUDA
  13.0, torch 2.11.0+cu128) attached successfully and the GPU check passed.
  This closes the "CUDA paths never executed" caveat carried since 2026-08-16 —
  though only the setup cells are verified; no full corpus workflow has been
  run on GPU end to end yet.

- **The T4 reports 14.6 GB usable, not 16.** Earlier model-fit estimates assumed
  ~16 GB, so the margin on pythia-2.8b (~11 GB fp16 plus activations) is thinner
  than documented. It should still fit; it is the one model where an OOM is
  plausible.

- **Colab ships torch 2.11.0+cu128**, newer than the local 2.10.0. This is why
  cell 3 installs `transformer_lens` and `sae_lens` but deliberately not torch —
  pip would replace the CUDA-matched build with a mismatched wheel.

- **Drive symlinks confirmed working.** Workflows print `/content/...` paths
  because they cannot see they are traversing a symlink; the bytes land in
  `MyDrive/residual-stream-dynamics/`. Worth re-checking the Drive path (not the
  `/content` one) before ending a long session — Colab's Drive mount can lag or
  fail silently on large writes.

- `gh` CLI is not installed on this machine, so the PR was created and merged in
  the browser. Installing it would let future PRs be driven from the terminal.
