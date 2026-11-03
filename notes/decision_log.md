# Decision log — Batch 1 (statistical controls on ablation findings)

Entries are appended in the order made. Entries 1–22 were written on
2026-11-02 BEFORE any Batch 1 number was computed (no flip rate, null, margin,
or regression had been looked at).

## Pre-registration (written before any computation)

1. **Gate.** `notes/predictions.md` exists but contains no Batch 1–specific predictions (only the 2026-11-02 framing paragraph). Proceeding on the user's explicit instruction "Do the work". Reason: user decision; recorded so the absence is visible.
2. **Pair IDs.** The ablation `.npz` files have no `pair_id` field; pair IDs are joined from `corpus/base_vs_contrast_n50.json` on exact prompt text (all 50 prompts match, verified before computation). Reason: the corpus has the field, so the "stop and ask" condition does not apply.
3. **Pythia-6.9b file.** Use `ablation_records_pythia-6.9b_base_vs_contrast_n50.npz`, not `_default_int_layers`. Reason: that is the filename the notebook loads.
4. **Primary k.** The k for the EV = 0.99 threshold (7th of the 9 EV-threshold ks: thresholds were 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0). Reason: the README/notebook claim is stated at 99% EV.
5. **Primary readout.** Post-hoc ablation, final layer (index n_layers−1), pre-LN projection (existing convention), top-1 flip = not `top1_preserved`. Reason: matches the notebook's fragility figure and is the model's actual output position.
6. **Reproduction criteria.** (a) An independent reader of the `.npz` must give exactly the same flip rates as the notebook's `npz_utils.get_ablation_records` path. (b) Re-running the existing post-hoc ablation math on freshly extracted activations must reproduce saved `top1_preserved` exactly for gpt2-small, gpt2-medium, pythia-160m (all layers, all k); and the intervention at the 5-layer subset (entry 11). (c) The published "~20% flip at 99% EV" is taken as the notebook figure's two quantities at EV = 0.99, averaged over the 8 models: post-hoc final-layer flip rate, and intervention flip rate averaged over layers with fractional depth in [0.3, 0.6] (the notebook's `FRAC_LO/FRAC_HI`). Matched if both lie in [0.15, 0.25]. Failure of (a), (b) or (c) → stop and report. Reason: "approximately 20%" needs a stated tolerance before looking.
7. **Bootstrap.** Resample the 25 pair IDs with replacement (both prompts of a pair travel together), B = 10,000, seed 20261102, 95% percentile CIs. Reason: pairs are the independent unit.
8. **Paired effect size.** Per pair d_i = flip_contrast − flip_base; report mean(d) and Cohen's d_z = mean(d)/sd(d) (ddof=1; NaN if sd = 0). Reason: standard paired effect size; sign convention makes the README claim "contrast flips more" positive.
9. **Random-subspace null basis.** Q = QR of a d_model×k standard Gaussian (numpy float64, cast to float32), `np.random.default_rng([20261102, d_model, k, draw])`. One basis per draw is applied to all 50 prompts; the null statistic is that draw's corpus flip rate. Reason: draw-level null for a corpus-level statistic; seeding by (d, k, draw) makes every draw reproducible independently of run order.
10. **Null sizes.** Post-hoc N = 100 (all models where activations can be extracted); intervention N = 50 (gpt2-small, gpt2-medium, pythia-160m only). Reason: as specified by the user.
11. **Layer subset.** Intervention nulls and post-hoc nulls both use layers `unique(round(linspace(0, n_layers−1, 5)))`. Post-hoc real W_U ablation is recomputed at all layers. Reason: user asked for a reduced layer subset for intervention; using the same subset for post-hoc nulls keeps cost bounded for the large models, and the primary readout (final layer) is in it.
12. **Null comparison.** Report the observed W_U flip rate's percentile within the null and a one-sided empirical p = (1 + #{null ≥ observed}) / (N + 1), plus null median and 5th/95th percentiles. Reason: fixed before seeing the null.
13. **Energy removed.** Per prompt, 1 − ‖QQᵀx‖² / ‖x‖², where x is the vector actually projected (pre-LN: raw residual at that layer; post-LN: ln_final(residual)). Report the mean over prompts per (k, draw). Reason: as specified by the user.
14. **Post-LN convention.** logits = (QQᵀ · ln_final(r)) @ W_U, compared with ln_final(r) @ W_U, with no b_U (matching the existing post-hoc code). Run for the real W_U basis and for the nulls, post-hoc only. Intervention uses pre-LN only. Reason: a post-LN projection has no counterpart inside the forward pass at intermediate layers.
15. **Margin.** From a clean forward pass, final position: logit gap = top-1 − top-2 logit (includes b_U), and top-1 softmax probability. Reason: this is the model's actual output, which the final-layer readout is compared against.
16. **Terciles.** Margin tercile boundaries at the 1/3 and 2/3 quantiles of all 50 prompts' logit gaps, per model, fixed once (not recomputed per bootstrap resample). Report the flip rate per tercile per role with pair-bootstrap CIs. Reason: avoids resample-dependent bins.
17. **Regression.** Logistic flip ~ 1 + z(logit gap) + role (contrast = 1), with the gap standardized within model, fit by unpenalized IRLS (≤ 100 iterations, tol 1e-8). A resample whose fit does not converge or has any |coef| > 15 (quasi-separation) is counted as failed; the CI is taken over converged resamples, and the convergence fraction is reported. If the full-sample fit fails, the adjusted analysis is "not estimable". Fit at the primary k for the verdict; all k reported as secondary. Reason: unpenalized avoids sklearn's default L2 shrinkage biasing the role coefficient toward 0.
18. **Fragility verdict** (per model, primary readout, pre-LN): survives if observed > null 95th percentile; weakened if null median < observed ≤ null 95th percentile; dies if observed ≤ null median. Reason: the claim is that W_U-complement removal is unusually disruptive, so the null defines "unusual".
19. **Differential verdict** (per model, primary readout): survives if the unadjusted mean(d) 95% CI lies entirely > 0 AND the margin-adjusted role coefficient 95% CI lies entirely > 0; weakened if mean(d) > 0 but either CI includes 0 (or adjusted is not estimable); dies if mean(d) ≤ 0. Reason: README direction is "base more stable".
20. **Overall verdict per claim.** The label held by the majority of {gpt2-small, gpt2-medium, pythia-160m}; if all three differ, "weakened". Other models are reported but not counted. Reason: these are the three models with the full control set (incl. intervention nulls).
21. **Secondary differential readout** (reported, not used for the verdict): the notebook's own summary, i.e. the intervention base−contrast top-1-preserved difference averaged over the 95%, 99% and 99.9% EV ks and over layers with fractional depth ≥ 0.75, with a pair-bootstrap CI. Reason: this is the statistic the notebook's "late-layer" claim rests on.
22. **Sanity checks.** k = d_model for the real basis and for every null draw (QR of a d×d Gaussian) must give zero flips: post-hoc under both conventions, and intervention. Seeds: rerun the gpt2-small post-hoc null and a slice of the intervention null and require bitwise-identical flip arrays. Any failure → results do not count until resolved. Reason: as specified.

## Implementation choices

23. **No new Record dataclass.** Control results are stored as array dicts via `save_controls`/`load_controls` in `src/ablation_controls_compute.py`. Reason: they are [conv, k, draw, layer, prompt] arrays that do not fit one-record-per-(prompt, k); AblationRecord must not be modified.
24. **Saved-file reader.** `saved_flip_table` reads the ablation `.npz` arrays once rather than calling `ablation_compute.load_ablation_records`. Reason: that loader re-reads each full array from the zip for every record and keeps slice views of every copy alive; on the gpt2-xl file it reached 23.7 GB RSS and was SIGKILLed (exit 137). Existing module left unmodified (out of scope). Logged as anomaly A1.
25. **EV-index check.** Verified the 7th saved k is the 99% EV point by recomputing W_U cumulative EV: gpt2-small k=726 → 0.9903; pythia-160m-deduped k=738 → 0.9904. Reason: rules out a labeling error before acting on the reproduction result.

## Outcomes

- **2026-11-02, Task 2 (reproduction): FAILED criterion (c). Stopped per instructions; no Batch 1 analysis run.**
  - (a) Independent reader matches the notebook code path exactly for all 8 models: PASS.
  - (b) Not reached (requires model runs, which come after the gate).
  - (c) At EV = 0.99, mean over 8 models: post-hoc final-layer flip 0.273 and intervention mid-depth flip 0.513. Both must lie in [0.15, 0.25]: FAIL (post-hoc is outside by 0.023, intervention by 0.263).

# Markov entropy-rate corpus (2026-11-02, before any model was run on it)

M1. **Dial.** P(λ) = (1−λ)Π + λU with Π a random single 20-cycle; λ solved by bisection so the row entropy equals the target H. Reason: every row has the same entropy and P is doubly stochastic, so the entropy rate is exactly h(λ) and the stationary (word-frequency) distribution is uniform at every H — only sequential structure varies.
M2. **Alphabet.** 20 concrete nouns, each a single token with leading space in both GPT-2 and Pythia tokenizers (verified per prompt: token count == length, 200/200, both tokenizers, all lengths). Reason: identical token alphabet across the dial and across model families.
M3. **Levels.** H ∈ {0, 0.5, …, 4.0, log2 20 = 4.32} bits/token, 10 seeds per level. Reason: even spacing over the full range of the dial.
M4. **Fresh cycle per sequence.** A new random cycle σ per prompt. Reason: otherwise all H = 0 prompts at a given start state would be identical, with no variance.
M5. **Contrast = shuffle of the same sequence.** Reason: keeps exact word counts, removes order. Note: with uniform stationary distribution, a shuffled sequence is ≈ an H = 4.32 sequence in expectation.
M6. **Lengths 64, 256, 1000.** 1024 was generated, then replaced by 1000: BOS + 1024 exceeds GPT-2's n_ctx of 1024. Reason: model constraint, found before any model run.
M7. **Seeding.** `default_rng([20261102, length, level_index, rep])` per sequence; regenerating each file reproduces it byte-identically (sha checked). Reason: reproducibility independent of generation order.
M8. **Leading space on the first word.** Reason: keeps every token in the ' word' form; without it the first word tokenizes differently.

## Markov analysis plan (adopted on the user's "Run", 2026-11-02, before any model run)

M9. **Residual entropy.** Existing `compute_residual_stream_entropy`, norm "energy", α = 1, every layer; per prompt, averaged over word positions in the second half of the prompt (positions L//2+1 … L; position 0 is BOS). Reason: proposed default, accepted; the second half gives the model context to infer the chain.
M10. **Output entropy.** Model next-token distribution at the final layer (existing `compute_logit_lens_entropy`, α = 1, last layer = ln_final(r) @ W_U), same positions, compared with the true rate H. Reason: proposed default, accepted.
M11. **Added before running (Claude's addition, flagged):** because the model distributes over ~50k tokens while the truth is over 20 words, also report (i) probability mass on the 20 alphabet tokens and (ii) entropy of the model distribution renormalized over the 20 alphabet tokens, and (iii) cross-entropy on the actual next word (positions with a next word only). All descriptive; (ii) is the like-for-like comparison with H.
M12. **Correlation test.** Spearman ρ(H, residual entropy) per layer over the 100 base prompts; same for the 100 shuffled prompts as control; 95% CIs from a pair bootstrap (B = 10,000, seed 20261102), resampling the 100 (base, shuffle) pairs. Same test for output entropy. Reason: proposed default, accepted.
M13. **Run order.** gpt2-small and pythia-160m at L = 64 and L = 256. L = 1000 deferred (memory). Reason: proposed default, accepted.
M14. **Added before running (Claude's addition, flagged):** logit-lens entropy (α = 1) at every layer, second-half mean, with the same Spearman test. Reason: the user said they lack intuition for how entropy varies across layers to the output; this is the per-layer view of the output side. Descriptive only.
M15. **Implementation.** Output stats use ln_final(r) @ W_U without b_U, the same convention as `compute_logit_lens_entropy`; level-mean CIs resample the 10 prompts within each H level (seeded by level). Reason: consistency with the existing logit lens.

## Markov outcomes (2026-11-02)

- Ran gpt2-small and pythia-160m at L = 64 and 256 exactly as planned in M9–M15; no parameter changed after results. Summaries: `data/markov/markov_summary_*.npz`; EntropyRecords: `data/markov/entropy_records_*.npz`; figures: `figures/workflows/markov_entropy_analysis/`.
- Runtime: about 4 min (L = 64) and 13 min (L = 256) per model on CPU.

# Phase 1 — toy transformer on in-context Markov chains (2026-11-03, before any training)

P1. **Replication target (from the literature, not a new prediction).** Edelman et al. (2024), "The Evolution of Statistical Induction Heads": 2-layer attention-only transformers trained on in-context Markov chains (fresh random transition matrix per sequence, rows ~ Dirichlet(1)) pass through stages — uniform → in-context unigram → in-context bigram — with plateaus, and the final drop coincides with a layer-1 previous-token head plus a layer-2 head attending to positions whose preceding token equals the current token. Replication succeeds if the model's loss ends near the Bayes-optimal bigram loss and the stage ordering and the two attention signatures appear. Reason: known answer for calibrating the entropy measures later.
P2. **Data.** k = 3 states, sequence length 100, rows ~ Dirichlet(α = 1), uniform start state, fresh matrix per sequence, sampled on the fly; fixed seeded eval set of 2048 sequences. Reason: matches the published setup closely enough to compare; the 20-word dial corpus is used later for evaluation, not training.
P3. **Model.** TransformerLens HookedTransformer, attn_only, 2 layers, 1 head per layer, d_model = d_head = 64, learned positional embeddings, LayerNorm, n_ctx = 100. Reason: minimal architecture that can implement a statistical induction head; one head per layer keeps attribution unambiguous.
P4. **Training.** AdamW, lr 1e-3, weight decay 0, batch 64, up to 20,000 steps, seed 0, eval and checkpoint every 250 steps. Reason: standard small-model settings; fixed before looking.
P5. **Reference predictors** (all computed exactly per position, in bits): uniform (log2 3); in-context unigram, (count_j + 1)/(t+1 + k); in-context bigram, (C[x_t, j] + 1)/(row total + k), which is Bayes-optimal for Dirichlet(1) rows. Report model cross-entropy and KL(strategy ‖ model) to each. Reason: these define the stages.
P6. **Attention signatures.** Layer 0: mean attention from t to t−1 (previous-token score). Layer 1: mean total attention from t to positions s with x_{s−1} = x_t (statistical-induction score), reported with its uniform-attention baseline. Reason: the two mechanisms named in P1.
P7. **Training is a new pipeline stage.** `src/toy_training.py` runs forward/backward passes; like `extraction.py` it produces the object later analyzed, so the "no forward passes in compute modules" rule (which protects analysis code) is not violated in spirit. Checkpoints `.pt` under data/toy/ (git-ignored). Reason: architecture convention, stated explicitly.

## Phase 1 outcome (2026-11-03), run k3_L100_l2h1d64_s0, config exactly as P2–P4

- Final (step 20,000): cross-entropy 1.2905 bits vs Bayes-optimal bigram 1.2781 (gap 0.012); KL(bigram ‖ model) 0.0121 vs KL(unigram ‖ model) 0.1733. Statistical-induction score 0.883 (uniform baseline 0.393); previous-token score 0.524. Training time 1904 s.
- Against P1: final loss near bigram — met; layer-1 statistical-induction signature — met; layer-0 previous-token signature — met. Stage ordering — NOT RESOLVED at the 250-step eval cadence: the model was already closer to bigram than unigram at step 250. The 200-step timing check (same config and seeds) showed KL→unigram 0.017 vs KL→bigram 0.175 at step 100, which suggests a brief unigram stage, but that check was not planned as a measurement.
- Anomaly: the previous-token score kept rising (0.15 at step 250 → 0.52 by step 5,000) long after the loss was near bigram (by step 500), and the induction score reached 0.78 at step 250 while the previous-token score was 0.15. Not investigated.
