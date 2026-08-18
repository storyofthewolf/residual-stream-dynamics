"""probe_compute.py — Linear probe computation over ActivationRecords.

Consumes ActivationRecords from extraction.py.
Produces ProbeRecords for consumption by probe_plots.py.

Pipeline position:
    extraction.py → COMPUTATION (this file) → probe_plots.py

--------------------------------------------------------------------------
What this measures, and why it is not an entropy metric
--------------------------------------------------------------------------
Entropy is a scalar summary of a distribution: it discards direction. Two
residual streams can encode very different content while having near-identical
entropy. If a property is carried as a DIRECTION in activation space, entropy is
close to the worst available probe for it.

This module asks the complementary question: at each layer, is the corpus's
base/contrast distinction linearly decodable from the residual stream at the
final token? It fits a regularized logistic classifier per layer and reports
cross-validated accuracy.

On the moral corpus this was decisive where entropy was not — entropy differences
were ~0.02 bits with signs that flipped across model sizes, while the probe
separated the two roles at 0.98 accuracy.

--------------------------------------------------------------------------
Three design points that determine whether the number means anything
--------------------------------------------------------------------------
1. GROUPED CROSS-VALIDATION. Each corpus pair contributes a base and a contrast
   prompt sharing a frame and a category. Splitting the two across folds lets the
   classifier exploit frame identity rather than the contrast of interest, which
   inflates accuracy. Folds are always split by pair_id (StratifiedGroupKFold)
   so both members of a pair land in the same fold. Do not replace this with a
   plain KFold.

2. PERMUTATION NULL. With n≈120 samples and d_model up to 1600, chance accuracy
   is not reliably 0.5. The null is estimated by refitting on shuffled labels,
   with whole pairs flipped together so the group structure is preserved.

3. GENERALIZATION SPLITS. Within-distribution accuracy conflates "the model
   represents the property" with "the classifier memorized this vocabulary".
   `generalize_by` holds out an entire category sub-factor (foundation, level, or
   style) and tests transfer to it. On the moral corpus, leave-one-foundation-out
   stayed at 0.95 (the property generalizes across foundations) while
   leave-one-style-out fell to near chance (the direction is frame-specific).
   The generalization number is the scientifically meaningful one; report it
   alongside the within-distribution number, never instead of it.

Layer 0 is retained deliberately. It is close to the embedding, so its accuracy
is the lexical baseline — how much of the separation is available from token
identity before any computation. The quantity of interest is the RISE above
layer 0, not the peak.

ProbeRecord fields:
    model_name    : str
    corpus_tag    : str
    hook_type     : str
    generalize_by : str | None   — None for grouped CV; else "foundation"/"level"/"style"
    accuracy      : np.ndarray  [n_layers]
    null_mean     : np.ndarray  [n_layers]
    p_value       : np.ndarray  [n_layers]
    group_names   : list[str]   — held-out group labels (empty when generalize_by is None)
    group_accuracy: np.ndarray  [n_layers, n_groups]  — per-held-out-group accuracy
    n_samples / n_pairs / n_layers : int

Public functions:
    compute_probe(records, ...)             -> ProbeRecord   (grouped CV + permutation null)
    compute_probe_generalization(records, by) -> ProbeRecord  (leave-one-group-out)

Serialization:
    save_probe_records(records, path)
    load_probe_records(path) -> list[ProbeRecord]

Requires scikit-learn. No torch, no model access, no forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from extraction import ActivationRecord


# Sub-factor positions within the compound `category` string, which the moral
# and neutral corpora build as f"{foundation}_{level}_{style}".
CATEGORY_FACTORS = {"foundation": 0, "level": 1, "style": 2}

DEFAULT_C = 0.05          # inverse regularization; small because d_model >> n
DEFAULT_N_SPLITS = 5
DEFAULT_N_PERM = 200


# ============================================================================
# DATACLASS
# ============================================================================

@dataclass
class ProbeRecord:
    """Per-layer linear decodability of the base/contrast distinction."""
    model_name:     str
    corpus_tag:     str
    hook_type:      str
    accuracy:       np.ndarray
    null_mean:      np.ndarray
    p_value:        np.ndarray
    n_samples:      int
    n_pairs:        int
    n_layers:       int
    generalize_by:  Optional[str] = None
    group_names:    list = field(default_factory=list)
    group_accuracy: Optional[np.ndarray] = None

    def rise_above_layer0(self) -> np.ndarray:
        """Accuracy gain over the layer-0 (near-embedding) lexical baseline.

        This is the part attributable to computation rather than token identity.
        """
        return self.accuracy - self.accuracy[0]


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _require_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression       # noqa: F401
        from sklearn.model_selection import StratifiedGroupKFold  # noqa: F401
        from sklearn.preprocessing import StandardScaler          # noqa: F401
        from sklearn.pipeline import make_pipeline                # noqa: F401
    except ImportError as e:
        raise ImportError(
            "probe_compute requires scikit-learn. Install with: pip install scikit-learn"
        ) from e


def _make_clf(C: float):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))


def _prepare(records: list, token_pos: int = -1):
    """Filter to base/contrast records and extract labels, groups, and layer count.

    Returns (records, y, groups, n_layers). Raises on an unusable corpus.
    """
    usable = [r for r in records if r.role in ("base", "contrast")]
    if not usable:
        raise ValueError("No records with role 'base' or 'contrast' — "
                         "probe requires a paired corpus.")
    if any(r.pair_id is None for r in usable):
        raise ValueError("Some records have pair_id=None; grouped CV requires "
                         "pair_id on every record.")
    y = np.array([1 if r.role == "base" else 0 for r in usable], dtype=int)
    if len(set(y)) < 2:
        raise ValueError("Only one role present — nothing to discriminate.")
    groups = np.array([int(r.pair_id) for r in usable])
    n_layers = min(r.activations.shape[0] for r in usable)
    return usable, y, groups, n_layers


def _layer_matrix(records: list, layer: int, token_pos: int) -> np.ndarray:
    """Stack the residual vector at one layer and token position across records."""
    return np.stack([r.activations[layer, token_pos, :] for r in records])


# ============================================================================
# PUBLIC COMPUTE FUNCTIONS
# ============================================================================

def compute_probe(
    records: list,
    model_name: str = "",
    corpus_tag: str = "",
    token_pos: int = -1,
    C: float = DEFAULT_C,
    n_splits: int = DEFAULT_N_SPLITS,
    n_perm: int = DEFAULT_N_PERM,
    seed: int = 0,
    verbose: bool = True,
) -> ProbeRecord:
    """Fit a per-layer linear probe with grouped CV and a permutation null.

    Folds are split by pair_id so a pair's base and contrast never straddle a
    fold boundary. The null flips whole pairs, preserving group structure.
    """
    _require_sklearn()
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

    recs, y, groups, n_layers = _prepare(records, token_pos)
    rng = np.random.default_rng(seed)
    uniq_groups = sorted(set(groups))

    acc  = np.zeros(n_layers)
    nullm = np.zeros(n_layers)
    pval = np.zeros(n_layers)

    hook = recs[0].hook_type
    if verbose:
        print(f"  Probe: {len(recs)} prompts, {len(uniq_groups)} pairs, "
              f"{n_layers} layers, d_model={recs[0].d_model}")

    for L in range(n_layers):
        X = _layer_matrix(recs, L, token_pos)
        clf = _make_clf(C)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        acc[L] = cross_val_score(clf, X, y, groups=groups, cv=cv,
                                 scoring="accuracy").mean()
        null = np.empty(n_perm)
        for i in range(n_perm):
            yp = y.copy()
            for g in uniq_groups:
                if rng.random() < 0.5:
                    m = groups == g
                    yp[m] = 1 - yp[m]
            null[i] = cross_val_score(clf, X, yp, groups=groups, cv=cv,
                                      scoring="accuracy").mean()
        nullm[L] = null.mean()
        pval[L]  = (null >= acc[L]).mean()
        if verbose:
            star = "*" if pval[L] < 0.05 else " "
            print(f"    L{L:2d}  acc={acc[L]:.3f}  null={nullm[L]:.3f}  p={pval[L]:.3f} {star}")

    return ProbeRecord(
        model_name=model_name, corpus_tag=corpus_tag, hook_type=hook,
        accuracy=acc, null_mean=nullm, p_value=pval,
        n_samples=len(recs), n_pairs=len(uniq_groups), n_layers=n_layers,
        generalize_by=None, group_names=[], group_accuracy=None,
    )


def compute_probe_generalization(
    records: list,
    by: str,
    model_name: str = "",
    corpus_tag: str = "",
    token_pos: int = -1,
    C: float = DEFAULT_C,
    verbose: bool = True,
) -> ProbeRecord:
    """Leave-one-group-out probe, where groups are a `category` sub-factor.

    `by` is one of "foundation", "level", "style" — an index into the compound
    category string. Trains on all groups but one and tests on the held-out
    group, whose vocabulary the classifier has never seen. This separates "the
    model represents the property" from "the classifier memorized the words".

    No permutation null here: with few groups the leave-one-out estimate is
    already the conservative number, and chance is 0.5 by construction because
    every held-out group is role-balanced.
    """
    _require_sklearn()

    if by not in CATEGORY_FACTORS:
        raise ValueError(f"generalize_by must be one of {sorted(CATEGORY_FACTORS)}, got {by!r}")

    recs, y, groups, n_layers = _prepare(records, token_pos)
    idx = CATEGORY_FACTORS[by]

    missing = [r for r in recs if not r.category or len(r.category.split("_")) <= idx]
    if missing:
        raise ValueError(
            f"{len(missing)} records lack a compound category with a '{by}' field. "
            f"This corpus does not use the foundation_level_style convention."
        )

    key = np.array([r.category.split("_")[idx] for r in recs])
    group_names = sorted(set(key))
    if len(group_names) < 2:
        raise ValueError(f"Need >=2 distinct '{by}' groups to hold one out; found {group_names}")

    group_acc = np.zeros((n_layers, len(group_names)))
    if verbose:
        print(f"  Leave-one-{by}-out: {len(group_names)} groups "
              f"({', '.join(group_names)})")

    for L in range(n_layers):
        X = _layer_matrix(recs, L, token_pos)
        for gi, g in enumerate(group_names):
            tr, te = key != g, key == g
            clf = _make_clf(C)
            clf.fit(X[tr], y[tr])
            group_acc[L, gi] = clf.score(X[te], y[te])
        if verbose:
            print(f"    L{L:2d}  " + "  ".join(f"{g[:8]}={group_acc[L, i]:.3f}"
                                               for i, g in enumerate(group_names))
                  + f"   mean={group_acc[L].mean():.3f}")

    acc = group_acc.mean(axis=1)
    return ProbeRecord(
        model_name=model_name, corpus_tag=corpus_tag, hook_type=recs[0].hook_type,
        accuracy=acc,
        null_mean=np.full(n_layers, 0.5),   # balanced held-out groups
        p_value=np.full(n_layers, np.nan),  # no permutation test in this mode
        n_samples=len(recs), n_pairs=len(set(groups)), n_layers=n_layers,
        generalize_by=by, group_names=group_names, group_accuracy=group_acc,
    )


# ============================================================================
# SERIALIZATION
# ============================================================================

def save_probe_records(records: list, path) -> None:
    """Save a list of ProbeRecords to a single .npz file.

    NaN-pads to the maximum n_layers and n_groups across records, following the
    convention in entropy_compute / mechanics_compute.
    """
    n = len(records)
    max_layers = max(r.n_layers for r in records)
    max_groups = max(len(r.group_names) for r in records) or 1

    acc_pad  = np.full((n, max_layers), np.nan, dtype=np.float32)
    null_pad = np.full((n, max_layers), np.nan, dtype=np.float32)
    p_pad    = np.full((n, max_layers), np.nan, dtype=np.float32)
    gacc_pad = np.full((n, max_layers, max_groups), np.nan, dtype=np.float32)

    for i, r in enumerate(records):
        nl = r.n_layers
        acc_pad[i, :nl]  = r.accuracy
        null_pad[i, :nl] = r.null_mean
        p_pad[i, :nl]    = r.p_value
        if r.group_accuracy is not None:
            ng = r.group_accuracy.shape[1]
            gacc_pad[i, :nl, :ng] = r.group_accuracy

    np.savez(
        path,
        accuracy       = acc_pad,
        null_mean      = null_pad,
        p_value        = p_pad,
        group_accuracy = gacc_pad,
        n_layers       = np.array([r.n_layers  for r in records], dtype=np.int32),
        n_samples      = np.array([r.n_samples for r in records], dtype=np.int32),
        n_pairs        = np.array([r.n_pairs   for r in records], dtype=np.int32),
        model_names    = np.array([r.model_name for r in records], dtype=object),
        corpus_tags    = np.array([r.corpus_tag for r in records], dtype=object),
        hook_types     = np.array([r.hook_type  for r in records], dtype=object),
        generalize_by  = np.array(["" if r.generalize_by is None else r.generalize_by
                                   for r in records], dtype=object),
        group_names    = np.array([list(r.group_names) for r in records], dtype=object),
    )
    print(f"  Saved {n} ProbeRecords to {path}")


def load_probe_records(path) -> list:
    """Load ProbeRecords from an .npz written by save_probe_records."""
    d = np.load(path, allow_pickle=True)
    records = []
    for i in range(len(d["n_layers"])):
        nl = int(d["n_layers"][i])
        gb = str(d["generalize_by"][i]) or None
        names = list(d["group_names"][i])
        gacc = d["group_accuracy"][i, :nl, :len(names)] if names else None
        records.append(ProbeRecord(
            model_name    = str(d["model_names"][i]),
            corpus_tag    = str(d["corpus_tags"][i]),
            hook_type     = str(d["hook_types"][i]),
            accuracy      = d["accuracy"][i, :nl],
            null_mean     = d["null_mean"][i, :nl],
            p_value       = d["p_value"][i, :nl],
            n_samples     = int(d["n_samples"][i]),
            n_pairs       = int(d["n_pairs"][i]),
            n_layers      = nl,
            generalize_by = gb,
            group_names   = names,
            group_accuracy= gacc,
        ))
    return records
