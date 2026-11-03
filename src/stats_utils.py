"""stats_utils.py — Seeded pair bootstrap, paired effect sizes, logistic IRLS.

Stateless statistical helpers for the Batch 1 ablation controls. Pure numpy;
no model access, no file I/O, no plotting.

The corpus is built from base/contrast PAIRS, so the pair (not the prompt) is
the independent unit. Every resampling function here draws pair IDs with
replacement and carries both prompts of a drawn pair together.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

DEFAULT_SEED = 20261102
DEFAULT_B    = 10_000


# ============================================================================
# PAIR BOOTSTRAP
# ============================================================================

def pair_bootstrap_indices(
    pair_ids: np.ndarray,
    B:        int = DEFAULT_B,
    seed:     int = DEFAULT_SEED,
) -> list[np.ndarray]:
    """
    Draw B pair-level bootstrap resamples.

    Args:
        pair_ids: shape [n_prompts], the pair ID of each prompt
        B:        number of resamples
        seed:     RNG seed

    Returns:
        list of B integer index arrays into the prompt axis. Each resample
        draws n_pairs pair IDs with replacement and includes every prompt
        of each drawn pair (duplicates kept).
    """
    pair_ids = np.asarray(pair_ids)
    uniq     = np.unique(pair_ids)
    members  = {p: np.flatnonzero(pair_ids == p) for p in uniq}
    rng      = np.random.default_rng(seed)
    draws    = rng.integers(0, len(uniq), size=(B, len(uniq)))
    return [np.concatenate([members[uniq[j]] for j in row]) for row in draws]


def pair_bootstrap(
    stat_fn:  Callable[[np.ndarray], float],
    pair_ids: np.ndarray,
    B:        int = DEFAULT_B,
    seed:     int = DEFAULT_SEED,
) -> np.ndarray:
    """
    Bootstrap distribution of stat_fn(idx), where idx indexes prompts.

    stat_fn receives an index array into the prompt axis and returns a scalar
    (NaN allowed, e.g. for a failed fit). Returns shape [B].
    """
    return np.array([stat_fn(idx) for idx in pair_bootstrap_indices(pair_ids, B, seed)],
                    dtype=np.float64)


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI over the finite entries of samples."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return (np.nan, np.nan)
    a = (1.0 - level) / 2.0
    return (float(np.quantile(s, a)), float(np.quantile(s, 1.0 - a)))


# ============================================================================
# PAIRED EFFECT SIZES
# ============================================================================

def paired_differences(
    values:   np.ndarray,
    roles:    np.ndarray,
    pair_ids: np.ndarray,
    hi_role:  str = "contrast",
    lo_role:  str = "base",
) -> np.ndarray:
    """
    Per-pair difference d_i = value[hi_role] − value[lo_role].

    Requires exactly one prompt of each role per pair. Returns shape [n_pairs]
    ordered by sorted pair ID.
    """
    values, roles, pair_ids = map(np.asarray, (values, roles, pair_ids))
    out = []
    for p in np.unique(pair_ids):
        m  = pair_ids == p
        hi = values[m & (roles == hi_role)]
        lo = values[m & (roles == lo_role)]
        if hi.size != 1 or lo.size != 1:
            raise ValueError(f"pair {p}: expected one {hi_role} and one {lo_role}, "
                             f"got {hi.size} and {lo.size}")
        out.append(float(hi[0]) - float(lo[0]))
    return np.array(out, dtype=np.float64)


def paired_effect_size(d: np.ndarray) -> dict:
    """Mean paired difference and Cohen's d_z = mean(d)/sd(d) (ddof=1)."""
    d    = np.asarray(d, dtype=np.float64)
    mean = float(d.mean())
    sd   = float(d.std(ddof=1)) if d.size > 1 else np.nan
    dz   = mean / sd if (np.isfinite(sd) and sd > 0) else np.nan
    return {"mean_diff": mean, "sd_diff": sd, "dz": dz, "n_pairs": int(d.size)}


def pair_bootstrap_paired_diff(
    values:   np.ndarray,
    roles:    np.ndarray,
    pair_ids: np.ndarray,
    B:        int = DEFAULT_B,
    seed:     int = DEFAULT_SEED,
) -> dict:
    """
    Paired effect size for (contrast − base) with a pair-bootstrap CI on the mean.

    Resampling pairs is equivalent to resampling the per-pair differences d_i.
    """
    d   = paired_differences(values, roles, pair_ids)
    eff = paired_effect_size(d)
    rng = np.random.default_rng(seed)
    bs  = d[rng.integers(0, d.size, size=(B, d.size))].mean(axis=1)
    eff["ci_lo"], eff["ci_hi"] = percentile_ci(bs)
    return eff


def pair_bootstrap_rate(
    flips:    np.ndarray,
    pair_ids: np.ndarray,
    B:        int = DEFAULT_B,
    seed:     int = DEFAULT_SEED,
) -> dict:
    """Mean of a 0/1 (or real) per-prompt array with a pair-bootstrap CI."""
    flips = np.asarray(flips, dtype=np.float64)
    bs    = pair_bootstrap(lambda idx: flips[idx].mean(), pair_ids, B, seed)
    lo, hi = percentile_ci(bs)
    return {"rate": float(flips.mean()), "ci_lo": lo, "ci_hi": hi, "n": int(flips.size)}


# ============================================================================
# LOGISTIC REGRESSION (unpenalized IRLS)
# ============================================================================

def logistic_irls(
    X:        np.ndarray,
    y:        np.ndarray,
    max_iter: int = 100,
    tol:      float = 1e-8,
    max_abs_coef: float = 15.0,
) -> tuple[np.ndarray, bool]:
    """
    Unpenalized logistic regression by Newton/IRLS.

    Args:
        X: design matrix [n, p] (include an intercept column yourself)
        y: binary outcome [n]

    Returns:
        (coef [p], converged). converged is False on non-convergence, a
        singular Hessian, or any |coef| > max_abs_coef (quasi-separation).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu  = 1.0 / (1.0 + np.exp(-eta))
        w   = mu * (1.0 - mu)
        H   = X.T @ (X * w[:, None])
        g   = X.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return beta, False
        beta = beta + step
        if np.max(np.abs(beta)) > max_abs_coef:
            return beta, False
        if np.max(np.abs(step)) < tol:
            return beta, True
    return beta, False
