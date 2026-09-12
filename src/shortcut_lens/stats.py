"""Statistics: percentile bootstrap CIs, one-sided Fisher exact test, Benjamini-Hochberg.

Concept: this is how the project turns a single-run number into an honest interval, and how it
controls the false discovery rate when testing many candidate slices at once (docs/EXPERIMENTS.md
pre-registers `fdr_q`). Benjamini-Hochberg is implemented from scratch (checked against
`statsmodels`) because it is small, central to the confirmation step, and a good learning
exercise (docs/LEARNING_PATH.md Phase 2).

Pipeline position: shared, pure. Used by `evaluation/core.py`, `discovery/confirm.py`,
`verification/verify.py`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import fisher_exact


def percentile_bootstrap(
    values: np.ndarray,
    rng: np.random.Generator,
    statistic_fn: Callable[..., float] = np.mean,
    strata: np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for `statistic_fn(values)` (or `statistic_fn(values, strata)`).

    `rng` is an explicit `numpy.random.Generator` (CLAUDE.md §5) -- derive it with
    `seeding.make_rng(seed, *context_keys)` so independent bootstraps (e.g. one per group in
    `evaluation/core.py`) never share a stream by accident.

    Without `strata`: each resample draws `len(values)` indices with replacement from the whole
    array and calls `statistic_fn(resampled_values)` -- e.g. `statistic_fn=np.mean` for an accuracy
    CI, where `values` is a 0/1 correctness array.

    With `strata`: each resample draws *within* each stratum, at that stratum's own size, with
    replacement (so relative group sizes stay fixed across resamples -- a stratified bootstrap),
    then calls `statistic_fn(resampled_values, resampled_strata)` -- e.g. a `statistic_fn` that
    computes each group's mean and returns the minimum, for a worst-group-accuracy CI.

    Returns `(point_estimate, ci_lo, ci_hi)`, where `point_estimate` is `statistic_fn` evaluated on
    the original (non-resampled) data.
    """
    if strata is not None and len(values) != len(strata):
        raise ValueError("values and strata must have the same length")

    if strata is None:
        point_estimate = float(statistic_fn(values))
    else:
        point_estimate = float(statistic_fn(values, strata))

    n = len(values)
    resample_statistics = np.empty(n_resamples, dtype=float)

    if strata is None:
        for i in range(n_resamples):
            resample_indices = rng.integers(0, n, size=n)
            resample_statistics[i] = statistic_fn(values[resample_indices])
    else:
        stratum_indices = {
            stratum: np.flatnonzero(strata == stratum) for stratum in np.unique(strata)
        }
        for i in range(n_resamples):
            resampled_value_parts = []
            resampled_stratum_parts = []
            for stratum, indices in stratum_indices.items():
                draw = rng.integers(0, len(indices), size=len(indices))
                resampled_value_parts.append(values[indices[draw]])
                resampled_stratum_parts.append(np.full(len(indices), stratum))
            resample_statistics[i] = statistic_fn(
                np.concatenate(resampled_value_parts), np.concatenate(resampled_stratum_parts)
            )

    tail = (1.0 - confidence) / 2.0
    ci_lo, ci_hi = np.quantile(resample_statistics, [tail, 1.0 - tail])
    return point_estimate, float(ci_lo), float(ci_hi)


def fisher_exact_one_sided(table: np.ndarray) -> tuple[float, float]:
    """One-sided Fisher exact test for "the slice has a higher error rate than the rest."

    `table` must be laid out as `[[errors_in_slice, correct_in_slice],
    [errors_in_rest, correct_in_rest]]`. Verified against `scipy.stats.fisher_exact`: with this
    layout, `alternative="greater"` tests whether the odds ratio
    `(errors_in_slice * correct_in_rest) / (correct_in_slice * errors_in_rest)` is greater than 1,
    i.e. whether the slice's error odds exceed the rest's -- exactly the one-sided hypothesis
    `discovery/confirm.py` needs (a slice can only be confirmed for having *excess* errors, not a
    deficit).

    Returns `(odds_ratio, p_value)`.
    """
    result = fisher_exact(table, alternative="greater")
    return float(result.statistic), float(result.pvalue)


def benjamini_hochberg(p_values: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg procedure: which hypotheses are significant, and their q-values.

    Formula (Benjamini & Hochberg, 1995): sort p-values ascending as p_(1) <= ... <= p_(m); reject
    all p_(i) for i <= the largest i with p_(i) <= (i / m) * q. The q-value (adjusted p-value) for
    rank i is `min_{j >= i} (p_(j) * m / j)`, computed as a reverse running minimum so it is
    monotone non-decreasing as i decreases -- this matches
    `statsmodels.stats.multitest.multipletests(method="fdr_bh")` exactly (verified on a
    hand-worked example, see `tests/unit/test_stats.py`).

    Returns `(reject, q_values)`, both in the original order of `p_values`.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    ranks = np.arange(1, m + 1)
    raw_q = sorted_p * m / ranks
    # reverse running minimum: q_(i) = min_{j >= i} raw_q_(j)
    sorted_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)

    thresholds = ranks / m * q
    below_threshold = sorted_p <= thresholds
    if below_threshold.any():
        largest_i = np.flatnonzero(below_threshold).max()
        sorted_reject = np.arange(m) <= largest_i
    else:
        sorted_reject = np.zeros(m, dtype=bool)

    reject = np.empty(m, dtype=bool)
    q_values = np.empty(m, dtype=float)
    reject[order] = sorted_reject
    q_values[order] = sorted_q
    return reject, q_values


def aggregate_over_seeds(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (`ddof=1`) across seeds."""
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        raise ValueError(f"need at least 2 seeds to compute a standard deviation, got {len(array)}")
    return float(np.mean(array)), float(np.std(array, ddof=1))
