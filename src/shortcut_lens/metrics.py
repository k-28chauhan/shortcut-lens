"""Pure metric functions: accuracy, group accuracy, worst-group accuracy, precision@k, AUROC.

Concept: metrics are computed here from arrays the caller already has -- no group labels are
loaded inside this module. That is what lets the exact same functions be called from both zones:
`evaluation/core.py` (oracle zone, with true groups) and, later, anywhere label-free code needs a
metric that does not require groups at all (e.g. average accuracy for checkpoint selection, D-004).
Every metric here has a frozen definition in docs/PRD.md §12 -- changing one requires a
DECISIONS entry (CLAUDE.md §3 rule 4). All accuracy-like metrics return fractions in [0, 1], not
percentages; "points" in gate thresholds and PRD prose means percentage points, i.e. a fraction
difference times 100 (D-030).

Pipeline position: shared, pure. No I/O, no dataset access.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Fraction of `y_hat` equal to `y` over the whole array."""
    return float(np.mean(y == y_hat))


def group_accuracy(y: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> dict[int, float]:
    """Accuracy within each unique value of `group`, as `{group_id: accuracy}`."""
    correct = y == y_hat
    return {
        int(group_id): float(np.mean(correct[group == group_id])) for group_id in np.unique(group)
    }


def worst_group_accuracy(y: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> float:
    """Minimum group accuracy over all groups present (PRD §12: WGA)."""
    return min(group_accuracy(y, y_hat, group).values())


def mean_group_accuracy(y: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> float:
    """Unweighted mean of per-group accuracies (every group counts equally regardless of size)."""
    values = list(group_accuracy(y, y_hat, group).values())
    return float(np.mean(values))


def weighted_average_accuracy(
    group_accuracies: Mapping[int, float], weights: Mapping[int, float]
) -> float:
    """Per-group accuracies weighted by `weights` (PRD §12: training-set group proportions).

    Takes plain mappings so this stays testable with arbitrary weights and provable independent of
    where the weights came from -- the oracle-zone caller (`evaluation/core.py`) is responsible for
    computing training-set group proportions from `oracle.parquet` and passing them in here.
    """
    if group_accuracies.keys() != weights.keys():
        raise ValueError(
            f"group_accuracies keys {sorted(group_accuracies)} != weights keys {sorted(weights)}"
        )
    return float(sum(group_accuracies[g] * weights[g] for g in group_accuracies))


def wga_gap(y: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> float:
    """Average accuracy minus worst-group accuracy (PRD §12: WGA gap)."""
    return accuracy(y, y_hat) - worst_group_accuracy(y, y_hat, group)


def precision_at_k(scores: np.ndarray, target_membership: np.ndarray, k: int) -> float:
    """Among the `k` highest-scoring examples, the fraction in `target_membership` (PRD §12).

    `scores` and `target_membership` must already cover exactly one class's examples (see
    `ClassView`, docs/ARCHITECTURE.md §3) -- this function does not filter by class itself.
    Raises if `k` exceeds the number of examples available: precision@k is undefined below `k`
    examples, and silently shrinking `k` would produce a plausible-looking but different number
    than the frozen definition asks for.
    """
    if len(scores) != len(target_membership):
        raise ValueError("scores and target_membership must have the same length")
    if k > len(scores):
        raise ValueError(f"k={k} exceeds the number of available examples ({len(scores)})")
    top_k = np.argsort(scores)[::-1][:k]
    return float(np.mean(target_membership[top_k]))


def slice_auroc(scores: np.ndarray, target_membership: np.ndarray) -> float:
    """AUROC of `scores` for identifying `target_membership` (PRD §12: slice AUROC).

    `scores`/`target_membership` must already cover exactly one class's examples (see
    `precision_at_k`). Returns `NaN` when `target_membership` is all-True or all-False -- AUROC is
    undefined with a single class present, and a degenerate slice is a real possible outcome here,
    not a bug, so it is reported explicitly rather than raised. Verified `sklearn.roc_auc_score`
    already returns `NaN` (with an `UndefinedMetricWarning`) in that case rather than raising.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        return float(roc_auc_score(target_membership, scores))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """|intersection| / |union| of two boolean masks; `NaN` if both masks are empty."""
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return float("nan")
    return float(np.sum(mask_a & mask_b) / union)


def recovery(wga_method: float, wga_erm: float, wga_dfr_oracle: float) -> float:
    """`(WGA_method - WGA_erm) / (WGA_dfr_oracle - WGA_erm)` (PRD §12).

    `NaN` if the denominator is under 2 percentage points (0.02 in the fraction units used here) --
    too small a gap between ERM and the oracle reference to divide by without the ratio being
    dominated by noise.
    """
    denominator = wga_dfr_oracle - wga_erm
    if denominator < 0.02:
        return float("nan")
    return (wga_method - wga_erm) / denominator
