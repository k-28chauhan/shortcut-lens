"""Tests for shortcut_lens.metrics: values vs sklearn/scipy references, bounds, edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, roc_auc_score

from shortcut_lens.metrics import (
    accuracy,
    group_accuracy,
    jaccard,
    mean_group_accuracy,
    precision_at_k,
    recovery,
    slice_auroc,
    weighted_average_accuracy,
    wga_gap,
    worst_group_accuracy,
)


def test_accuracy_matches_sklearn() -> None:
    y = np.array([0, 1, 1, 0, 1])
    y_hat = np.array([0, 1, 0, 0, 1])
    assert accuracy(y, y_hat) == pytest.approx(accuracy_score(y, y_hat))


def test_group_accuracy_and_worst_group_accuracy() -> None:
    y = np.array([0, 0, 1, 1, 1, 1])
    y_hat = np.array([0, 1, 1, 1, 0, 0])
    group = np.array([0, 0, 1, 1, 1, 1])
    # group 0: 1/2 correct; group 1: 2/4 correct
    accuracies = group_accuracy(y, y_hat, group)
    assert accuracies == {0: pytest.approx(0.5), 1: pytest.approx(0.5)}
    assert worst_group_accuracy(y, y_hat, group) == pytest.approx(0.5)
    assert mean_group_accuracy(y, y_hat, group) == pytest.approx(0.5)


def test_worst_group_accuracy_picks_the_minimum() -> None:
    y = np.array([0, 0, 0, 0])
    y_hat = np.array([0, 0, 1, 1])  # group 0 all correct, group 1 all wrong
    group = np.array([0, 0, 1, 1])
    assert worst_group_accuracy(y, y_hat, group) == pytest.approx(0.0)
    assert mean_group_accuracy(y, y_hat, group) == pytest.approx(0.5)


def test_wga_gap() -> None:
    y = np.array([0, 0, 1, 1])
    y_hat = np.array([0, 0, 1, 0])  # overall accuracy 0.75, group 1 accuracy 0.5
    group = np.array([0, 0, 1, 1])
    assert wga_gap(y, y_hat, group) == pytest.approx(0.75 - 0.5)


def test_weighted_average_accuracy() -> None:
    group_accuracies = {0: 0.9, 1: 0.5}
    weights = {0: 0.8, 1: 0.2}
    assert weighted_average_accuracy(group_accuracies, weights) == pytest.approx(
        0.9 * 0.8 + 0.5 * 0.2
    )


def test_weighted_average_accuracy_mismatched_keys_raises() -> None:
    with pytest.raises(ValueError, match="keys"):
        weighted_average_accuracy({0: 0.9, 1: 0.5}, {0: 1.0})


def test_precision_at_k() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
    target = np.array([False, True, False, True, False])
    # top 2 by score: index 1 (0.9, target) and index 3 (0.8, target) -> precision@2 = 1.0
    assert precision_at_k(scores, target, k=2) == pytest.approx(1.0)
    # top 3: adds index 2 (0.5, not target) -> precision@3 = 2/3
    assert precision_at_k(scores, target, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_raises_when_k_exceeds_available_examples() -> None:
    scores = np.array([0.1, 0.9])
    target = np.array([True, False])
    with pytest.raises(ValueError, match="exceeds"):
        precision_at_k(scores, target, k=10)


def test_slice_auroc_matches_sklearn() -> None:
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    target = np.array([False, False, True, True])
    assert slice_auroc(scores, target) == pytest.approx(roc_auc_score(target, scores))


def test_slice_auroc_degenerate_returns_nan() -> None:
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    all_false = np.array([False, False, False, False])
    assert np.isnan(slice_auroc(scores, all_false))


def test_jaccard() -> None:
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    # intersection = {0}, union = {0,1,2} -> 1/3
    assert jaccard(a, b) == pytest.approx(1 / 3)


def test_jaccard_empty_union_is_nan() -> None:
    a = np.array([False, False])
    b = np.array([False, False])
    assert np.isnan(jaccard(a, b))


def test_recovery_normal_case() -> None:
    # method halves the gap between ERM and oracle
    wga_erm, wga_oracle = 0.5, 0.9
    wga_method = 0.7
    assert recovery(wga_method, wga_erm, wga_oracle) == pytest.approx(0.5)


def test_recovery_nan_below_two_point_denominator() -> None:
    assert np.isnan(recovery(wga_method=0.51, wga_erm=0.50, wga_dfr_oracle=0.505))
