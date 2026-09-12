"""Property tests (Hypothesis) for shortcut_lens.metrics: invariants over random inputs.

See docs/PLAN.md M2: permutation invariance, bounds, WGA <= mean-group <= max-group, metric
values equal sklearn references, recovery guard.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from sklearn.metrics import roc_auc_score

from shortcut_lens.metrics import (
    accuracy,
    group_accuracy,
    mean_group_accuracy,
    recovery,
    slice_auroc,
    worst_group_accuracy,
)

# a shared strategy: parallel y, y_hat, group arrays of equal length, group in {0, 1, 2}
_labels_and_groups = st.integers(min_value=4, max_value=30).flatmap(
    lambda n: st.tuples(
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
        st.lists(st.integers(0, 1), min_size=n, max_size=n),
        st.lists(st.integers(0, 2), min_size=n, max_size=n),
    )
)


@given(_labels_and_groups)
def test_accuracy_is_bounded(data: tuple[list[int], list[int], list[int]]) -> None:
    y, y_hat, _group = data
    result = accuracy(np.array(y), np.array(y_hat))
    assert 0.0 <= result <= 1.0


@given(_labels_and_groups)
def test_permutation_invariance(data: tuple[list[int], list[int], list[int]]) -> None:
    y, y_hat, group = (np.array(x) for x in data)
    rng = np.random.default_rng(0)
    permutation = rng.permutation(len(y))

    original = (
        accuracy(y, y_hat),
        group_accuracy(y, y_hat, group),
        worst_group_accuracy(y, y_hat, group),
        mean_group_accuracy(y, y_hat, group),
    )
    permuted = (
        accuracy(y[permutation], y_hat[permutation]),
        group_accuracy(y[permutation], y_hat[permutation], group[permutation]),
        worst_group_accuracy(y[permutation], y_hat[permutation], group[permutation]),
        mean_group_accuracy(y[permutation], y_hat[permutation], group[permutation]),
    )
    assert original == permuted


@given(_labels_and_groups)
def test_wga_at_most_mean_group_at_most_max_group(
    data: tuple[list[int], list[int], list[int]],
) -> None:
    y, y_hat, group = (np.array(x) for x in data)
    accuracies = group_accuracy(y, y_hat, group)
    wga = worst_group_accuracy(y, y_hat, group)
    mean_acc = mean_group_accuracy(y, y_hat, group)
    max_acc = max(accuracies.values())

    # floating point: allow a tiny tolerance
    assert wga <= mean_acc + 1e-9
    assert mean_acc <= max_acc + 1e-9


_target_and_scores = st.integers(min_value=4, max_value=30).flatmap(
    lambda n: st.tuples(
        st.lists(st.booleans(), min_size=n, max_size=n),
        st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=n, max_size=n),
    )
)


@given(_target_and_scores)
def test_slice_auroc_matches_sklearn_reference(data: tuple[list[bool], list[float]]) -> None:
    target, scores = (np.array(x) for x in data)
    if len(set(target.tolist())) < 2:
        return  # degenerate case is covered separately in tests/unit/test_metrics.py
    assert slice_auroc(scores, target) == roc_auc_score(target, scores)


@given(
    wga_erm=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    gap=st.floats(min_value=0.0, max_value=0.019, allow_nan=False),
    wga_method=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_recovery_guard_below_two_points(wga_erm: float, gap: float, wga_method: float) -> None:
    wga_dfr_oracle = wga_erm + gap
    assert np.isnan(recovery(wga_method, wga_erm, wga_dfr_oracle))
