"""Tests for shortcut_lens.evaluation.core: id-aligned join, tidy group-metrics table."""

from __future__ import annotations

import pandas as pd
import pytest

from shortcut_lens.evaluation.core import group_metrics_table, join_predictions_with_groups


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "y": [0, 0, 1, 1],
            "y_hat": [0, 1, 1, 0],  # a correct, b wrong, c correct, d wrong
        }
    )


def _oracle_groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["a", "b", "c", "d"],
            "group": [0, 0, 1, 1],
            "group_name": ["cat|no_patch", "cat|no_patch", "dog|patch", "dog|patch"],
        }
    )


def test_join_predictions_with_groups() -> None:
    joined = join_predictions_with_groups(_predictions(), _oracle_groups())
    assert len(joined) == 4
    assert set(joined.columns) >= {"example_id", "y", "y_hat", "group", "group_name"}


def test_join_raises_on_mismatched_ids() -> None:
    predictions = _predictions()
    oracle_groups = _oracle_groups().iloc[:-1]  # drop id "d"
    with pytest.raises(ValueError, match="do not cover the same example_ids"):
        join_predictions_with_groups(predictions, oracle_groups)


def test_join_raises_on_missing_columns() -> None:
    predictions = _predictions().drop(columns=["y_hat"])
    with pytest.raises(ValueError, match="missing required columns"):
        join_predictions_with_groups(predictions, _oracle_groups())


def test_group_metrics_table_matches_hand_computed_accuracies() -> None:
    table = group_metrics_table(_predictions(), _oracle_groups(), seed=0, n_resamples=200)

    assert list(table["group"]) == [0, 1]
    assert list(table["n"]) == [2, 2]
    # group 0 ("a" correct, "b" wrong): accuracy 0.5; group 1 ("c" correct, "d" wrong): accuracy 0.5
    assert table["accuracy"].tolist() == pytest.approx([0.5, 0.5])
    assert (table["ci_lo"] <= table["accuracy"]).all()
    assert (table["accuracy"] <= table["ci_hi"]).all()
    assert list(table["group_name"]) == ["cat|no_patch", "dog|patch"]
