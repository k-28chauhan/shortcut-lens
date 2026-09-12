"""Tests for shortcut_lens.stats: BH vs statsmodels, Fisher wrapper direction, bootstrap, seeds."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from shortcut_lens.seeding import make_rng
from shortcut_lens.stats import (
    aggregate_over_seeds,
    benjamini_hochberg,
    fisher_exact_one_sided,
    percentile_bootstrap,
)


def test_benjamini_hochberg_matches_hand_worked_example() -> None:
    # from docs/PLAN.md M2 checkpoint: "re-derive the BH procedure on paper for five p-values"
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.20])
    reject, q_values = benjamini_hochberg(p_values, q=0.05)

    np.testing.assert_array_equal(reject, [True, True, True, True, False])
    np.testing.assert_allclose(q_values, [0.05, 0.05, 0.05, 0.05, 0.20])


def test_benjamini_hochberg_matches_statsmodels_on_random_pvalues() -> None:
    rng = np.random.default_rng(0)
    p_values = rng.uniform(0.0, 1.0, size=25)

    reject, q_values = benjamini_hochberg(p_values, q=0.1)
    sm_reject, sm_q_values, _, _ = multipletests(p_values, alpha=0.1, method="fdr_bh")

    np.testing.assert_array_equal(reject, sm_reject)
    np.testing.assert_allclose(q_values, sm_q_values)


def test_benjamini_hochberg_none_significant() -> None:
    p_values = np.array([0.5, 0.6, 0.9])
    reject, _q_values = benjamini_hochberg(p_values, q=0.05)
    assert not reject.any()


def test_fisher_exact_one_sided_direction() -> None:
    # slice: 8 errors, 2 correct; rest: 2 errors, 8 correct -- slice clearly worse
    table = np.array([[8, 2], [2, 8]])
    odds_ratio, p_value = fisher_exact_one_sided(table)

    assert odds_ratio > 1.0
    assert p_value < 0.05

    # the reverse table (slice better than rest) should not be significant under "greater"
    reversed_table = np.array([[2, 8], [8, 2]])
    _reversed_odds, reversed_p = fisher_exact_one_sided(reversed_table)
    assert reversed_p > 0.5


def test_aggregate_over_seeds() -> None:
    mean, std = aggregate_over_seeds([0.8, 0.9, 1.0])
    assert mean == pytest.approx(0.9)
    assert std == pytest.approx(np.std([0.8, 0.9, 1.0], ddof=1))


def test_aggregate_over_seeds_requires_at_least_two() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        aggregate_over_seeds([0.9])


def test_percentile_bootstrap_is_deterministic_given_equal_rng_material() -> None:
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])

    result_a = percentile_bootstrap(values, make_rng(0, "test"), n_resamples=200)
    result_b = percentile_bootstrap(values, make_rng(0, "test"), n_resamples=200)

    assert result_a == result_b


def test_percentile_bootstrap_point_estimate_is_the_sample_mean() -> None:
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    point, ci_lo, ci_hi = percentile_bootstrap(values, make_rng(0, "test"), n_resamples=500)

    assert point == pytest.approx(np.mean(values))
    assert ci_lo <= point <= ci_hi


def test_percentile_bootstrap_with_strata_computes_worst_group_statistic() -> None:
    # group 0: all correct (mean 1.0); group 1: all wrong (mean 0.0) -> "WGA" statistic is 0.0
    values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    strata = np.array([0, 0, 0, 1, 1, 1])

    def worst_group_mean(values: np.ndarray, strata: np.ndarray) -> float:
        return min(float(np.mean(values[strata == g])) for g in np.unique(strata))

    point, ci_lo, ci_hi = percentile_bootstrap(
        values, make_rng(0, "test"), statistic_fn=worst_group_mean, strata=strata, n_resamples=200
    )
    assert point == pytest.approx(0.0)
    assert ci_lo <= point <= ci_hi
