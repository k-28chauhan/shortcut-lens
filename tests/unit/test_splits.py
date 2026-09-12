"""Tests for shortcut_lens.build.splits: half-splits, class balance, attribute assignment."""

from __future__ import annotations

from shortcut_lens.build.splits import (
    assign_by_target_fraction,
    class_balance_by_subsampling,
    realistic_subsample,
    stratified_half_split,
)


def test_stratified_half_split_equal_within_stratum() -> None:
    ids = [f"id-{i}" for i in range(100)]
    strata = [0] * 60 + [1] * 40
    half_a, half_b = stratified_half_split(ids, strata, seed=0)

    assert len(half_a) + len(half_b) == 100
    assert set(half_a).isdisjoint(half_b)
    # 60 in stratum 0 -> 30/30; 40 in stratum 1 -> 20/20
    assert len([i for i in half_a if strata[ids.index(i)] == 0]) == 30
    assert len([i for i in half_a if strata[ids.index(i)] == 1]) == 20


def test_stratified_half_split_deterministic() -> None:
    ids = [f"id-{i}" for i in range(50)]
    strata = [i % 2 for i in range(50)]
    a1, b1 = stratified_half_split(ids, strata, seed=0)
    a2, b2 = stratified_half_split(ids, strata, seed=0)
    assert a1 == a2
    assert b1 == b2


def test_class_balance_by_subsampling() -> None:
    ids = [f"id-{i}" for i in range(150)]
    y = [0] * 100 + [1] * 50
    kept = class_balance_by_subsampling(ids, y, seed=0)

    kept_y = [y[ids.index(i)] for i in kept]
    assert kept_y.count(0) == kept_y.count(1) == 50


def test_assign_by_target_fraction_exact_count() -> None:
    ids = [f"id-{i}" for i in range(200)]
    present = assign_by_target_fraction(ids, 0.3, 0, "test")
    assert len(present) == 60


def test_assign_by_target_fraction_deterministic() -> None:
    ids = [f"id-{i}" for i in range(50)]
    a = assign_by_target_fraction(ids, 0.5, 0, "x")
    b = assign_by_target_fraction(ids, 0.5, 0, "x")
    assert a == b


def test_assign_by_target_fraction_differs_by_tag() -> None:
    ids = [f"id-{i}" for i in range(50)]
    a = assign_by_target_fraction(ids, 0.5, 0, "a")
    b = assign_by_target_fraction(ids, 0.5, 0, "b")
    assert a != b


def test_realistic_subsample_keeps_all_majority() -> None:
    ids = [f"id-{i}" for i in range(110)]
    y = [0] * 110
    is_minority = [False] * 100 + [True] * 10
    kept = realistic_subsample(ids, y, is_minority, minority_fraction=0.05, seed=0)

    majority_ids = set(ids[:100])
    assert majority_ids.issubset(set(kept))


def test_realistic_subsample_hits_target_fraction() -> None:
    ids = [f"id-{i}" for i in range(1100)]
    y = [0] * 1100
    is_minority = [False] * 1000 + [True] * 100
    kept = realistic_subsample(ids, y, is_minority, minority_fraction=0.05, seed=0)

    kept_minority = sum(1 for i in kept if is_minority[ids.index(i)])
    fraction = kept_minority / len(kept)
    assert abs(fraction - 0.05) < 0.01
