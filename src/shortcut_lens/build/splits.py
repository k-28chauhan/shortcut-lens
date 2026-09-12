"""Stratified train/val_a/val_b/test splitting; balanced vs realistic validation composition.

Concept: `val_a` (discovery, last-layer retraining) and `val_b` (confirmation, selection) must be
disjoint so that discovering and confirming on the same data cannot let random clusters look like
real failures (D-005) -- `stratified_half_split` is what keeps that split reproducible and
class-balanced. `class_balance_by_subsampling` gives every dataset's training split equal class
counts. Two different tools build the "realistic" validation mode (D-006), depending on whether
the attribute can be *assigned* or only *observed*: `assign_by_target_fraction` picks which
existing examples get a boolean attribute (used where every image already exists and only a
patch/no-patch decision changes, e.g. PlantedPets); `realistic_subsample` instead drops examples
to hit a target composition (used where the attribute is a fixed, unchangeable fact about a real
photo, e.g. Waterbirds' background).

Pipeline position: oracle zone. Used by `build/pets.py`, `build/waterbirds.py`,
`build/synthetic_shapes.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from shortcut_lens.seeding import make_rng


def stratified_half_split(
    example_ids: Sequence[str], strata: Sequence[Any], seed: int
) -> tuple[list[str], list[str]]:
    """Split `example_ids` into two halves of equal size within each value of `strata`.

    Deterministic in `seed`. An odd-sized stratum puts its extra example in the first half.
    Used to split `val` into `val_a`/`val_b` stratified by class only (never by group -- that
    would require reading the oracle table here, which this module must not do).
    """
    if len(example_ids) != len(strata):
        raise ValueError("example_ids and strata must have the same length")

    rng = make_rng(seed, "stratified_half_split")
    ids_arr = np.asarray(example_ids)
    strata_arr = np.asarray(strata)

    half_a: list[str] = []
    half_b: list[str] = []
    for value in sorted(set(strata), key=str):
        stratum_ids = ids_arr[strata_arr == value]
        shuffled = stratum_ids[rng.permutation(len(stratum_ids))]
        cut = len(shuffled) // 2 + (len(shuffled) % 2)
        half_a.extend(shuffled[:cut].tolist())
        half_b.extend(shuffled[cut:].tolist())
    return half_a, half_b


def assign_by_target_fraction(
    example_ids: Sequence[str], target_fraction: float, seed: int, *tag: str
) -> set[str]:
    """Choose which `example_ids` get a boolean attribute, hitting `target_fraction` exactly (±1).

    Picks `round(target_fraction * len(example_ids))` ids by exact count, not independent coin
    flips, so the realised proportion never drifts by more than one example from the target --
    see docs/PLAN.md M1's "planted proportions within ±1 image of target per split" test.
    `*tag` (e.g. `(split, class_name)`) gives each call its own deterministic stream.
    """
    if not (0.0 <= target_fraction <= 1.0):
        raise ValueError(f"target_fraction must be in [0, 1], got {target_fraction}")
    rng = make_rng(seed, "assign_by_target_fraction", *tag)
    ids_arr = np.asarray(example_ids)
    n_present = round(target_fraction * len(ids_arr))
    order = rng.permutation(len(ids_arr))
    return set(ids_arr[order[:n_present]].tolist())


def class_balance_by_subsampling(
    example_ids: Sequence[str], y: Sequence[int], seed: int
) -> list[str]:
    """Subsample every class down to the smallest class's count, so classes end up exactly equal."""
    if len(example_ids) != len(y):
        raise ValueError("example_ids and y must have the same length")

    rng = make_rng(seed, "class_balance")
    ids_arr = np.asarray(example_ids)
    y_arr = np.asarray(y)
    classes, counts = np.unique(y_arr, return_counts=True)
    target = int(counts.min())

    kept: list[str] = []
    for cls in classes:
        cls_ids = ids_arr[y_arr == cls]
        chosen = cls_ids[rng.permutation(len(cls_ids))[:target]]
        kept.extend(chosen.tolist())
    return kept


def realistic_subsample(
    example_ids: Sequence[str],
    y: Sequence[int],
    is_minority: Sequence[bool],
    minority_fraction: float,
    seed: int,
) -> list[str]:
    """Within each class, keep every majority example and subsample minority examples down to
    `minority_fraction` of the resulting class size.

    `minority_fraction` is interpreted as `n_minority_kept / (n_majority + n_minority_kept)`, so
    e.g. `minority_fraction=0.05` means the minority group ends up as 5% of that class after
    subsampling -- matching how `rho` is expressed elsewhere in this project. Never subsamples the
    majority group, and never *upsamples* the minority group beyond what exists.
    """
    if not (0.0 <= minority_fraction < 1.0):
        raise ValueError(f"minority_fraction must be in [0, 1), got {minority_fraction}")
    if len({len(example_ids), len(y), len(is_minority)}) != 1:
        raise ValueError("example_ids, y and is_minority must have the same length")

    rng = make_rng(seed, "realistic_subsample")
    ids_arr = np.asarray(example_ids)
    y_arr = np.asarray(y)
    minority_arr = np.asarray(is_minority, dtype=bool)

    kept: list[str] = []
    for cls in np.unique(y_arr):
        cls_mask = y_arr == cls
        majority_ids = ids_arr[cls_mask & ~minority_arr]
        minority_ids = ids_arr[cls_mask & minority_arr]

        n_majority = len(majority_ids)
        target_minority = round(n_majority * minority_fraction / (1 - minority_fraction))
        n_minority_kept = min(target_minority, len(minority_ids))

        chosen_minority = minority_ids[rng.permutation(len(minority_ids))[:n_minority_kept]]
        kept.extend(majority_ids.tolist())
        kept.extend(chosen_minority.tolist())
    return kept
