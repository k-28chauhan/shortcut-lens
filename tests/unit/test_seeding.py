"""Tests for shortcut_lens.seeding: repeatable global draws, deterministic local generators."""

from __future__ import annotations

import random

import numpy as np
import torch

from shortcut_lens.seeding import make_rng, set_all_seeds


def test_set_all_seeds_makes_numpy_draws_repeatable() -> None:
    set_all_seeds(0)
    first = np.random.rand(5)
    set_all_seeds(0)
    second = np.random.rand(5)
    np.testing.assert_array_equal(first, second)


def test_set_all_seeds_makes_torch_draws_repeatable() -> None:
    set_all_seeds(0)
    first = torch.rand(5)
    set_all_seeds(0)
    second = torch.rand(5)
    torch.testing.assert_close(first, second)


def test_set_all_seeds_makes_python_random_repeatable() -> None:
    set_all_seeds(0)
    first = [random.random() for _ in range(5)]
    set_all_seeds(0)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_all_seeds_different_seeds_diverge() -> None:
    set_all_seeds(0)
    first = torch.rand(5)
    set_all_seeds(1)
    second = torch.rand(5)
    assert not torch.equal(first, second)


def test_make_rng_deterministic_for_same_seed_and_keys() -> None:
    a = make_rng(0, "example-1", "build")
    b = make_rng(0, "example-1", "build")
    np.testing.assert_array_equal(a.random(5), b.random(5))


def test_make_rng_differs_across_keys() -> None:
    a = make_rng(0, "example-1")
    b = make_rng(0, "example-2")
    assert not np.array_equal(a.random(5), b.random(5))


def test_make_rng_differs_across_seeds() -> None:
    a = make_rng(0, "example-1")
    b = make_rng(1, "example-1")
    assert not np.array_equal(a.random(5), b.random(5))


def test_make_rng_returns_independent_generator_instances() -> None:
    a = make_rng(0, "k")
    b = make_rng(0, "k")
    assert a is not b
