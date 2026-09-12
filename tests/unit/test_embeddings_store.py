"""Tests for shortcut_lens.embeddings.store: round-trip, alignment checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shortcut_lens.embeddings.store import assert_aligned, read_embeddings, write_embeddings


def test_write_read_round_trip(tmp_path: Path) -> None:
    vectors = np.random.default_rng(0).standard_normal((5, 8)).astype(np.float32)
    ids = [f"ex-{i}" for i in range(5)]

    write_embeddings(tmp_path, "val_a", vectors, ids)
    read_vectors, read_ids = read_embeddings(tmp_path, "val_a")

    assert read_ids == ids
    # float16 round-trip: not exact, but close
    np.testing.assert_allclose(read_vectors, vectors, atol=1e-3)


def test_write_rejects_length_mismatch(tmp_path: Path) -> None:
    vectors = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="rows"):
        write_embeddings(tmp_path, "val_a", vectors, ["a", "b"])


def test_assert_aligned_passes_for_identical_order() -> None:
    assert_aligned(["a", "b", "c"], ["a", "b", "c"], context="test")


def test_assert_aligned_raises_on_different_order() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        assert_aligned(["a", "b", "c"], ["a", "c", "b"], context="test")


def test_assert_aligned_raises_on_different_members() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        assert_aligned(["a", "b"], ["a", "b", "c"], context="test")
