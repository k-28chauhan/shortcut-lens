"""Tests for shortcut_lens.embeddings.clip_space: shape, L2 normalisation, order, id alignment.

Uses the `fake_clip` fixture (tests/conftest.py) for fast tests; real open_clip is exercised only
in the `network`-marked test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from shortcut_lens.embeddings.clip_space import ClipEmbedConfig, embed_clip_space, load_clip


class _FakeImageDataset:
    def __init__(self, n: int, size: int = 224) -> None:
        rng = np.random.default_rng(0)
        self._images = rng.random((n, 3, size, size)).astype(np.float32)
        self._ids = [f"ex-{i}" for i in range(n)]

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": torch.tensor(self._images[index]),
            "y": 0,
            "example_id": self._ids[index],
        }


def test_embed_clip_space_is_l2_normalised_and_ordered(fake_clip: object) -> None:
    dataset = _FakeImageDataset(n=6)

    vectors, example_ids = embed_clip_space(fake_clip, dataset, device="cpu", batch_size=4)

    assert example_ids == [f"ex-{i}" for i in range(6)]
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_embed_clip_space_deterministic_given_same_model(fake_clip: object) -> None:
    dataset = _FakeImageDataset(n=4)

    vectors_a, _ = embed_clip_space(fake_clip, dataset, device="cpu", batch_size=2)
    vectors_b, _ = embed_clip_space(fake_clip, dataset, device="cpu", batch_size=2)

    np.testing.assert_array_equal(vectors_a, vectors_b)


@pytest.mark.network
def test_load_clip_real_model_encodes_to_expected_dimension() -> None:
    model, preprocess = load_clip(ClipEmbedConfig(), device="cpu")
    assert callable(preprocess)

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = model.encode_image(dummy)
    assert features.shape == (2, 512)
