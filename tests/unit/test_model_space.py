"""Tests for shortcut_lens.embeddings.model_space: shape, order, id alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from shortcut_lens.embeddings.model_space import embed_model_space
from shortcut_lens.models.backbones import build_backbone


class _FakeImageDataset:
    def __init__(self, n: int, size: int = 16) -> None:
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


def test_embed_model_space_shape_and_order() -> None:
    dataset = _FakeImageDataset(n=10)
    backbone = build_backbone("tiny_cnn", num_classes=2, pretrained=None)

    vectors, example_ids = embed_model_space(backbone, dataset, device="cpu", batch_size=4)

    assert vectors.shape == (10, backbone.feature_dim)
    assert example_ids == [f"ex-{i}" for i in range(10)]


def test_embed_model_space_matches_direct_features_call() -> None:
    dataset = _FakeImageDataset(n=3)
    backbone = build_backbone("tiny_cnn", num_classes=2, pretrained=None)

    vectors, _ids = embed_model_space(backbone, dataset, device="cpu", batch_size=8)

    images = torch.stack([dataset[i]["image"] for i in range(3)])
    backbone.eval()
    with torch.no_grad():
        expected = backbone.features(images).numpy()
    np.testing.assert_allclose(vectors, expected, atol=1e-6)
