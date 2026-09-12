"""Shared pytest fixtures. See docs/TESTING.md §2 for the full fixture list as it grows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_FAKE_CLIP_OUT_DIM = 32
_FAKE_CLIP_DOWNSAMPLE = 8


@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> Path:
    """A temporary artefact root, isolated per test."""
    root = tmp_path / "artifacts"
    root.mkdir()
    return root


class _FakeClipModel:
    """Deterministic stand-in for a real CLIP model: a fixed random projection of downsampled
    pixels (docs/TESTING.md §2) -- exposes just `encode_image()`, matching what
    `embeddings.clip_space.embed_clip_space` calls on the real thing."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        in_dim = 3 * _FAKE_CLIP_DOWNSAMPLE * _FAKE_CLIP_DOWNSAMPLE
        self._projection = torch.tensor(
            rng.standard_normal((in_dim, _FAKE_CLIP_OUT_DIM)), dtype=torch.float32
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        downsampled = F.adaptive_avg_pool2d(images, (_FAKE_CLIP_DOWNSAMPLE, _FAKE_CLIP_DOWNSAMPLE))
        return downsampled.flatten(1) @ self._projection


@pytest.fixture
def fake_clip() -> _FakeClipModel:
    """A `fake_clip` model: same `encode_image()` interface as `open_clip`'s, no download."""
    return _FakeClipModel()
