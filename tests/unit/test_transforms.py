"""Tests for shortcut_lens.data.transforms: output shape, dtype, determinism."""

from __future__ import annotations

import torch
from PIL import Image

from shortcut_lens.data.transforms import eval_transform, train_transform


def _sample_image() -> Image.Image:
    return Image.new("RGB", (224, 224), (100, 150, 200))


def test_train_transform_output_shape_and_dtype() -> None:
    out = train_transform()(_sample_image())
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_eval_transform_output_shape_and_dtype() -> None:
    out = eval_transform()(_sample_image())
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_eval_transform_is_deterministic() -> None:
    image = _sample_image()
    a = eval_transform()(image)
    b = eval_transform()(image)
    torch.testing.assert_close(a, b)
