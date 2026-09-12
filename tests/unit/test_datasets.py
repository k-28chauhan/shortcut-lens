"""Tests for shortcut_lens.build.datasets.RenderedImageDataset: patch overlay, pass-through mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from shortcut_lens.build.datasets import RenderedImageDataset
from shortcut_lens.build.planting import PatchSpec


def _write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    Image.new("RGB", (64, 64), color).save(path)


def _public_table(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [f"id-{i}" for i in range(n)],
            "dataset": ["synthetic_shapes"] * n,
            "split": ["train"] * n,
            "y": [0, 1][:n],
            "class_name": ["circle", "square"][:n],
            "image_ref": [f"id-{i}.jpg" for i in range(n)],
        }
    )


def test_pass_through_mode_returns_image_unmodified(tmp_path: Path) -> None:
    table = _public_table()
    for ref in table["image_ref"]:
        _write_image(tmp_path / ref)

    dataset = RenderedImageDataset(table, tmp_path)
    item = dataset[0]
    assert item["example_id"] == "id-0"
    assert item["y"] == 0
    assert item["image"].size == (64, 64)


def test_len_matches_table_length(tmp_path: Path) -> None:
    table = _public_table(2)
    for ref in table["image_ref"]:
        _write_image(tmp_path / ref)
    dataset = RenderedImageDataset(table, tmp_path)
    assert len(dataset) == 2


def test_patch_applied_only_when_flagged(tmp_path: Path) -> None:
    table = _public_table(2)
    for ref in table["image_ref"]:
        _write_image(tmp_path / ref)

    spec = PatchSpec(size_px=16, color=(255, 0, 255))
    dataset = RenderedImageDataset(
        table, tmp_path, patch_spec=spec, patch_flags={"id-0": True, "id-1": False}
    )

    patched = dataset[0]["image"]
    unpatched = dataset[1]["image"]
    patched_colors = {color for _, color in patched.getcolors(64 * 64) or []}
    unpatched_colors = {color for _, color in unpatched.getcolors(64 * 64) or []}
    assert (255, 0, 255) in patched_colors
    assert (255, 0, 255) not in unpatched_colors


def test_requires_patch_flags_when_patch_spec_given(tmp_path: Path) -> None:
    table = _public_table(1)
    _write_image(tmp_path / "id-0.jpg")
    spec = PatchSpec(size_px=16, color=(255, 0, 255))

    with pytest.raises(ValueError, match="patch_flags"):
        RenderedImageDataset(table, tmp_path, patch_spec=spec)


def test_transform_is_applied(tmp_path: Path) -> None:
    table = _public_table(1)
    _write_image(tmp_path / "id-0.jpg")

    dataset = RenderedImageDataset(table, tmp_path, transform=lambda img: img.size)
    assert dataset[0]["image"] == (64, 64)
