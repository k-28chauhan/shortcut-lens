"""Contract test: every `ImageDataset` implementation returns exactly {"image","y","example_id"}.

D-003 / FR-D4: this is what makes it safe to hand a dataset object to label-free code. Real
downloads (pets, waterbirds) are exercised in their own `network`-marked module tests; this test
covers the one dataset that needs no download, `synthetic_shapes`, on every CI run.
"""

from __future__ import annotations

from pathlib import Path

from shortcut_lens.build.datasets import RenderedImageDataset
from shortcut_lens.build.synthetic_shapes import SyntheticShapesConfig, build_synthetic_shapes
from shortcut_lens.types import ImageDataset


def test_rendered_image_dataset_satisfies_image_dataset_protocol(tmp_path: Path) -> None:
    config = SyntheticShapesConfig(variant="dot", rho=0.9, n_train=20, n_val=10, n_test=10)
    public, _ = build_synthetic_shapes(config, tmp_path / "images")

    dataset = RenderedImageDataset(public, tmp_path / "images")
    assert isinstance(dataset, ImageDataset)


def test_rendered_image_dataset_returns_exact_keys(tmp_path: Path) -> None:
    config = SyntheticShapesConfig(variant="dot", rho=0.9, n_train=20, n_val=10, n_test=10)
    public, _ = build_synthetic_shapes(config, tmp_path / "images")

    dataset = RenderedImageDataset(public, tmp_path / "images")
    for i in range(len(dataset)):
        item = dataset[i]
        assert set(item.keys()) == {"image", "y", "example_id"}
        assert isinstance(item["y"], int)
        assert isinstance(item["example_id"], str)
