"""Tests for shortcut_lens.build.cifar_pets. Network-marked: downloads CIFAR-10 on first run."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shortcut_lens.build.cifar_pets import CifarPetsConfig, build_cifar_pets
from shortcut_lens.build.datasets import RenderedImageDataset

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def torchvision_root() -> Path:
    return Path.home() / ".cache" / "shortcut-lens" / "torchvision"


def _build(
    tmp_path: Path, torchvision_root: Path, **overrides: object
) -> tuple[pd.DataFrame, pd.DataFrame]:
    defaults: dict[str, object] = {"rho": 0.95, "build_seed": 0}
    defaults.update(overrides)
    config = CifarPetsConfig.model_validate(defaults)
    return build_cifar_pets(config, torchvision_root, tmp_path / "images")


def test_splits_disjoint_and_dataset_returns_exact_keys(
    tmp_path: Path, torchvision_root: Path
) -> None:
    public, _ = _build(tmp_path, torchvision_root)
    by_split = {
        split: set(public.loc[public["split"] == split, "example_id"])
        for split in ("train", "val_a", "val_b", "test")
    }
    assert by_split["val_a"].isdisjoint(by_split["val_b"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert public["example_id"].is_unique


def test_train_is_class_balanced_and_patch_matches_rho(
    tmp_path: Path, torchvision_root: Path
) -> None:
    public, oracle = _build(tmp_path, torchvision_root, rho=0.95)
    merged = public.merge(oracle, on="example_id")
    train = merged[merged["split"] == "train"]

    class_counts = train["class_name"].value_counts()
    assert class_counts["cat"] == class_counts["dog"] == 5000  # CIFAR-10's official train count

    cat = train[train["class_name"] == "cat"]
    dog = train[train["class_name"] == "dog"]
    cat_patch_fraction = (cat["attribute"] == 1).mean()
    dog_patch_fraction = (dog["attribute"] == 1).mean()
    assert abs(cat_patch_fraction - 0.95) < 0.01
    assert abs(dog_patch_fraction - 0.05) < 0.01


def test_test_split_always_patch_balanced(tmp_path: Path, torchvision_root: Path) -> None:
    public, oracle = _build(tmp_path, torchvision_root, rho=0.95)
    merged = public.merge(oracle, on="example_id")
    test = merged[merged["split"] == "test"]

    for class_name in ("cat", "dog"):
        cls = test[test["class_name"] == class_name]
        fraction = (cls["attribute"] == 1).mean()
        assert abs(fraction - 0.5) < 0.05


def test_control_rho_has_near_zero_reliance_signal(tmp_path: Path, torchvision_root: Path) -> None:
    public, oracle = _build(tmp_path, torchvision_root, rho=0.5)
    merged = public.merge(oracle, on="example_id")
    train = merged[merged["split"] == "train"]

    for class_name in ("cat", "dog"):
        cls = train[train["class_name"] == class_name]
        fraction = (cls["attribute"] == 1).mean()
        assert abs(fraction - 0.5) < 0.03


def test_dataset_returns_exactly_image_y_example_id(tmp_path: Path, torchvision_root: Path) -> None:
    public, _ = _build(tmp_path, torchvision_root)
    dataset = RenderedImageDataset(public, tmp_path / "images")
    item = dataset[0]
    assert set(item.keys()) == {"image", "y", "example_id"}


def test_images_are_upscaled_to_224(tmp_path: Path, torchvision_root: Path) -> None:
    public, _ = _build(tmp_path, torchvision_root)
    from PIL import Image

    row = public.iloc[0]
    image = Image.open(tmp_path / "images" / row["image_ref"])
    assert image.size == (224, 224)


def test_patch_renders_correctly_on_upscaled_image(tmp_path: Path, torchvision_root: Path) -> None:
    import numpy as np

    from shortcut_lens.build.planting import PatchSpec, add_patch

    public, oracle = _build(tmp_path, torchvision_root, rho=0.95)
    merged = public.merge(oracle, on="example_id")
    patched_row = merged[merged["has_patch"]].iloc[0]

    from PIL import Image

    base = Image.open(tmp_path / "images" / patched_row["image_ref"]).convert("RGB")
    spec = PatchSpec(size_px=32, color=(255, 0, 255), alpha=1.0)
    rendered = add_patch(base, spec, patched_row["example_id"], 0)
    arr = np.array(rendered)
    magenta_pixels = np.sum((arr[:, :, 0] > 200) & (arr[:, :, 1] < 50) & (arr[:, :, 2] > 200))
    assert magenta_pixels == 32 * 32
