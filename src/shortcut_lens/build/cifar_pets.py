"""CIFAR-10 cat vs dog with a planted patch shortcut -- the D-002/D-037 fallback testbed.

Concept: `planted_pets` (Oxford-IIIT Pet, 224px) failed gate G3 at both rho=0.95 and rho=0.99 -- an
ImageNet-pretrained ResNet-50 solves cat-vs-dog almost immediately from real features alone, so a
small patch never becomes necessary to exploit (D-037). CIFAR-10's native 32x32 resolution carries
genuinely less real-image information, so the same classifier is expected to rely on the shortcut
more once real-feature accuracy stops being close to saturated. Everything else about the
construction is identical to `planted_pets` on purpose, so the two are a controlled A/B on exactly
one variable (image information content): images are upscaled to 224x224 *before* the patch is
painted, using the same `build/planting.py` functions unchanged, so the patch stays exactly as
large and crisp as in `planted_pets` -- only the real-image content underneath changes.
`P(patch | cat) = rho`, `P(patch | dog) = 1 - rho`, matching `planted_pets` exactly (verified via
`torchvision.datasets.CIFAR10`: `classes[3] == "cat"`, `classes[5] == "dog"`, 5000/5000 in the
official train split, 1000/1000 in test).

Pipeline position: oracle zone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import CIFAR10

from shortcut_lens.build.planting import PatchSpec, add_patch
from shortcut_lens.build.splits import (
    assign_by_target_fraction,
    class_balance_by_subsampling,
    stratified_half_split,
)
from shortcut_lens.config import StrictBaseModel

IMAGE_SIZE = 224
_CIFAR_CAT_INDEX = 3  # verified: CIFAR10(...).classes[3] == "cat"
_CIFAR_DOG_INDEX = 5  # verified: CIFAR10(...).classes[5] == "dog"
_CLASS_NAMES = ("cat", "dog")
_PATCH_LABELS = ("no_patch", "patch")


class CifarPetsConfig(StrictBaseModel):
    """See `build/pets.py`'s `PlantedPetsConfig` -- identical fields and meanings (D-002/D-037)."""

    rho: float
    patch_size_px: int = 32
    patch_color: tuple[int, int, int] = (255, 0, 255)
    patch_alpha: float = 1.0
    val_mode: Literal["balanced", "realistic"] = "balanced"
    build_seed: int = 0

    @property
    def patch_spec(self) -> PatchSpec:
        return PatchSpec(size_px=self.patch_size_px, color=self.patch_color, alpha=self.patch_alpha)


def _upscale(image: Image.Image) -> Image.Image:
    """32x32 -> 224x224. Exact resize (source is already square, so aspect ratio is a non-issue)
    -- the patch is painted afterwards, at 224 resolution, so it stays crisp (D-037's plan)."""
    resized = TF.resize(image.convert("RGB"), [IMAGE_SIZE, IMAGE_SIZE])
    assert isinstance(resized, Image.Image)
    return resized


def download_and_cache_images(root: Path, image_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    """Download CIFAR-10 (if needed), cache every cat/dog image upscaled to 224px, once.

    Returns `(image_ref_by_id, y_by_id)`. `example_id`s are `f"{split}-{index}"` against
    torchvision's own (stable, deterministic) indexing -- CIFAR-10 has no natural per-image id.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    image_ref_by_id: dict[str, str] = {}
    y_by_id: dict[str, int] = {}

    for split, is_train in (("train", True), ("test", False)):
        dataset = CIFAR10(root=str(root), train=is_train, download=True)
        targets = np.asarray(dataset.targets)
        cat_dog_indices = np.flatnonzero(np.isin(targets, [_CIFAR_CAT_INDEX, _CIFAR_DOG_INDEX]))

        for index in cat_dog_indices:
            example_id = f"{split}-{index}"
            filename = f"{example_id}.jpg"
            cached_path = image_dir / filename
            if not cached_path.exists():
                image, _ = dataset[int(index)]
                _upscale(image).save(cached_path, quality=95)
            image_ref_by_id[example_id] = filename
            y_by_id[example_id] = 0 if targets[index] == _CIFAR_CAT_INDEX else 1

    return image_ref_by_id, y_by_id


def build_cifar_pets(
    config: CifarPetsConfig, torchvision_root: Path, image_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build `(public_table, oracle_table)` for planted_cifar_pets -- same shape as
    `build_planted_pets` (D-002/D-037: identical construction, different base images)."""
    image_ref_by_id, y_by_id = download_and_cache_images(torchvision_root, image_dir)
    all_ids = list(image_ref_by_id)

    train_ids_raw = [eid for eid in all_ids if eid.startswith("train-")]
    test_pool_ids_raw = [eid for eid in all_ids if eid.startswith("test-")]

    train_ids = class_balance_by_subsampling(
        train_ids_raw, [y_by_id[eid] for eid in train_ids_raw], config.build_seed
    )
    test_pool_ids = class_balance_by_subsampling(
        test_pool_ids_raw, [y_by_id[eid] for eid in test_pool_ids_raw], config.build_seed
    )

    val_ids, test_ids = stratified_half_split(
        test_pool_ids, [y_by_id[eid] for eid in test_pool_ids], config.build_seed
    )
    val_a_ids, val_b_ids = stratified_half_split(
        val_ids, [y_by_id[eid] for eid in val_ids], config.build_seed
    )

    def _patch_flags_for(ids: list[str], target_rho: float, tag: str) -> dict[str, bool]:
        flags: dict[str, bool] = {}
        for class_name, y in zip(_CLASS_NAMES, (0, 1), strict=True):
            class_ids = [eid for eid in ids if y_by_id[eid] == y]
            fraction = target_rho if class_name == "cat" else 1.0 - target_rho
            present = assign_by_target_fraction(
                class_ids, fraction, config.build_seed, tag, class_name
            )
            flags.update({eid: eid in present for eid in class_ids})
        return flags

    train_patch = _patch_flags_for(train_ids, config.rho, "train")
    test_patch = _patch_flags_for(test_ids, 0.5, "test")
    val_target_rho = config.rho if config.val_mode == "realistic" else 0.5
    val_patch = _patch_flags_for(val_ids, val_target_rho, "val")

    minority_patch_by_class: dict[int, bool] = {}
    for y in (0, 1):
        cls_ids = [eid for eid in train_ids if y_by_id[eid] == y]
        patch_fraction = sum(train_patch[eid] for eid in cls_ids) / len(cls_ids)
        minority_patch_by_class[y] = patch_fraction < 0.5

    public_records: list[dict[str, object]] = []
    oracle_records: list[dict[str, object]] = []

    def _add_rows(ids: list[str], patch_of: dict[str, bool], split: str) -> None:
        for eid in ids:
            y = y_by_id[eid]
            class_name = _CLASS_NAMES[y]
            has_patch = patch_of[eid]
            public_records.append(
                {
                    "example_id": eid,
                    "dataset": "planted_cifar_pets",
                    "split": split,
                    "y": y,
                    "class_name": class_name,
                    "image_ref": image_ref_by_id[eid],
                }
            )
            attribute = int(has_patch)
            patch_label = _PATCH_LABELS[attribute]
            oracle_records.append(
                {
                    "example_id": eid,
                    "attribute": attribute,
                    "group": 2 * y + attribute,
                    "group_name": f"{class_name}|{patch_label}",
                    "is_minority": has_patch == minority_patch_by_class[y],
                    "has_patch": has_patch,
                    "patch_size": config.patch_size_px if has_patch else None,
                }
            )

    _add_rows(train_ids, train_patch, "train")
    _add_rows(val_a_ids, val_patch, "val_a")
    _add_rows(val_b_ids, val_patch, "val_b")
    _add_rows(test_ids, test_patch, "test")

    return pd.DataFrame.from_records(public_records), pd.DataFrame.from_records(oracle_records)


def render_cifar_pets_image(
    example_id: str,
    image_dir: Path,
    image_ref: str,
    has_patch: bool,
    spec: PatchSpec,
    build_seed: int,
) -> Image.Image:
    """Load a cached, upscaled CIFAR-10 image and paint the patch on if `has_patch`."""
    image = Image.open(image_dir / image_ref).convert("RGB")
    if has_patch:
        image = add_patch(image, spec, example_id, build_seed)
    return image
