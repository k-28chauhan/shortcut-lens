"""Oxford-IIIT Pet (cat vs dog) with a planted patch shortcut -- the main ground-truth testbed.

Concept: natural images at real resolution let CLIP and BLIP actually see the planted patch,
unlike small synthetic images. `P(patch | cat) = rho`, `P(patch | dog) = 1 - rho`, so the
minority (failure) groups are *cat without patch* and *dog with patch* (see D-001, docs/PRD.md §7).
Train is class-balanced by subsampling the larger class; val/test come from the official `test`
split (also class-balanced, then split 50/50). Test is always patch-balanced 50/50 per class;
val is either 50/50 (`balanced`) or matches train's rho (`realistic`) -- both are direct
attribute-assignment decisions (`assign_by_target_fraction`), not example-dropping, because every
image already exists and only the patch on/off decision changes (see `build/splits.py`).

`torchvision.datasets.OxfordIIITPet(target_types="binary-category")` is used only to trigger the
download; labels are parsed from its own raw `annotations/<split>.txt` files directly (format
`<example_id> <category> <binary_class 1=cat,2=dog> <breed_id>`), verified against torchvision's
own parsing logic (`bin_classes = ["Cat", "Dog"]`, so `binary_class - 1` gives 0=cat, 1=dog) rather
than relying on the dataset object's internal (non-public) attributes.

Pipeline position: oracle zone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.datasets import OxfordIIITPet

from shortcut_lens.build.planting import PatchSpec, add_patch
from shortcut_lens.build.splits import (
    assign_by_target_fraction,
    class_balance_by_subsampling,
    stratified_half_split,
)
from shortcut_lens.config import StrictBaseModel

IMAGE_SIZE = 224
_RESIZE_SHORTER_SIDE = round(IMAGE_SIZE * 256 / 224)  # 256, PRD §7's "resize shorter side 256"
_CLASS_NAMES = ("cat", "dog")  # binary_class - 1: 0=cat, 1=dog (verified from torchvision source)
_PATCH_LABELS = ("no_patch", "patch")


class PlantedPetsConfig(StrictBaseModel):
    """See docs/PRD.md §7 and docs/ARCHITECTURE.md §6 for the field meanings."""

    rho: float
    patch_size_px: int = 32
    patch_color: tuple[int, int, int] = (255, 0, 255)
    patch_alpha: float = 1.0
    val_mode: Literal["balanced", "realistic"] = "balanced"
    build_seed: int = 0

    @property
    def patch_spec(self) -> PatchSpec:
        return PatchSpec(size_px=self.patch_size_px, color=self.patch_color, alpha=self.patch_alpha)


def _preprocess(image: Image.Image) -> Image.Image:
    """Resize the shorter side to 256px, then centre-crop to 224x224 (docs/PRD.md §7)."""
    resized = TF.resize(image.convert("RGB"), _RESIZE_SHORTER_SIDE)
    cropped = TF.center_crop(resized, IMAGE_SIZE)
    assert isinstance(cropped, Image.Image)
    return cropped


def _read_annotations(root: Path, split: Literal["trainval", "test"]) -> list[tuple[str, int]]:
    """Parse `annotations/<split>.txt`: one `(example_id, y)` pair per line, y in {0=cat, 1=dog}."""
    path = root / "oxford-iiit-pet" / "annotations" / f"{split}.txt"
    rows: list[tuple[str, int]] = []
    for line in path.read_text().splitlines():
        example_id, _category, binary_class, _breed = line.split()
        rows.append((example_id, int(binary_class) - 1))
    return rows


def download_and_cache_images(root: Path, image_dir: Path) -> dict[str, str]:
    """Download (if needed) and preprocess every Oxford-IIIT Pet image once; return id -> filename.

    `root` is the torchvision download root; `image_dir` is the 224px JPEG cache this project
    controls. Skips preprocessing an id that is already cached.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    image_ref_by_id: dict[str, str] = {}

    for split in ("trainval", "test"):
        OxfordIIITPet(root=str(root), split=split, target_types="binary-category", download=True)
        images_dir = root / "oxford-iiit-pet" / "images"
        for example_id, _y in _read_annotations(root, split):
            filename = f"{example_id}.jpg"
            cached_path = image_dir / filename
            if not cached_path.exists():
                image = Image.open(images_dir / f"{example_id}.jpg")
                _preprocess(image).save(cached_path, quality=95)
            image_ref_by_id[example_id] = filename
    return image_ref_by_id


def build_planted_pets(
    config: PlantedPetsConfig, torchvision_root: Path, image_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build `(public_table, oracle_table)` for planted_pets: caches images, assigns patches."""
    image_ref_by_id = download_and_cache_images(torchvision_root, image_dir)

    train_pairs = _read_annotations(torchvision_root, "trainval")
    test_pool_pairs = _read_annotations(torchvision_root, "test")

    train_ids = class_balance_by_subsampling(
        [eid for eid, _ in train_pairs], [y for _, y in train_pairs], config.build_seed
    )
    train_y = {eid: y for eid, y in train_pairs}

    test_pool_ids = class_balance_by_subsampling(
        [eid for eid, _ in test_pool_pairs], [y for _, y in test_pool_pairs], config.build_seed
    )
    test_pool_y = {eid: y for eid, y in test_pool_pairs}

    val_ids, test_ids = stratified_half_split(
        test_pool_ids, [test_pool_y[eid] for eid in test_pool_ids], config.build_seed
    )
    val_a_ids, val_b_ids = stratified_half_split(
        val_ids, [test_pool_y[eid] for eid in val_ids], config.build_seed
    )

    def _patch_flags_for(
        ids: list[str], y_of: dict[str, int], target_rho: float, tag: str
    ) -> dict[str, bool]:
        flags: dict[str, bool] = {}
        for class_name, y in zip(_CLASS_NAMES, (0, 1), strict=True):
            class_ids = [eid for eid in ids if y_of[eid] == y]
            fraction = target_rho if class_name == "cat" else 1.0 - target_rho
            present = assign_by_target_fraction(
                class_ids, fraction, config.build_seed, tag, class_name
            )
            flags.update({eid: eid in present for eid in class_ids})
        return flags

    train_patch = _patch_flags_for(train_ids, train_y, config.rho, "train")
    test_patch = _patch_flags_for(test_ids, test_pool_y, 0.5, "test")
    val_target_rho = config.rho if config.val_mode == "realistic" else 0.5
    val_patch = _patch_flags_for(val_ids, test_pool_y, val_target_rho, "val")

    # minority attribute value per class, from realised TRAINING proportions (ARCHITECTURE §4)
    minority_patch_by_class: dict[int, bool] = {}
    for y in (0, 1):
        cls_ids = [eid for eid in train_ids if train_y[eid] == y]
        patch_fraction = sum(train_patch[eid] for eid in cls_ids) / len(cls_ids)
        minority_patch_by_class[y] = patch_fraction < 0.5

    public_records: list[dict[str, object]] = []
    oracle_records: list[dict[str, object]] = []

    def _add_rows(
        ids: list[str], y_of: dict[str, int], patch_of: dict[str, bool], split: str
    ) -> None:
        for eid in ids:
            y = y_of[eid]
            class_name = _CLASS_NAMES[y]
            has_patch = patch_of[eid]
            public_records.append(
                {
                    "example_id": eid,
                    "dataset": "planted_pets",
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

    _add_rows(train_ids, train_y, train_patch, "train")
    _add_rows(val_a_ids, test_pool_y, val_patch, "val_a")
    _add_rows(val_b_ids, test_pool_y, val_patch, "val_b")
    _add_rows(test_ids, test_pool_y, test_patch, "test")

    return pd.DataFrame.from_records(public_records), pd.DataFrame.from_records(oracle_records)


def render_pets_image(
    example_id: str,
    image_dir: Path,
    image_ref: str,
    has_patch: bool,
    spec: PatchSpec,
    build_seed: int,
) -> Image.Image:
    """Load a cached, preprocessed pet image and paint the patch on if `has_patch`."""
    image = Image.open(image_dir / image_ref).convert("RGB")
    if has_patch:
        image = add_patch(image, spec, example_id, build_seed)
    return image
