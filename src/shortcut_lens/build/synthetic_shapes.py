"""Procedurally generated circle-vs-square dataset with a planted shortcut, for CPU CI.

Concept: the smallest possible testbed for the whole pipeline. Images are generated (not
downloaded), so `make smoke` and CI can run the full discover-confirm-name-verify-mitigate loop
in minutes on a laptop with no GPU and no network. Two attribute variants: a small red `dot`
(painted in a fixed corner) or a `background` colour tint, either plantable with a controlled
`rho`. Class assignment mirrors PlantedPets exactly: `P(attribute | circle) = rho`,
`P(attribute | square) = 1 - rho`, so at high rho the minority (failure) groups are *circle
without the attribute* and *square with the attribute* -- matching the frozen group names in
`vocab/eval_keywords.yaml` for the `dot` variant (`square|dot`, `circle|no_dot`). Attribute
presence per class is assigned by exact target count (rounded), not independent coin flips, so
the realised proportion never drifts by more than one image from `rho` -- see docs/PLAN.md M1's
"planted proportions within ±1 image of target per split" test.

Pipeline position: oracle zone. Feeds `build/tables.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from shortcut_lens.build.splits import assign_by_target_fraction, stratified_half_split
from shortcut_lens.config import StrictBaseModel
from shortcut_lens.seeding import make_rng

CANVAS_PX = 32
CLASS_NAMES = ("circle", "square")

_SHAPE_COLOR = (30, 30, 30)
_BACKGROUND_COLOR = (235, 235, 235)
_TINT_COLOR = (180, 205, 235)
_DOT_COLOR = (215, 25, 25)
_DOT_RADIUS = 3
_SHAPE_MARGIN = 6


class SyntheticShapesConfig(StrictBaseModel):
    """See docs/PRD.md §7 and docs/ARCHITECTURE.md §6 for the field meanings."""

    variant: Literal["dot", "background"]
    rho: float
    build_seed: int = 0
    n_train: int = 400
    n_val: int = 400
    n_test: int = 400


@dataclass(frozen=True)
class _Row:
    example_id: str
    split: str
    y: int
    class_name: str
    attribute_present: bool


def _draw_shape(canvas: Image.Image, class_name: str, rng: np.random.Generator) -> None:
    draw = ImageDraw.Draw(canvas)
    jitter = int(rng.integers(-2, 3))
    size = CANVAS_PX - 2 * _SHAPE_MARGIN + jitter
    x0 = (CANVAS_PX - size) // 2
    y0 = (CANVAS_PX - size) // 2
    x1, y1 = x0 + size, y0 + size
    if class_name == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=_SHAPE_COLOR)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=_SHAPE_COLOR)


def _draw_dot(canvas: Image.Image) -> None:
    cx, cy = CANVAS_PX - 6, 6
    draw = ImageDraw.Draw(canvas)
    draw.ellipse(
        [cx - _DOT_RADIUS, cy - _DOT_RADIUS, cx + _DOT_RADIUS, cy + _DOT_RADIUS], fill=_DOT_COLOR
    )


def render_shape(
    example_id: str,
    class_name: str,
    attribute_present: bool,
    variant: Literal["dot", "background"],
    build_seed: int,
) -> Image.Image:
    """Render one synthetic_shapes image. Deterministic in `(build_seed, example_id)`."""
    if class_name not in CLASS_NAMES:
        raise ValueError(f"class_name must be one of {CLASS_NAMES}, got {class_name!r}")
    rng = make_rng(build_seed, example_id)
    tinted = variant == "background" and attribute_present
    background = _TINT_COLOR if tinted else _BACKGROUND_COLOR
    canvas = Image.new("RGB", (CANVAS_PX, CANVAS_PX), background)
    _draw_shape(canvas, class_name, rng)
    if variant == "dot" and attribute_present:
        _draw_dot(canvas)
    return canvas


def _make_split_rows(split: str, n: int, rho: float, build_seed: int) -> list[_Row]:
    """Assign `n` examples to circle/square (balanced) and attribute presence (exact count)."""
    n_circle = n // 2
    ids = [f"synthetic_shapes-{split}-{i:05d}" for i in range(n)]
    class_groups = (("circle", ids[:n_circle]), ("square", ids[n_circle:]))

    rows: list[_Row] = []
    for class_name, group_ids in class_groups:
        target_rho = rho if class_name == "circle" else 1.0 - rho
        present_ids = assign_by_target_fraction(
            group_ids, target_rho, build_seed, split, class_name
        )
        y = CLASS_NAMES.index(class_name)
        rows.extend(_Row(eid, split, y, class_name, eid in present_ids) for eid in group_ids)
    return rows


def _minority_attribute_present(train_rows: list[_Row]) -> dict[str, bool]:
    """Per class, whether 'attribute present' (vs 'absent') is the minority in the training data.

    Computed from realised training proportions, not assumed from rho's direction (ARCHITECTURE
    §4: minority groups come from training-set proportions, never hardcoded).
    """
    result: dict[str, bool] = {}
    for class_name in CLASS_NAMES:
        cls_rows = [r for r in train_rows if r.class_name == class_name]
        present_fraction = sum(r.attribute_present for r in cls_rows) / len(cls_rows)
        result[class_name] = present_fraction < 0.5
    return result


def build_synthetic_shapes(
    config: SyntheticShapesConfig, image_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Render every image and return `(public_table, oracle_table)` for synthetic_shapes."""
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _make_split_rows("train", config.n_train, config.rho, config.build_seed)
    val_rows = _make_split_rows("val", config.n_val, config.rho, config.build_seed)
    test_rows = _make_split_rows("test", config.n_test, config.rho, config.build_seed)

    val_a_id_list, _ = stratified_half_split(
        [r.example_id for r in val_rows], [r.y for r in val_rows], config.build_seed
    )
    val_a_ids = set(val_a_id_list)

    def _assign_val_split(row: _Row) -> _Row:
        split = "val_a" if row.example_id in val_a_ids else "val_b"
        return _Row(row.example_id, split, row.y, row.class_name, row.attribute_present)

    all_rows = list(train_rows)
    all_rows.extend(_assign_val_split(r) for r in val_rows)
    all_rows.extend(test_rows)

    minority_present = _minority_attribute_present(train_rows)
    attribute_name = "dot" if config.variant == "dot" else "tint"

    public_records: list[dict[str, object]] = []
    oracle_records: list[dict[str, object]] = []
    for row in all_rows:
        image = render_shape(
            row.example_id, row.class_name, row.attribute_present, config.variant, config.build_seed
        )
        image_ref = f"{row.example_id}.png"
        image.save(image_dir / image_ref)

        public_records.append(
            {
                "example_id": row.example_id,
                "dataset": "synthetic_shapes",
                "split": row.split,
                "y": row.y,
                "class_name": row.class_name,
                "image_ref": image_ref,
            }
        )

        attribute = int(row.attribute_present)
        attribute_label = attribute_name if row.attribute_present else f"no_{attribute_name}"
        oracle_records.append(
            {
                "example_id": row.example_id,
                "attribute": attribute,
                "group": 2 * row.y + attribute,
                "group_name": f"{row.class_name}|{attribute_label}",
                "is_minority": row.attribute_present == minority_present[row.class_name],
            }
        )

    return pd.DataFrame.from_records(public_records), pd.DataFrame.from_records(oracle_records)
