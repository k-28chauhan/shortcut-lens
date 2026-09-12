"""Tests for shortcut_lens.verification.reliance: recovers a known R_net on synthetic data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from PIL import Image

from shortcut_lens.build.planting import PatchSpec, add_patch, null_patch
from shortcut_lens.data.transforms import eval_transform
from shortcut_lens.verification.reliance import compute_reliance

_CANVAS = 8
_PATCH_SIZE = 4
_BUILD_SEED = 0


class _ThresholdModel:
    """A hand-built, non-learned model: predicts "dog" iff whole-image brightness <= `threshold`.

    `threshold` is set (in the test) strictly between the null(grey)-patch brightness and the
    real(white)-patch brightness, so this model's reliance on the patch is exactly knowable: it
    reacts to the patch's *colour*, not to the mere presence of a same-sized painted region.
    """

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def eval(self) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        brightness = x.mean(dim=(1, 2, 3))
        logit1 = torch.where(brightness <= self._threshold, torch.tensor(10.0), torch.tensor(-10.0))
        return torch.stack([-logit1, logit1], dim=1)


def _make_black_image(image_root: Path) -> None:
    image_root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (_CANVAS, _CANVAS), (0, 0, 0)).save(image_root / "base.jpg")


def _joined_table(n_dogs: int, n_cats: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i in range(n_dogs):
        rows.append(
            {
                "example_id": f"dog-{i}",
                "image_ref": "base.jpg",
                "y": 1,
                "class_name": "dog",
                "has_patch": False,
                "correct": True,
            }
        )
    for i in range(n_cats):
        rows.append(
            {
                "example_id": f"cat-{i}",
                "image_ref": "base.jpg",
                "y": 0,
                "class_name": "cat",
                "has_patch": True,
                "correct": True,
            }
        )
    return pd.DataFrame(rows)


def test_compute_reliance_recovers_known_r_net(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _make_black_image(image_root)
    spec = PatchSpec(size_px=_PATCH_SIZE, color=(255, 255, 255))
    black = Image.new("RGB", (_CANVAS, _CANVAS), (0, 0, 0))

    transform = eval_transform()
    grey_brightness = transform(null_patch(black, spec, "ref", _BUILD_SEED)).mean().item()
    white_brightness = transform(add_patch(black, spec, "ref", _BUILD_SEED)).mean().item()
    threshold = (grey_brightness + white_brightness) / 2
    model = _ThresholdModel(threshold)

    joined = _joined_table(n_dogs=5, n_cats=5)
    result = compute_reliance(
        model, "cpu", joined, image_root, spec, _BUILD_SEED, seed=0, batch_size=8, n_resamples=200
    )

    def _rate(direction: str, intervention: str) -> float:
        row = result[(result["direction"] == direction) & (result["intervention"] == intervention)]
        return float(row["break_rate"].iloc[0])

    # add direction: real (white) always breaks dogs' baseline; null (grey) never does
    assert _rate("add", "real") == pytest.approx(1.0)
    assert _rate("add", "null") == pytest.approx(0.0)
    # remove direction: both dropping to clean and swapping to null land below threshold too
    assert _rate("remove", "real") == pytest.approx(1.0)
    assert _rate("remove", "null") == pytest.approx(1.0)

    r_net_row = result[result["direction"] == "net"].iloc[0]
    assert r_net_row["break_rate"] == pytest.approx(0.5)
    assert r_net_row["lo"] <= r_net_row["break_rate"] <= r_net_row["hi"]


def test_compute_reliance_raises_when_a_group_is_empty(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    _make_black_image(image_root)
    spec = PatchSpec(size_px=_PATCH_SIZE, color=(255, 255, 255))
    joined = _joined_table(n_dogs=3, n_cats=0)  # no cats-with-patch at all

    with pytest.raises(ValueError, match="cats with a patch"):
        compute_reliance(_ThresholdModel(0.0), "cpu", joined, image_root, spec, _BUILD_SEED, seed=0)
