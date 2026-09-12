"""Tests for shortcut_lens.build.planting: exact remove(add(x)) == x, positions, null patch."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from shortcut_lens.build.planting import (
    PatchSpec,
    add_patch,
    null_patch,
    patch_position,
    remove_patch,
)

CANVAS = 224


def _blank_image() -> Image.Image:
    return Image.new("RGB", (CANVAS, CANVAS), (10, 20, 30))


def test_remove_add_is_exact() -> None:
    base = _blank_image()
    spec = PatchSpec(size_px=32, color=(255, 0, 255))

    patched = add_patch(base, spec, "example-1", build_seed=0)
    restored = remove_patch(base)

    np.testing.assert_array_equal(np.array(restored), np.array(base))
    # sanity: the patch actually changed something, so this isn't a vacuous test
    assert not np.array_equal(np.array(patched), np.array(base))


def test_patch_position_in_bounds() -> None:
    for i in range(50):
        x, y = patch_position(f"example-{i}", build_seed=0, canvas_size=CANVAS, size_px=32)
        assert 0 <= x <= CANVAS - 32
        assert 0 <= y <= CANVAS - 32


def test_patch_position_deterministic() -> None:
    a = patch_position("example-1", build_seed=0, canvas_size=CANVAS, size_px=32)
    b = patch_position("example-1", build_seed=0, canvas_size=CANVAS, size_px=32)
    assert a == b


def test_patch_position_varies_by_example_id() -> None:
    positions = {patch_position(f"example-{i}", 0, CANVAS, 32) for i in range(20)}
    assert len(positions) > 1


def test_patch_position_rejects_oversized_patch() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        patch_position("example-1", 0, canvas_size=32, size_px=64)


def test_null_patch_same_position_different_colour() -> None:
    base = _blank_image()
    spec = PatchSpec(size_px=32, color=(255, 0, 255))

    real = add_patch(base, spec, "example-1", build_seed=0)
    control = null_patch(base, spec, "example-1", build_seed=0)

    real_arr = np.array(real)
    control_arr = np.array(control)
    base_arr = np.array(base)

    real_changed = np.any(real_arr != base_arr, axis=-1)
    control_changed = np.any(control_arr != base_arr, axis=-1)

    # same footprint (same position, same size)
    assert np.array_equal(real_changed, control_changed)
    assert real_changed.sum() == 32 * 32
    # different colour where changed
    assert not np.array_equal(real_arr[real_changed], control_arr[control_changed])


def test_add_patch_paints_exact_colour_at_full_alpha() -> None:
    base = _blank_image()
    spec = PatchSpec(size_px=32, color=(255, 0, 255), alpha=1.0)
    patched = add_patch(base, spec, "example-1", build_seed=0)

    x, y = patch_position("example-1", 0, CANVAS, 32)
    pixel = patched.getpixel((x + 16, y + 16))
    assert pixel == (255, 0, 255)


def test_horizontal_flip_preserves_patch_pixel_count() -> None:
    """D-flip note (ARCHITECTURE §8): a flip is a bijection, so pixel count is unchanged."""
    base = _blank_image()
    spec = PatchSpec(size_px=32, color=(255, 0, 255))
    patched = add_patch(base, spec, "example-1", build_seed=0)

    flipped = patched.transpose(Image.FLIP_LEFT_RIGHT)

    def count_patch_pixels(image: Image.Image) -> int:
        arr = np.array(image)
        return int(np.all(arr == np.array(spec.color), axis=-1).sum())

    assert count_patch_pixels(patched) == count_patch_pixels(flipped) == 32 * 32


def test_patch_spec_rejects_unsupported_position() -> None:
    with pytest.raises(NotImplementedError):
        PatchSpec(size_px=32, color=(255, 0, 255), position="fixed")


def test_patch_spec_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        PatchSpec(size_px=32, color=(255, 0, 255), alpha=1.5)
