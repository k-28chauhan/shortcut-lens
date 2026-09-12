"""Shortcut patch planting: `PatchSpec`, `add_patch`, `null_patch`, deterministic positions.

Concept: a "planted shortcut" is a synthetic spurious feature (a coloured square patch) painted
onto an image with controlled correlation to the label (rho) and controlled visibility (patch
size). Because we choose rho and the patch ourselves, we know the ground-truth answer a
discovery tool should find (see docs/ABSTRACT.md). Patch position is deterministic per image,
derived from a hash of (example_id, build_seed) via `shortcut_lens.seeding.make_rng`. `null_patch`
paints a grey square of the same size and position, used as an occlusion control (D-008): since
both patches share the same position draw, comparing the two isolates the effect of colour from
the effect of covering part of the image at all. Painting happens at load time on decoded images,
so JPEG compression never touches the patch; "removing" a patch is not pixel-level unpainting
(impossible once painted over at alpha=1.0) but re-rendering from the retained clean base image
(FR-D3) -- that is what makes `remove(add(x)) == x` exact rather than approximate.

Pipeline position: oracle zone. Used by `build/pets.py`, `build/waterbirds.py` and
`build/datasets.py` when rendering images.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from shortcut_lens.seeding import make_rng

_NULL_PATCH_COLOR: tuple[int, int, int] = (128, 128, 128)


@dataclass(frozen=True)
class PatchSpec:
    """One patch configuration: size, colour and alpha. Position is derived, not stored here."""

    size_px: int
    color: tuple[int, int, int]
    alpha: float = 1.0
    position: str = "random"

    def __post_init__(self) -> None:
        if self.position != "random":
            raise NotImplementedError(f"PatchSpec.position={self.position!r} is not supported yet")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.size_px <= 0:
            raise ValueError(f"size_px must be positive, got {self.size_px}")


def patch_position(
    example_id: str, build_seed: int, canvas_size: int, size_px: int
) -> tuple[int, int]:
    """Deterministic top-left (x, y) for a `size_px`-square patch fully inside a square canvas.

    Deterministic in `(build_seed, example_id)` only -- not in the patch's colour or alpha -- so
    `add_patch` and `null_patch` always agree on where the patch goes (D-008).
    """
    max_offset = canvas_size - size_px
    if max_offset < 0:
        raise ValueError(f"patch size {size_px} does not fit in a {canvas_size}px canvas")
    rng = make_rng(build_seed, example_id)
    x = int(rng.integers(0, max_offset + 1))
    y = int(rng.integers(0, max_offset + 1))
    return x, y


def add_patch(image: Image.Image, spec: PatchSpec, example_id: str, build_seed: int) -> Image.Image:
    """Return a copy of `image` with `spec`'s coloured patch painted at a deterministic position."""
    result = image.convert("RGB").copy()
    x, y = patch_position(example_id, build_seed, result.width, spec.size_px)
    patch = Image.new("RGB", (spec.size_px, spec.size_px), spec.color)
    if spec.alpha >= 1.0:
        result.paste(patch, (x, y))
    else:
        region = result.crop((x, y, x + spec.size_px, y + spec.size_px))
        result.paste(Image.blend(region, patch, spec.alpha), (x, y))
    return result


def null_patch(
    image: Image.Image, spec: PatchSpec, example_id: str, build_seed: int
) -> Image.Image:
    """Paint a same-size, same-position grey square instead -- the occlusion control (D-008)."""
    grey_spec = PatchSpec(size_px=spec.size_px, color=_NULL_PATCH_COLOR, alpha=spec.alpha)
    return add_patch(image, grey_spec, example_id, build_seed)


def remove_patch(clean_base: Image.Image) -> Image.Image:
    """Undo a patch by returning the untouched clean base image (FR-D3: remove = re-render).

    Takes no patched image and no spec: the patch is not undone pixel-by-pixel (impossible once
    painted at alpha=1.0), the clean base that was never patched is simply used instead.
    """
    return clean_base.convert("RGB").copy()
