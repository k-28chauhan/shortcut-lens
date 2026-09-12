"""Planted CLIP embedding cache, keyed by (image, patch spec, has_patch).

Concept: CLIP embeddings for planted data only need recomputing when the render (patch size,
colour, position) changes, not for every rho -- caching by render spec lets the rho sweep (E3)
reuse embeddings across its 5 rho values (FR-E4). `RenderKey` deliberately duplicates the handful
of fields it needs from `build.planting.PatchSpec` as plain data rather than importing it: this
module is label-free zone and `build` is oracle zone (D-003) -- the composition root (`cli.py`)
is the one place that reads a real `PatchSpec` and turns it into a `RenderKey`.

Pipeline position: label-free zone. Used by `embeddings/clip_space.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shortcut_lens.artifacts import ArtifactStore, cache_dir
from shortcut_lens.config import StrictBaseModel, short_hash
from shortcut_lens.embeddings.store import read_embeddings, write_embeddings


class RenderKey(StrictBaseModel):
    """Everything that changes what a patched image actually looks like, as plain data."""

    base_image_ref: str
    has_patch: bool
    patch_size_px: int | None = None
    patch_color: tuple[int, int, int] | None = None
    patch_alpha: float | None = None
    build_seed: int = 0


def render_hash(key: RenderKey) -> str:
    """Short hash of a `RenderKey`, used as the cache directory name (ARCHITECTURE §5)."""
    return short_hash(key.model_dump())


def clip_cache_dir(store: ArtifactStore, keys: list[RenderKey]) -> Path:
    """Cache directory for a split, keyed by the *set* of render keys it contains.

    One directory per split's full set of renders (not one per image) -- `render_hash` of the
    sorted list of per-image keys, so changing any single image's patch (or adding/removing images)
    changes the directory, while an unchanged set of renders always resolves to the same one.
    """
    combined_hash = short_hash({"keys": sorted(render_hash(key) for key in keys)})
    return cache_dir(store, "clip", combined_hash)


def load_cached(embeddings_dir: Path, split: str) -> tuple[np.ndarray, list[str]] | None:
    """Return `(vectors, example_ids)` if `split` is already cached under `embeddings_dir`, else
    `None` -- a cache miss, not an error."""
    npy_path, ids_path = embeddings_dir / f"{split}.npy", embeddings_dir / f"{split}.ids.parquet"
    if not (npy_path.exists() and ids_path.exists()):
        return None
    return read_embeddings(embeddings_dir, split)


def save_to_cache(
    embeddings_dir: Path, split: str, vectors: np.ndarray, example_ids: list[str]
) -> None:
    write_embeddings(embeddings_dir, split, vectors, example_ids)
