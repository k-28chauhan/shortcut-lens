"""Tests for shortcut_lens.embeddings.cache: render-key hashing, cache hit/miss."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shortcut_lens.artifacts import LocalStore
from shortcut_lens.embeddings.cache import RenderKey, clip_cache_dir, load_cached, save_to_cache


def test_render_hash_is_deterministic_and_sensitive_to_patch_size() -> None:
    base = RenderKey(base_image_ref="a.jpg", has_patch=True, patch_size_px=32)
    same = RenderKey(base_image_ref="a.jpg", has_patch=True, patch_size_px=32)
    different_size = RenderKey(base_image_ref="a.jpg", has_patch=True, patch_size_px=16)

    from shortcut_lens.embeddings.cache import render_hash

    assert render_hash(base) == render_hash(same)
    assert render_hash(base) != render_hash(different_size)


def test_clip_cache_dir_stable_across_rho_for_same_render_keys(tmp_path: Path) -> None:
    """Two runs with different rho but the same underlying set of renders share a cache dir
    (FR-E4: the rho sweep reuses embeddings across its 5 rho values)."""
    store = LocalStore(root=tmp_path)
    keys_run_a = [
        RenderKey(base_image_ref="a.jpg", has_patch=True, patch_size_px=32),
        RenderKey(base_image_ref="b.jpg", has_patch=False),
    ]
    keys_run_b = [
        RenderKey(base_image_ref="b.jpg", has_patch=False),
        RenderKey(base_image_ref="a.jpg", has_patch=True, patch_size_px=32),
    ]
    assert clip_cache_dir(store, keys_run_a) == clip_cache_dir(store, keys_run_b)


def test_load_cached_is_none_on_miss_then_hit_after_save(tmp_path: Path) -> None:
    assert load_cached(tmp_path, "val_a") is None

    vectors = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
    ids = ["a", "b", "c"]
    save_to_cache(tmp_path, "val_a", vectors, ids)

    cached = load_cached(tmp_path, "val_a")
    assert cached is not None
    cached_vectors, cached_ids = cached
    assert cached_ids == ids
    np.testing.assert_allclose(cached_vectors, vectors, atol=1e-3)
