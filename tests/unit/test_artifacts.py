"""Tests for shortcut_lens.artifacts: LocalStore path helpers."""

from __future__ import annotations

from pathlib import Path

from shortcut_lens.artifacts import LocalStore, build_dir, cache_dir, run_dir


def test_local_store_path_resolves_under_root(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert store.path("runs", "r0") == tmp_path / "runs" / "r0"


def test_local_store_exists(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert not store.exists("runs", "r0")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "r0").touch()
    assert store.exists("runs", "r0")


def test_local_store_ensure_dir_creates_nested_directories(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    created = store.ensure_dir("runs", "r0", "predictions")
    assert created.is_dir()
    assert created == tmp_path / "runs" / "r0" / "predictions"


def test_build_dir_layout(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    expected = tmp_path / "builds" / "planted_pets" / "abcd1234"
    assert build_dir(store, "planted_pets", "abcd1234") == expected


def test_cache_dir_layout(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    assert cache_dir(store, "clip", "render5678") == tmp_path / "cache" / "clip" / "render5678"


def test_run_dir_layout(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    run_id = "e2-planted_pets-abcd1234-s0"
    assert run_dir(store, run_id) == tmp_path / "runs" / run_id
