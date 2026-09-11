"""Tests for shortcut_lens.manifest: round-trip, and validate() catching tampering."""

from __future__ import annotations

from pathlib import Path

from shortcut_lens.config import config_hash
from shortcut_lens.manifest import (
    Manifest,
    new_manifest,
    read_manifest,
    validate_manifest,
    write_manifest,
)


def _make_manifest(**overrides: object) -> Manifest:
    manifest = new_manifest(
        run_id="e0-synthetic_shapes-abcd1234-s0",
        stage="build",
        config={"dataset": "synthetic_shapes", "rho": 0.95},
        seed=0,
        inputs={},
        duration_s=1.23,
    )
    if overrides:
        manifest = manifest.model_copy(update=overrides)
    return manifest


def test_new_manifest_fills_environment_fields() -> None:
    manifest = _make_manifest()
    assert manifest.config_hash == config_hash({"dataset": "synthetic_shapes", "rho": 0.95})
    assert len(manifest.git_commit) == 40
    assert manifest.python.startswith("3.11")
    assert "torch" in manifest.packages
    assert manifest.packages["torch"]


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = _make_manifest()
    path = tmp_path / "manifest.json"

    write_manifest(path, manifest)
    loaded = read_manifest(path)

    assert loaded == manifest


def test_manifest_write_creates_parent_directories(tmp_path: Path) -> None:
    manifest = _make_manifest()
    path = tmp_path / "runs" / "e0-run" / "manifest.json"

    write_manifest(path, manifest)

    assert path.exists()


def test_validate_manifest_passes_for_untampered_manifest(tmp_path: Path) -> None:
    manifest = _make_manifest(inputs={"build": "public.parquet"})
    (tmp_path / "public.parquet").write_text("x")

    problems = validate_manifest(manifest, run_dir=tmp_path)

    assert problems == []


def test_validate_manifest_detects_tampered_config_hash(tmp_path: Path) -> None:
    manifest = _make_manifest(config_hash="0" * 64)

    problems = validate_manifest(manifest, run_dir=tmp_path)

    assert any("config_hash mismatch" in p for p in problems)


def test_validate_manifest_detects_unexpected_config_hash(tmp_path: Path) -> None:
    manifest = _make_manifest()

    problems = validate_manifest(manifest, run_dir=tmp_path, expected_config_hash="0" * 64)

    assert any("does not match expected" in p for p in problems)


def test_validate_manifest_detects_malformed_git_commit(tmp_path: Path) -> None:
    manifest = _make_manifest(git_commit="not-a-real-sha")

    problems = validate_manifest(manifest, run_dir=tmp_path)

    assert any("not a well-formed sha" in p for p in problems)


def test_validate_manifest_detects_missing_input_file(tmp_path: Path) -> None:
    manifest = _make_manifest(inputs={"build": "public.parquet"})

    problems = validate_manifest(manifest, run_dir=tmp_path)

    assert any("not found" in p for p in problems)
