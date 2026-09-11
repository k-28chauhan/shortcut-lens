"""Shared pytest fixtures. See docs/TESTING.md §2 for the full fixture list as it grows."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> Path:
    """A temporary artefact root, isolated per test."""
    root = tmp_path / "artifacts"
    root.mkdir()
    return root
