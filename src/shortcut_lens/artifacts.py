"""Artefact storage: path helpers and `LocalStore` (an `HFHubStore` backend lands in M3).

Concept: every stage reads and writes artefacts under a fixed layout (builds, cache, per-run
outputs -- see docs/ARCHITECTURE.md §5) so that later stages, `slens report`, and the human can
always find a given run's files the same way regardless of which stage produced them.
`ArtifactStore` is the seam that lets the same path logic work whether artefacts live on a laptop
disk or get pulled from Hugging Face Hub after a GPU handoff (D-017).

Pipeline position: shared, pure (path arithmetic only -- no dataset or label access). Used by
every CLI stage to decide where to read inputs from and write outputs to.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ArtifactStore(Protocol):
    """Anything that can resolve artefact-layout paths under a root. See `LocalStore`."""

    root: Path

    def path(self, *parts: str) -> Path: ...

    def exists(self, *parts: str) -> bool: ...

    def ensure_dir(self, *parts: str) -> Path: ...


class LocalStore:
    """Artefact store rooted at a local directory (default `artifacts/`, gitignored)."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)

    def path(self, *parts: str) -> Path:
        """Resolve `parts` under the store's root, e.g. `store.path("runs", run_id)`."""
        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def ensure_dir(self, *parts: str) -> Path:
        """Resolve `parts` under the root and create it (and parents) as a directory."""
        directory = self.path(*parts)
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def build_dir(store: ArtifactStore, dataset: str, build_hash: str) -> Path:
    """`artifacts/builds/<dataset>/<build_hash>/` -- public/oracle parquet + build manifest."""
    return store.path("builds", dataset, build_hash)


def cache_dir(store: ArtifactStore, space: str, render_hash: str) -> Path:
    """`artifacts/cache/<space>/<render_hash>/` -- e.g. the planted CLIP embedding cache (FR-E4)."""
    return store.path("cache", space, render_hash)


def run_dir(store: ArtifactStore, run_id: str) -> Path:
    """`artifacts/runs/<run_id>/` -- one run's manifest, checkpoints and downstream artefacts."""
    return store.path("runs", run_id)
