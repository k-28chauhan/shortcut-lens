"""Artefact storage: path helpers, `LocalStore`, and `HFHubStore` (D-017).

Concept: every stage reads and writes artefacts under a fixed layout (builds, cache, per-run
outputs -- see docs/ARCHITECTURE.md §5) so that later stages, `slens report`, and the human can
always find a given run's files the same way regardless of which stage produced them.
`ArtifactStore` is the seam that lets the same path logic work whether artefacts live on a laptop
disk (`LocalStore`) or get pushed to / pulled from a private Hugging Face Hub dataset repo after a
GPU handoff (`HFHubStore`) -- both resolve paths under a local root the same way; `HFHubStore`
additionally knows how to sync one run's folder with the Hub.

Pipeline position: shared, pure path arithmetic plus (for `HFHubStore`) network I/O confined to
`push_run`/`pull_run`. Used by every CLI stage to decide where to read inputs from and write
outputs to; `push_run`/`pull_run` are used by `slens run-jobs` and `slens pull-artifacts`.
"""

from __future__ import annotations

import os
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


class HFHubStore:
    """Artefact store rooted locally, backed by a private Hugging Face Hub dataset repo.

    Path resolution (`path`/`exists`/`ensure_dir`) is identical to `LocalStore` -- every stage
    reads/writes the same local layout regardless of backend. `push_run`/`pull_run` are the only
    methods that touch the network, and only ever move one run's folder at a time (never the
    whole `artifacts/` tree), keeping a handoff's data transfer small and explicit.
    """

    def __init__(
        self, repo_id: str, root: str | Path = "artifacts", token: str | None = None
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root)
        self._token = token or os.environ.get("HF_TOKEN")

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def exists(self, *parts: str) -> bool:
        return self.path(*parts).exists()

    def ensure_dir(self, *parts: str) -> Path:
        directory = self.path(*parts)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def push_run(self, run_id: str) -> None:
        """Upload the whole `runs/<run_id>/` folder (checkpoints included) to the Hub repo.

        Checkpoints are kept, not just predictions/embeddings, because `slens gradcam` (M8) needs
        the actual trained weights back locally -- excluding them would silently break that later
        without saving much (a ResNet checkpoint is tens of MB, not a bottleneck for HF Hub).
        """
        from huggingface_hub import HfApi

        HfApi(token=self._token).upload_folder(
            repo_id=self.repo_id,
            repo_type="dataset",
            folder_path=str(run_dir(self, run_id)),
            path_in_repo=f"runs/{run_id}",
        )

    def pull_run(self, run_id: str) -> None:
        """Download `runs/<run_id>/` from the Hub repo into this store's local root."""
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            local_dir=self.root,
            allow_patterns=[f"runs/{run_id}/*"],
            token=self._token,
        )


def build_dir(store: ArtifactStore, dataset: str, build_hash: str) -> Path:
    """`artifacts/builds/<dataset>/<build_hash>/` -- public/oracle parquet + build manifest."""
    return store.path("builds", dataset, build_hash)


def cache_dir(store: ArtifactStore, space: str, render_hash: str) -> Path:
    """`artifacts/cache/<space>/<render_hash>/` -- e.g. the planted CLIP embedding cache (FR-E4)."""
    return store.path("cache", space, render_hash)


def run_dir(store: ArtifactStore, run_id: str) -> Path:
    """`artifacts/runs/<run_id>/` -- one run's manifest, checkpoints and downstream artefacts."""
    return store.path("runs", run_id)
