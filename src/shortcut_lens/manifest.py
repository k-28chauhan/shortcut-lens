"""Manifests: write/read/validate the record that makes every reported number traceable.

Concept: "every reported number is traceable" (CLAUDE.md §3 rule 7) means each stage run writes a
manifest recording exactly what produced its outputs -- git commit, dirty flag, resolved config
and its hash, seed, input artefact ids, package versions, hardware, and duration. A tampered or
mismatched manifest (wrong config hash, a commit that does not exist, a missing input file) means
the run cannot be trusted and should be rejected, not silently used.

Pipeline position: shared, pure (writes/reads JSON only; no dataset or label access). Written by
every CLI stage in `cli.py`; read and checked by `slens validate-run` (full CLI wiring in M3).
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from shortcut_lens.config import StrictBaseModel, config_hash

_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

_TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "open_clip_torch",
    "transformers",
    "scikit-learn",
    "numpy",
)


class Manifest(StrictBaseModel):
    """One stage run's provenance record. See docs/ARCHITECTURE.md §5 for the field list."""

    run_id: str
    stage: str
    created_at: datetime
    git_commit: str
    git_dirty: bool
    config: dict[str, Any]
    config_hash: str
    seed: int
    inputs: dict[str, str]
    python: str
    packages: dict[str, str]
    hardware: dict[str, str]
    duration_s: float
    frozen_vocab_sha256: str | None = None


def _git_commit(repo_root: Path | None = None) -> str:
    """Current HEAD commit sha. Raises if not run inside a git repository."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_dirty(repo_root: Path | None = None) -> bool:
    """True if the working tree has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


def _package_versions() -> dict[str, str]:
    """Installed version of every package whose version could affect reproducibility."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _hardware_info() -> dict[str, str]:
    """Device name and CUDA version actually in use for this process."""
    if torch.cuda.is_available():
        return {"device": torch.cuda.get_device_name(0), "cuda": str(torch.version.cuda)}
    return {"device": "cpu", "cuda": "none"}


def new_manifest(
    *,
    run_id: str,
    stage: str,
    config: dict[str, Any],
    seed: int,
    inputs: dict[str, str],
    duration_s: float,
    frozen_vocab_sha256: str | None = None,
    repo_root: Path | None = None,
) -> Manifest:
    """Build a `Manifest` for the current run, filling in git/environment fields automatically.

    `config` should be the fully resolved config dict (post-composition, see `config.py`); its
    hash is computed here rather than accepted as a parameter, so the two can never drift apart.
    """
    return Manifest(
        run_id=run_id,
        stage=stage,
        created_at=datetime.now(UTC),
        git_commit=_git_commit(repo_root),
        git_dirty=_git_dirty(repo_root),
        config=config,
        config_hash=config_hash(config),
        seed=seed,
        inputs=inputs,
        python=platform.python_version(),
        packages=_package_versions(),
        hardware=_hardware_info(),
        duration_s=duration_s,
        frozen_vocab_sha256=frozen_vocab_sha256,
    )


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    """Write `manifest` as JSON to `path`, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2))


def read_manifest(path: str | Path) -> Manifest:
    """Read and validate a manifest JSON file back into a `Manifest`."""
    return Manifest.model_validate_json(Path(path).read_text())


def validate_manifest(
    manifest: Manifest,
    *,
    run_dir: Path,
    expected_config_hash: str | None = None,
) -> list[str]:
    """Return problems found with `manifest` (empty list means it passes every check).

    Checks: `config_hash` matches a fresh hash of `manifest.config` (catches a hand-edited
    manifest); `git_commit` is a well-formed sha; every input artefact id resolves to a file that
    still exists under `run_dir`. If `expected_config_hash` is given (e.g. the hash of the config
    a caller is about to reuse this run for), it must also match.
    """
    problems: list[str] = []

    recomputed = config_hash(manifest.config)
    if recomputed != manifest.config_hash:
        problems.append(
            f"config_hash mismatch: manifest says {manifest.config_hash}, recomputed {recomputed}"
        )
    if expected_config_hash is not None and manifest.config_hash != expected_config_hash:
        problems.append(
            f"config_hash {manifest.config_hash} does not match expected {expected_config_hash}"
        )
    if not _GIT_SHA_RE.fullmatch(manifest.git_commit):
        problems.append(f"git_commit is not a well-formed sha: {manifest.git_commit!r}")
    for name, artefact_id in manifest.inputs.items():
        if not (run_dir / artefact_id).exists():
            problems.append(f"input artefact {name!r} -> {artefact_id!r} not found under {run_dir}")

    return problems
