"""Contract test (second line of defense): no label-free module names `oracle.parquet`.

`lint-imports` (run in CI and `make check`) is the primary enforcement of the group-label
firewall via the import graph (docs/ARCHITECTURE.md §1). This is an independent, grep-style check
that would also catch a hardcoded path string bypassing the import graph entirely
(docs/TESTING.md §5).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "shortcut_lens"

LABEL_FREE_PACKAGES = (
    "data",
    "models",
    "training",
    "embeddings",
    "discovery",
    "naming",
    "mitigation/label_free",
    "mitigation/selection.py",
    "app",
)


def test_label_free_zone_never_mentions_oracle_parquet() -> None:
    offenders: list[str] = []
    for relative in LABEL_FREE_PACKAGES:
        target = SRC_ROOT / relative
        if target.is_file():
            paths = [target]
        elif target.is_dir():
            paths = list(target.rglob("*.py"))
        else:
            paths = []
        for path in paths:
            if "oracle.parquet" in path.read_text():
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, f"label-free modules reference 'oracle.parquet': {offenders}"
