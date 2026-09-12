"""Tests for shortcut_lens.build.waterbirds. Network-marked: downloads the HF parquet mirror.

D-027 (v2): downloaded from `grodino/waterbirds` on Hugging Face rather than the official CodaLab
tarball (impractically slow on this connection), after verifying every group count below matches
docs/PRD.md §7 exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shortcut_lens.build.datasets import RenderedImageDataset
from shortcut_lens.build.waterbirds import (
    EXPECTED_GROUP_COUNTS,
    WaterbirdsConfig,
    WaterbirdsSourceFile,
    build_waterbirds,
    download_and_verify,
)

pytestmark = pytest.mark.network

_SOURCE = {
    "hf_revision": "e9856c710d0da2e4029d116cdd9d5fce7cc2bc80",
    "train_parquet": {
        "url": "https://huggingface.co/api/datasets/grodino/waterbirds/parquet/default/train/0.parquet",
        "sha256": "575b462a17ae00e6a03b4f1833e3a5952e28492e278d3246186586c27a46367a",
    },
    "val_parquet": {
        "url": "https://huggingface.co/api/datasets/grodino/waterbirds/parquet/default/validation/0.parquet",
        "sha256": "9ab73e8625fb5856e132ecbbe509f20829e5d59a9bd087b6cda86f5c478350fb",
    },
    "test_parquet": {
        "url": "https://huggingface.co/api/datasets/grodino/waterbirds/parquet/default/test/0.parquet",
        "sha256": "fdb8696ce4db78c0ad48dd0120aaae40ce4330416d37fc41c9ea57faadc007f7",
    },
}


@pytest.fixture(scope="module")
def cache_dir() -> Path:
    return Path.home() / ".cache" / "shortcut-lens" / "downloads"


def _build(
    tmp_path: Path, cache_dir: Path, **overrides: object
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = dict(_SOURCE, **overrides)
    cfg = WaterbirdsConfig.model_validate(config)
    return build_waterbirds(cfg, cache_dir, tmp_path / "images")


def test_group_counts_match_frozen_prd_values(tmp_path: Path, cache_dir: Path) -> None:
    public, oracle = _build(tmp_path, cache_dir)
    merged = public.merge(oracle, on="example_id")

    split_map = {"train": "train", "test": "test"}
    for internal_split, prd_split in split_map.items():
        counts = (
            merged.loc[merged["split"] == internal_split].groupby("group_name").size().to_dict()
        )
        assert counts == EXPECTED_GROUP_COUNTS[prd_split]

    val_counts = (
        merged.loc[merged["split"].isin(["val_a", "val_b"])].groupby("group_name").size().to_dict()
    )
    assert val_counts == EXPECTED_GROUP_COUNTS["val"]


def test_val_a_val_b_disjoint_and_stratified_by_class(tmp_path: Path, cache_dir: Path) -> None:
    public, _ = _build(tmp_path, cache_dir)
    val_a = public.loc[public["split"] == "val_a"]
    val_b = public.loc[public["split"] == "val_b"]

    assert set(val_a["example_id"]).isdisjoint(set(val_b["example_id"]))
    # each half's class counts sum to the official val class totals (599 landbird, 600 waterbird)
    for y, total in ((0, 933), (1, 266)):
        assert (val_a["y"] == y).sum() + (val_b["y"] == y).sum() == total


def test_dataset_returns_exactly_image_y_example_id(tmp_path: Path, cache_dir: Path) -> None:
    public, _ = _build(tmp_path, cache_dir)
    dataset = RenderedImageDataset(public, tmp_path / "images")
    item = dataset[0]
    assert set(item.keys()) == {"image", "y", "example_id"}


def test_checksum_mismatch_raises(tmp_path: Path) -> None:
    """Isolated from the real cache: a fresh 'download' (a local file://) that doesn't match
    the expected sha256 must raise, rather than being silently accepted (CLAUDE.md §3)."""
    fake_source = tmp_path / "fake_upstream.parquet"
    fake_source.write_bytes(b"not the real parquet content")
    source = WaterbirdsSourceFile(url=fake_source.as_uri(), sha256="0" * 64)

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_and_verify(tmp_path / "cache", "train", source)
