"""Tests for shortcut_lens.data.public and shortcut_lens.oracle.groups: schema validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shortcut_lens.data.public import read_public_table
from shortcut_lens.oracle.groups import read_oracle_table


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    df.to_parquet(path, index=False)
    return path


def test_read_public_table_round_trip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "dataset": ["synthetic_shapes", "synthetic_shapes"],
            "split": ["train", "val_a"],
            "y": [0, 1],
            "class_name": ["circle", "square"],
            "image_ref": ["a.png", "b.png"],
        }
    )
    path = _write_parquet(df, tmp_path / "public.parquet")
    loaded = read_public_table(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_read_public_table_rejects_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"example_id": ["a"], "y": [0]})
    path = _write_parquet(df, tmp_path / "public.parquet")
    with pytest.raises(ValueError, match="missing required columns"):
        read_public_table(path)


def test_read_public_table_rejects_duplicate_ids(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "example_id": ["a", "a"],
            "dataset": ["d", "d"],
            "split": ["train", "train"],
            "y": [0, 1],
            "class_name": ["x", "y"],
            "image_ref": ["a.png", "a2.png"],
        }
    )
    path = _write_parquet(df, tmp_path / "public.parquet")
    with pytest.raises(ValueError, match="duplicate example_id"):
        read_public_table(path)


def test_read_public_table_rejects_invalid_split(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "example_id": ["a"],
            "dataset": ["d"],
            "split": ["bogus_split"],
            "y": [0],
            "class_name": ["x"],
            "image_ref": ["a.png"],
        }
    )
    path = _write_parquet(df, tmp_path / "public.parquet")
    with pytest.raises(ValueError, match="invalid split"):
        read_public_table(path)


def test_read_oracle_table_round_trip(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "attribute": [0, 1],
            "group": [0, 3],
            "group_name": ["circle|no_dot", "square|dot"],
            "is_minority": [True, True],
        }
    )
    path = _write_parquet(df, tmp_path / "oracle.parquet")
    loaded = read_oracle_table(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_read_oracle_table_rejects_duplicate_ids(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "example_id": ["a", "a"],
            "attribute": [0, 1],
            "group": [0, 1],
            "group_name": ["x", "y"],
            "is_minority": [False, False],
        }
    )
    path = _write_parquet(df, tmp_path / "oracle.parquet")
    with pytest.raises(ValueError, match="duplicate example_id"):
        read_oracle_table(path)
