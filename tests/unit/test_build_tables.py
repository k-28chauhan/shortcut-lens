"""Tests for shortcut_lens.build.tables: writes public/oracle parquet + a manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shortcut_lens.artifacts import LocalStore
from shortcut_lens.build.tables import write_build_tables
from shortcut_lens.data.public import read_public_table
from shortcut_lens.oracle.groups import read_oracle_table


def test_write_build_tables_round_trip(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    public = pd.DataFrame(
        {
            "example_id": ["a"],
            "dataset": ["synthetic_shapes"],
            "split": ["train"],
            "y": [0],
            "class_name": ["circle"],
            "image_ref": ["a.png"],
        }
    )
    oracle = pd.DataFrame(
        {
            "example_id": ["a"],
            "attribute": [0],
            "group": [0],
            "group_name": ["circle|no_dot"],
            "is_minority": [False],
        }
    )

    manifest = write_build_tables(
        store,
        "synthetic_shapes",
        "abcd1234",
        public,
        oracle,
        config={"variant": "dot", "rho": 0.95},
        seed=0,
        duration_s=1.0,
    )

    build_dir = tmp_path / "builds" / "synthetic_shapes" / "abcd1234"
    loaded_public = read_public_table(build_dir / "public.parquet")
    loaded_oracle = read_oracle_table(build_dir / "oracle.parquet")

    pd.testing.assert_frame_equal(loaded_public, public)
    pd.testing.assert_frame_equal(loaded_oracle, oracle)
    assert (build_dir / "manifest.json").exists()
    assert manifest.stage == "build"
