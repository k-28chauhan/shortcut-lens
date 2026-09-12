"""Reads `public.parquet`: example_id, dataset, split, y, class_name, image_ref.

Concept: this is the only table label-free code may read. Loading validates required columns,
no duplicate example_id, and valid split names (see docs/ARCHITECTURE.md §4) -- a silent id
misalignment or a typo'd split name is exactly the kind of bug that produces a plausible-looking
wrong result later, so it is checked once, here, at the boundary.

Pipeline position: label-free zone. Used by every label-free stage that needs class labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"example_id", "dataset", "split", "y", "class_name", "image_ref"}
VALID_SPLITS = {"train", "val_a", "val_b", "test"}


def read_public_table(path: str | Path) -> pd.DataFrame:
    """Read and validate `public.parquet`."""
    df = pd.read_parquet(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"public table {path} missing required columns: {sorted(missing)}")

    if df["example_id"].duplicated().any():
        duplicates = df.loc[df["example_id"].duplicated(), "example_id"].tolist()
        raise ValueError(f"public table {path} has duplicate example_id values: {duplicates[:5]}")

    invalid_splits = set(df["split"].unique()) - VALID_SPLITS
    if invalid_splits:
        raise ValueError(f"public table {path} has invalid split names: {sorted(invalid_splits)}")

    return df
