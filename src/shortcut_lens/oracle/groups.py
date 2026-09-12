"""The only reader of `oracle.parquet`: exposes group labels to oracle-zone code.

Concept: centralising the one place that opens `oracle.parquet` makes the label-free/oracle
boundary enforceable by a simple grep-based contract test (`tests/contracts/`) in addition to
import-linter (see docs/TESTING.md §5) -- both must agree this file, and nothing in the label-free
zone, is where `oracle.parquet` gets read.

Pipeline position: oracle zone. Used by `evaluation/`, `verification/`, `mitigation/oracle_ref/`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"example_id", "attribute", "group", "group_name", "is_minority"}


def read_oracle_table(path: str | Path) -> pd.DataFrame:
    """Read and validate `oracle.parquet`."""
    df = pd.read_parquet(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"oracle table {path} missing required columns: {sorted(missing)}")

    if df["example_id"].duplicated().any():
        duplicates = df.loc[df["example_id"].duplicated(), "example_id"].tolist()
        raise ValueError(f"oracle table {path} has duplicate example_id values: {duplicates[:5]}")

    return df
