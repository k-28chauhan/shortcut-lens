"""Stores embeddings as float16 `.npy` plus an aligned id file; validates alignment on load.

Concept: a silent id misalignment between an embedding table and the predictions/oracle table it
is joined with would produce a plausible-looking wrong result -- this is exactly the kind of bug
docs/TESTING.md exists to catch. Storage is float16 on disk (half the size, and the precision loss
is far below what changes a nearest-neighbour or clustering result) but always read back as
float32, since numpy/sklearn arithmetic on float16 is slow and rarely supported well.

Pipeline position: label-free zone. Used by `embeddings/model_space.py` and `clip_space.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def embedding_paths(embeddings_dir: Path, split: str) -> tuple[Path, Path]:
    """`(<split>.npy, <split>.ids.parquet)` under `embeddings_dir` (ARCHITECTURE §4)."""
    return embeddings_dir / f"{split}.npy", embeddings_dir / f"{split}.ids.parquet"


def write_embeddings(
    embeddings_dir: Path, split: str, vectors: np.ndarray, example_ids: Sequence[str]
) -> None:
    """Write `vectors` (float16 on disk) and their aligned `example_ids` for `split`."""
    if len(vectors) != len(example_ids):
        raise ValueError(
            f"vectors has {len(vectors)} rows but example_ids has {len(example_ids)} entries"
        )
    npy_path, ids_path = embedding_paths(embeddings_dir, split)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, vectors.astype(np.float16))
    pd.DataFrame({"example_id": list(example_ids)}).to_parquet(ids_path, index=False)


def read_embeddings(embeddings_dir: Path, split: str) -> tuple[np.ndarray, list[str]]:
    """Read `(vectors, example_ids)` for `split`, upcast to float32. Raises on length mismatch."""
    npy_path, ids_path = embedding_paths(embeddings_dir, split)
    vectors = np.load(npy_path).astype(np.float32)
    example_ids = pd.read_parquet(ids_path)["example_id"].tolist()
    if len(vectors) != len(example_ids):
        raise ValueError(
            f"{npy_path} has {len(vectors)} rows but {ids_path} has {len(example_ids)} ids"
        )
    return vectors, example_ids


def assert_aligned(ids_a: Sequence[str], ids_b: Sequence[str], *, context: str) -> None:
    """Raise if two id sequences are not identical, in the same order (ARCHITECTURE §4)."""
    if list(ids_a) != list(ids_b):
        raise ValueError(f"{context}: id sequences are not aligned (different order or members)")
