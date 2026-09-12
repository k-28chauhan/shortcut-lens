"""Writes `public.parquet`, `oracle.parquet` and the build manifest for a dataset.

Concept: this is the boundary artefact of the oracle zone -- `public.parquet` exposes only
`example_id, dataset, split, y, class_name, image_ref`; `oracle.parquet` (group labels) is never
read outside `oracle/groups.py` (see docs/ARCHITECTURE.md §4 for exact schemas). Each dataset's
own `build_*` function (in `synthetic_shapes.py`, `pets.py`, `waterbirds.py`) does the dataset-
specific work of producing the two tables; this module is the single place they get written to
disk with a manifest, so every build is traceable the same way regardless of dataset.

Pipeline position: oracle zone. Entry point for `slens build`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from shortcut_lens.artifacts import ArtifactStore, build_dir
from shortcut_lens.manifest import Manifest, new_manifest, write_manifest


def write_build_tables(
    store: ArtifactStore,
    dataset: str,
    build_hash: str,
    public_table: pd.DataFrame,
    oracle_table: pd.DataFrame,
    *,
    config: dict[str, object],
    seed: int,
    duration_s: float,
) -> Manifest:
    """Write public.parquet, oracle.parquet and manifest.json under builds/<dataset>/<hash>/."""
    directory = build_dir(store, dataset, build_hash)
    directory.mkdir(parents=True, exist_ok=True)

    public_table.to_parquet(directory / "public.parquet", index=False)
    oracle_table.to_parquet(directory / "oracle.parquet", index=False)

    manifest = new_manifest(
        run_id=f"{dataset}-{build_hash}",
        stage="build",
        config=config,
        seed=seed,
        inputs={},
        duration_s=duration_s,
    )
    write_manifest(directory / "manifest.json", manifest)
    return manifest


def write_data_report(
    public_table: pd.DataFrame,
    oracle_table: pd.DataFrame,
    image_root: Path,
    dataset: str,
    results_dir: Path,
    figures_dir: Path,
    samples_per_group: int = 6,
    thumb_px: int = 96,
) -> tuple[Path, Path]:
    """Write group counts per split (CSV) and a sample image grid, one row per group (FR-D5).

    Group counts come from joining `public_table` and `oracle_table` on `example_id` -- this is
    the one place outside the oracle zone's own modules where that join is allowed, since
    `slens data report` is itself an oracle-zone command (docs/ARCHITECTURE.md §7).
    """
    merged = public_table.merge(oracle_table, on="example_id")
    counts = (
        merged.groupby(["split", "group_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "group_name"])
        .reset_index(drop=True)
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    counts_path = results_dir / f"data_counts_{dataset}.csv"
    counts.to_csv(counts_path, index=False)

    groups = sorted(oracle_table["group_name"].unique())
    grid = Image.new("RGB", (thumb_px * samples_per_group, thumb_px * len(groups)), (255, 255, 255))
    for row_idx, group_name in enumerate(groups):
        sample = merged.loc[merged["group_name"] == group_name].head(samples_per_group)
        for col_idx, image_ref in enumerate(sample["image_ref"]):
            thumbnail = Image.open(image_root / image_ref).convert("RGB")
            thumbnail = thumbnail.resize((thumb_px, thumb_px))
            grid.paste(thumbnail, (col_idx * thumb_px, row_idx * thumb_px))

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / f"samples_{dataset}.png"
    grid.save(figure_path)

    return counts_path, figure_path
