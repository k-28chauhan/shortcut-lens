"""Waterbirds (Sagawa et al. 2020): landbird/waterbird x land/water background, natural benchmark.

Concept: 95% of waterbirds appear on water and 95% of landbirds on land in training data, so a
model can learn "background predicts label" instead of the bird itself. Unlike PlantedPets, we
did not create this shortcut -- it lets us check whether the pipeline's findings on a shortcut we
control generalise to one we did not (see docs/PRD.md §7).

Downloaded from a Hugging Face parquet mirror (`grodino/waterbirds`) rather than the official
CodaLab tarball WILDS' own source points to (D-027 v2, docs/DECISIONS.md): the CodaLab download
proved impractically slow on this connection (~30-40KB/s for ~470MB; HF's CDN serves the same data
in seconds). D-027 originally rejected third-party mirrors on provenance grounds -- what changes
that here is empirical verification, not just trust: all 12 (class x background) group counts
across all three splits match docs/PRD.md §7's frozen values EXACTLY, strong evidence this is a
faithful repackaging of the same official data, not a different construction (its own dataset card
description is misleadingly generic and does not match this evidence, which is why the counts were
checked directly rather than taken on faith). Each split's parquet file URL and SHA-256, and the HF
dataset's git revision, are pinned in `configs/datasets/waterbirds.yaml`; the SHA-256 checks are
what actually gate the build against future drift.

Pipeline position: oracle zone.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image

from shortcut_lens.build.splits import realistic_subsample, stratified_half_split
from shortcut_lens.config import StrictBaseModel

_Y_NAMES = {0: "landbird", 1: "waterbird"}
_PLACE_NAMES = {0: "land", 1: "water"}

# Frozen from docs/PRD.md §7. A mismatch means stop and investigate -- never adapt (CLAUDE.md §3).
EXPECTED_GROUP_COUNTS: dict[str, dict[str, int]] = {
    "train": {
        "landbird|land": 3498,
        "landbird|water": 184,
        "waterbird|land": 56,
        "waterbird|water": 1057,
    },
    "val": {
        "landbird|land": 467,
        "landbird|water": 466,
        "waterbird|land": 133,
        "waterbird|water": 133,
    },
    "test": {
        "landbird|land": 2255,
        "landbird|water": 2255,
        "waterbird|land": 642,
        "waterbird|water": 642,
    },
}


class WaterbirdsSourceFile(StrictBaseModel):
    url: str
    sha256: str


class WaterbirdsConfig(StrictBaseModel):
    """See docs/PRD.md §7 and docs/ARCHITECTURE.md §6 for the field meanings."""

    hf_revision: str
    train_parquet: WaterbirdsSourceFile
    val_parquet: WaterbirdsSourceFile
    test_parquet: WaterbirdsSourceFile
    val_mode: Literal["balanced", "realistic"] = "balanced"
    minority_fraction: float | None = None
    build_seed: int = 0

    def source_file(self, split: Literal["train", "val", "test"]) -> WaterbirdsSourceFile:
        return {"train": self.train_parquet, "val": self.val_parquet, "test": self.test_parquet}[
            split
        ]


def sha256_of(path: Path) -> str:
    """Streamed sha256 of a (potentially large) file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_verify(cache_dir: Path, split: str, source: WaterbirdsSourceFile) -> Path:
    """Download one split's parquet file if needed, verifying its checksum.

    Skips the download if a cached copy already matches `source.sha256`. Deletes and re-downloads
    on a mismatch, and raises if the fresh copy still does not match -- never silently accepts a
    wrong file (CLAUDE.md §3).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"waterbirds_{split}.parquet"

    if path.exists():
        if sha256_of(path) == source.sha256:
            return path
        path.unlink()

    urllib.request.urlretrieve(source.url, path)  # checksum-verified below

    actual = sha256_of(path)
    if actual != source.sha256:
        path.unlink()
        raise ValueError(
            f"waterbirds {split} parquet checksum mismatch: expected {source.sha256}, got {actual}"
        )
    return path


def _read_split(
    cache_dir: Path, split: Literal["train", "val", "test"], source: WaterbirdsSourceFile
) -> pd.DataFrame:
    parquet_path = download_and_verify(cache_dir, split, source)
    df = pd.read_parquet(parquet_path, columns=["image", "label", "place"]).reset_index(drop=True)
    df["example_id"] = [f"waterbirds-{split}-{i:06d}" for i in range(len(df))]
    return df


def _assert_expected_group_counts(dfs: dict[str, pd.DataFrame]) -> None:
    for split, df in dfs.items():
        actual = {
            f"{y_name}|{place_name}": int(((df["label"] == y) & (df["place"] == place)).sum())
            for y, y_name in _Y_NAMES.items()
            for place, place_name in _PLACE_NAMES.items()
        }
        expected = EXPECTED_GROUP_COUNTS[split]
        if actual != expected:
            raise ValueError(
                f"waterbirds {split!r} split counts do not match docs/PRD.md §7: "
                f"expected {expected}, got {actual}. Stop and investigate; do not adapt the check."
            )


def build_waterbirds(
    config: WaterbirdsConfig, cache_dir: Path, image_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (public_table, oracle_table) from the pinned, checksum-verified HF parquet mirror."""
    image_dir.mkdir(parents=True, exist_ok=True)

    dfs = {
        split: _read_split(cache_dir, split, config.source_file(split))
        for split in ("train", "val", "test")
    }
    _assert_expected_group_counts(dfs)

    train_df = dfs["train"]
    minority_place_by_class: dict[int, int] = {}
    for y in (0, 1):
        cls_train_place = train_df.loc[train_df["label"] == y, "place"]
        water_fraction = float((cls_train_place == 1).mean())
        minority_place_by_class[y] = 1 if water_fraction < 0.5 else 0

    for df in dfs.values():
        df["is_minority"] = [
            int(place) == minority_place_by_class[int(y)]
            for y, place in zip(df["label"], df["place"], strict=True)
        ]

    val_df = dfs["val"]
    if config.val_mode == "realistic":
        fraction = config.minority_fraction
        if fraction is None:
            fractions = []
            for y in (0, 1):
                cls_train_place = train_df.loc[train_df["label"] == y, "place"]
                fractions.append(float((cls_train_place == minority_place_by_class[y]).mean()))
            fraction = sum(fractions) / len(fractions)
        kept_ids = set(
            realistic_subsample(
                val_df["example_id"].tolist(),
                val_df["label"].tolist(),
                val_df["is_minority"].tolist(),
                fraction,
                config.build_seed,
            )
        )
        val_df = val_df[val_df["example_id"].isin(kept_ids)]

    val_a_split, _ = stratified_half_split(
        val_df["example_id"].tolist(), val_df["label"].tolist(), config.build_seed
    )
    val_a_ids = set(val_a_split)
    kept_val_ids = set(val_df["example_id"])

    public_records: list[dict[str, object]] = []
    oracle_records: list[dict[str, object]] = []

    def _emit(df: pd.DataFrame, resolve_split: Callable[[str], str | None]) -> None:
        for record in df.to_dict("records"):
            example_id = str(record["example_id"])
            resolved_split = resolve_split(example_id)
            if resolved_split is None:
                continue

            y, place = int(record["label"]), int(record["place"])
            class_name, place_name = _Y_NAMES[y], _PLACE_NAMES[place]

            image_ref = f"{example_id}.jpg"
            image_path = image_dir / image_ref
            if not image_path.exists():
                image_bytes = record["image"]["bytes"]
                Image.open(io.BytesIO(image_bytes)).convert("RGB").save(image_path, quality=95)

            public_records.append(
                {
                    "example_id": example_id,
                    "dataset": "waterbirds",
                    "split": resolved_split,
                    "y": y,
                    "class_name": class_name,
                    "image_ref": image_ref,
                }
            )
            oracle_records.append(
                {
                    "example_id": example_id,
                    "attribute": place,
                    "group": 2 * y + place,
                    "group_name": f"{class_name}|{place_name}",
                    "is_minority": bool(record["is_minority"]),
                }
            )

    _emit(dfs["train"], lambda _eid: "train")
    _emit(
        dfs["val"],
        lambda eid: ("val_a" if eid in val_a_ids else "val_b") if eid in kept_val_ids else None,
    )
    _emit(dfs["test"], lambda _eid: "test")

    return pd.DataFrame.from_records(public_records), pd.DataFrame.from_records(oracle_records)
