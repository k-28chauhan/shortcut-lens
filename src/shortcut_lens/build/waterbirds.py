"""Waterbirds (Sagawa et al. 2020): landbird/waterbird x land/water background, natural benchmark.

Concept: 95% of waterbirds appear on water and 95% of landbirds on land in training data, so a
model can learn "background predicts label" instead of the bird itself. Unlike PlantedPets, we
did not create this shortcut -- it lets us check whether the pipeline's findings on a shortcut we
control generalise to one we did not (see docs/PRD.md §7).

Downloaded directly from the canonical CodaLab tarball (D-027: rejected the `wilds` PyPI package
-- it pulls in an unmaintained transitive dependency for one dataset). The URL and its SHA-256 are
pinned in `configs/datasets/waterbirds.yaml`; a mismatch means stop and report, never adapt.
`metadata.csv`'s columns (`y`, `place`, `img_filename`, `split`) and split encoding
(train=0, val=1, test=2) were read directly from WILDS' own `waterbirds_dataset.py` and
`wilds_dataset.py` source (CLAUDE.md §5: verify by reading/running, not from memory), not assumed.
The expected group counts per split are frozen from docs/PRD.md §7.

Pipeline position: oracle zone.
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd

from shortcut_lens.build.splits import realistic_subsample, stratified_half_split
from shortcut_lens.config import StrictBaseModel

TARBALL_URL = (
    "https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"
)

_SPLIT_CODES = {0: "train", 1: "val", 2: "test"}
_Y_NAMES = {0: "landbird", 1: "waterbird"}
_PLACE_NAMES = {0: "land", 1: "water"}
_REQUIRED_COLUMNS = {"y", "place", "img_filename", "split"}

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


class WaterbirdsConfig(StrictBaseModel):
    """See docs/PRD.md §7 and docs/ARCHITECTURE.md §6 for the field meanings."""

    tarball_url: str = TARBALL_URL
    tarball_sha256: str
    val_mode: Literal["balanced", "realistic"] = "balanced"
    minority_fraction: float | None = None
    build_seed: int = 0


def sha256_of(path: Path) -> str:
    """Streamed sha256 of a (potentially large) file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_verify(cache_dir: Path, url: str, expected_sha256: str) -> Path:
    """Download the tarball to `cache_dir` if needed, verifying its checksum.

    Skips the download if a copy already there matches `expected_sha256`. Deletes and re-downloads
    on a checksum mismatch, and raises if the freshly downloaded copy still does not match --
    never silently accepts a wrong file (CLAUDE.md §3).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tarball_path = cache_dir / "waterbirds_v1.0.tar.gz"

    if tarball_path.exists():
        if sha256_of(tarball_path) == expected_sha256:
            return tarball_path
        tarball_path.unlink()

    urllib.request.urlretrieve(url, tarball_path)

    actual = sha256_of(tarball_path)
    if actual != expected_sha256:
        tarball_path.unlink()
        raise ValueError(
            f"Waterbirds tarball checksum mismatch: expected {expected_sha256}, got {actual}"
        )
    return tarball_path


def extract(tarball_path: Path, extract_dir: Path) -> Path:
    """Extract the tarball unless already done; return the directory holding metadata.csv."""
    existing = list(extract_dir.rglob("metadata.csv")) if extract_dir.exists() else []
    if existing:
        return existing[0].parent

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path) as tar:
        tar.extractall(extract_dir)

    found = list(extract_dir.rglob("metadata.csv"))
    if not found:
        raise RuntimeError(f"no metadata.csv found after extracting {tarball_path!r}")
    return found[0].parent


def _validate_columns(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"waterbirds metadata.csv missing columns: {sorted(missing)}")


def _assert_expected_group_counts(df: pd.DataFrame) -> None:
    for split_code, split_name in _SPLIT_CODES.items():
        split_df = df[df["split"] == split_code]
        actual = {
            f"{y_name}|{place_name}": int(
                ((split_df["y"] == y) & (split_df["place"] == place)).sum()
            )
            for y, y_name in _Y_NAMES.items()
            for place, place_name in _PLACE_NAMES.items()
        }
        expected = EXPECTED_GROUP_COUNTS[split_name]
        if actual != expected:
            raise ValueError(
                f"waterbirds {split_name!r} split counts do not match docs/PRD.md §7: "
                f"expected {expected}, got {actual}. Stop and investigate; do not adapt the check."
            )


def build_waterbirds(config: WaterbirdsConfig, data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (public_table, oracle_table) from an extracted, checksum-verified download."""
    df = pd.read_csv(data_dir / "metadata.csv").reset_index(drop=True)
    _validate_columns(df)
    _assert_expected_group_counts(df)

    example_ids = [f"waterbirds-{i:06d}" for i in range(len(df))]
    df = df.assign(example_id=example_ids, split_name=df["split"].map(_SPLIT_CODES))

    train_mask = df["split_name"] == "train"
    minority_place_by_class: dict[int, int] = {}
    for y in (0, 1):
        cls_train = df.loc[train_mask & (df["y"] == y)]
        water_fraction = float((cls_train["place"] == 1).mean())
        minority_place_by_class[y] = 1 if water_fraction < 0.5 else 0

    is_minority = [
        int(place) == minority_place_by_class[int(y)]
        for y, place in zip(df["y"], df["place"], strict=True)
    ]
    df = df.assign(is_minority=is_minority)

    val_df = df.loc[df["split_name"] == "val"]
    if config.val_mode == "realistic":
        fraction = config.minority_fraction
        if fraction is None:
            fractions = []
            for y in (0, 1):
                cls_train_place = df.loc[train_mask & (df["y"] == y), "place"]
                fractions.append(float((cls_train_place == minority_place_by_class[y]).mean()))
            fraction = sum(fractions) / len(fractions)
        kept_ids = set(
            realistic_subsample(
                val_df["example_id"].tolist(),
                val_df["y"].tolist(),
                val_df["is_minority"].tolist(),
                fraction,
                config.build_seed,
            )
        )
        val_df = val_df[val_df["example_id"].isin(kept_ids)]

    val_a_split, _ = stratified_half_split(
        val_df["example_id"].tolist(), val_df["y"].tolist(), config.build_seed
    )
    val_a_ids = set(val_a_split)
    kept_val_ids = set(val_df["example_id"])

    public_records: list[dict[str, object]] = []
    oracle_records: list[dict[str, object]] = []
    for record in df.to_dict("records"):
        example_id = str(record["example_id"])
        split_name = str(record["split_name"])
        if split_name == "val":
            if example_id not in kept_val_ids:
                continue
            resolved_split = "val_a" if example_id in val_a_ids else "val_b"
        else:
            resolved_split = split_name

        y, place = int(record["y"]), int(record["place"])
        class_name, place_name = _Y_NAMES[y], _PLACE_NAMES[place]
        public_records.append(
            {
                "example_id": example_id,
                "dataset": "waterbirds",
                "split": resolved_split,
                "y": y,
                "class_name": class_name,
                "image_ref": record["img_filename"],
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
    return pd.DataFrame.from_records(public_records), pd.DataFrame.from_records(oracle_records)
