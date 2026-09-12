"""Tests for shortcut_lens.build.synthetic_shapes: split sizes, class balance, proportions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from shortcut_lens.build.synthetic_shapes import SyntheticShapesConfig, build_synthetic_shapes


def _build(tmp_path: Path, **overrides: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    defaults: dict[str, object] = {
        "variant": "dot",
        "rho": 0.95,
        "build_seed": 0,
        "n_train": 200,
        "n_val": 80,
        "n_test": 80,
    }
    defaults.update(overrides)
    config = SyntheticShapesConfig.model_validate(defaults)
    return build_synthetic_shapes(config, tmp_path / "images")


def test_split_sizes_and_val_half_split(tmp_path: Path) -> None:
    public, _ = _build(tmp_path)
    counts = public["split"].value_counts()
    assert counts["train"] == 200
    assert counts["test"] == 80
    assert counts["val_a"] == 40
    assert counts["val_b"] == 40


def test_val_a_val_b_disjoint(tmp_path: Path) -> None:
    public, _ = _build(tmp_path)
    val_a = set(public.loc[public["split"] == "val_a", "example_id"])
    val_b = set(public.loc[public["split"] == "val_b", "example_id"])
    assert val_a.isdisjoint(val_b)


def test_class_balance_per_split(tmp_path: Path) -> None:
    public, _ = _build(tmp_path)
    for split in ("train", "test"):
        counts = public.loc[public["split"] == split, "class_name"].value_counts()
        assert counts["circle"] == counts["square"]


def test_planted_proportions_within_one_image_of_target(tmp_path: Path) -> None:
    public, oracle = _build(tmp_path, n_train=400, rho=0.95)
    merged = public.merge(oracle, on="example_id")
    train = merged[merged["split"] == "train"]

    for class_name, target_rho in (("circle", 0.95), ("square", 0.05)):
        cls_rows = train[train["class_name"] == class_name]
        n_present = (cls_rows["attribute"] == 1).sum()
        target = target_rho * len(cls_rows)
        assert abs(n_present - target) <= 1


def test_group_names_match_frozen_vocab_for_dot_variant(tmp_path: Path) -> None:
    _, oracle = _build(tmp_path, variant="dot")
    assert set(oracle["group_name"].unique()) == {
        "circle|dot",
        "circle|no_dot",
        "square|dot",
        "square|no_dot",
    }


def test_minority_flag_matches_low_frequency_combination(tmp_path: Path) -> None:
    public, oracle = _build(tmp_path, rho=0.95, n_train=400)
    merged = public.merge(oracle, on="example_id")
    train = merged[merged["split"] == "train"]

    assert bool(train.loc[train["group_name"] == "circle|no_dot", "is_minority"].all())
    assert bool(train.loc[train["group_name"] == "square|dot", "is_minority"].all())
    assert not train.loc[train["group_name"] == "circle|dot", "is_minority"].any()
    assert not train.loc[train["group_name"] == "square|no_dot", "is_minority"].any()


def test_deterministic_across_runs(tmp_path: Path) -> None:
    public_a, oracle_a = _build(tmp_path / "a")
    public_b, oracle_b = _build(tmp_path / "b")
    pd.testing.assert_frame_equal(public_a, public_b)
    pd.testing.assert_frame_equal(oracle_a, oracle_b)


def test_images_are_written_to_disk(tmp_path: Path) -> None:
    public, _ = _build(tmp_path)
    image_dir = tmp_path / "images"
    for image_ref in public["image_ref"]:
        assert (image_dir / image_ref).exists()


def test_background_variant_has_no_dot(tmp_path: Path) -> None:
    public, oracle = _build(tmp_path, variant="background", n_train=40, n_val=20, n_test=20)
    merged = public.merge(oracle, on="example_id")
    image_dir = tmp_path / "images"

    # a 'background' variant image with attribute present should have no red dot pixels
    row = merged[(merged["class_name"] == "circle") & (merged["attribute"] == 1)].iloc[0]
    image = np.array(Image.open(image_dir / row["image_ref"]))
    red_dot_pixels = np.all(image == np.array([215, 25, 25]), axis=-1).sum()
    assert red_dot_pixels == 0
