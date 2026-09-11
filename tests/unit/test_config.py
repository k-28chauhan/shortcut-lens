"""Tests for shortcut_lens.config: hashing, YAML composition, validation errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from shortcut_lens.config import (
    StrictBaseModel,
    config_hash,
    load_config,
    load_yaml_composed,
    short_hash,
)


class _TrainConfig(StrictBaseModel):
    lr: float
    epochs: int
    optimizer: str = "sgd"


def test_config_hash_independent_of_key_order() -> None:
    a = {"lr": 0.1, "epochs": 10, "nested": {"x": 1, "y": 2}}
    b = {"nested": {"y": 2, "x": 1}, "epochs": 10, "lr": 0.1}
    assert config_hash(a) == config_hash(b)


def test_config_hash_stable_across_calls() -> None:
    data = {"lr": 0.1, "epochs": 10}
    assert config_hash(data) == config_hash(dict(data))


def test_config_hash_changes_with_value() -> None:
    a = {"lr": 0.1}
    b = {"lr": 0.2}
    assert config_hash(a) != config_hash(b)


def test_short_hash_is_prefix_of_full_hash() -> None:
    data = {"lr": 0.1, "epochs": 10}
    assert config_hash(data).startswith(short_hash(data))
    assert len(short_hash(data)) == 8
    assert len(short_hash(data, length=4)) == 4


def test_load_yaml_composed_merges_base(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("lr: 0.1\nepochs: 10\noptimizer: sgd\n")
    child = tmp_path / "child.yaml"
    child.write_text("base: base.yaml\nlr: 0.5\n")

    resolved = load_yaml_composed(child)

    assert resolved == {"lr": 0.5, "epochs": 10, "optimizer": "sgd"}


def test_load_yaml_composed_merges_nested_dicts(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("train:\n  lr: 0.1\n  epochs: 10\n")
    child = tmp_path / "child.yaml"
    child.write_text("base: base.yaml\ntrain:\n  lr: 0.5\n")

    resolved = load_yaml_composed(child)

    assert resolved == {"train": {"lr": 0.5, "epochs": 10}}


def test_load_yaml_composed_chains_multiple_bases(tmp_path: Path) -> None:
    grandparent = tmp_path / "grandparent.yaml"
    grandparent.write_text("a: 1\nb: 1\n")
    parent = tmp_path / "parent.yaml"
    parent.write_text("base: grandparent.yaml\nb: 2\n")
    child = tmp_path / "child.yaml"
    child.write_text("base: parent.yaml\nc: 3\n")

    resolved = load_yaml_composed(child)

    assert resolved == {"a": 1, "b": 2, "c": 3}


def test_load_yaml_composed_rejects_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- 1\n- 2\n")

    with pytest.raises(ValueError, match="mapping"):
        load_yaml_composed(bad)


def test_load_config_validates_and_returns_hash(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("lr: 0.1\nepochs: 5\n")

    validated, digest = load_config(config_path, _TrainConfig)

    assert validated == _TrainConfig(lr=0.1, epochs=5, optimizer="sgd")
    assert digest == config_hash({"lr": 0.1, "epochs": 5})


def test_load_config_rejects_unknown_field(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("lr: 0.1\nepochs: 5\ntypo_field: 1\n")

    with pytest.raises(ValidationError, match="typo_field"):
        load_config(config_path, _TrainConfig)


def test_load_config_readable_error_on_wrong_type(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("lr: not_a_number\nepochs: 5\n")

    with pytest.raises(ValidationError, match="lr"):
        load_config(config_path, _TrainConfig)
