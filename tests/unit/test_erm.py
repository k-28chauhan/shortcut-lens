"""Tests for shortcut_lens.training.erm: overfitting, exact resume, device/precision policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from shortcut_lens.manifest import read_manifest
from shortcut_lens.training.erm import (
    OptimizerConfig,
    TrainConfig,
    resolve_device,
    resolve_precision_mode,
    train_erm,
)

_IMAGE_SIZE = 16


class _FakeImageDataset:
    """A tiny in-memory `ImageDataset`: random images, y correlated with mean pixel value."""

    def __init__(self, n: int, seed: int, tag: str) -> None:
        rng = np.random.default_rng(seed)
        self._images = rng.random((n, 3, _IMAGE_SIZE, _IMAGE_SIZE)).astype(np.float32)
        means = self._images.mean(axis=(1, 2, 3))
        self._y = (means > np.median(means)).astype(int)
        self._ids = [f"{tag}-{i}" for i in range(n)]

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": torch.tensor(self._images[index]),
            "y": int(self._y[index]),
            "example_id": self._ids[index],
        }


def _datasets() -> dict[str, _FakeImageDataset]:
    return {
        "train": _FakeImageDataset(16, seed=0, tag="train"),
        "val_a": _FakeImageDataset(8, seed=1, tag="val_a"),
        "val_b": _FakeImageDataset(8, seed=2, tag="val_b"),
        "test": _FakeImageDataset(8, seed=3, tag="test"),
    }


def _tiny_config(epochs: int, batch_size: int = 4, lr: float = 0.05) -> TrainConfig:
    return TrainConfig(
        arch="tiny_cnn",
        pretrained=None,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=OptimizerConfig(lr=lr),
        seed=0,
        device="cpu",
        num_workers=0,
    )


def test_resolve_device_auto_prefers_mps_then_cuda_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_resolve_device_explicit_passthrough() -> None:
    assert resolve_device("cpu") == "cpu"


def test_resolve_precision_mode_cuda_amp_true_is_fp16_amp() -> None:
    assert resolve_precision_mode(amp=True, device="cuda") == "fp16_amp"


def test_resolve_precision_mode_non_cuda_amp_true_warns_and_is_fp32(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        mode = resolve_precision_mode(amp=True, device="mps")
    assert mode == "fp32"
    assert "amp=true has no effect on mps" in caplog.text


def test_resolve_precision_mode_amp_false_is_fp32_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        mode = resolve_precision_mode(amp=False, device="cpu")
    assert mode == "fp32"
    assert caplog.text == ""


def test_tiny_cnn_overfits_a_batch_of_eight(tmp_path: Path) -> None:
    datasets = _datasets()
    train_only = {**datasets, "train": _FakeImageDataset(8, seed=0, tag="overfit")}
    config = _tiny_config(epochs=40, batch_size=8, lr=0.1)
    config_dict = config.model_dump()

    history = train_erm(config, config_dict, train_only, tmp_path / "run", run_id="overfit-test")

    last_row = history.iloc[-1]
    assert last_row["train_acc"] == pytest.approx(1.0)
    assert last_row["train_loss"] < 0.1


def test_history_records_a_positive_duration_per_epoch(tmp_path: Path) -> None:
    datasets = _datasets()
    config = _tiny_config(epochs=3)

    history = train_erm(config, config.model_dump(), datasets, tmp_path / "run", run_id="timing")

    assert "duration_s" in history.columns
    assert (history["duration_s"] > 0).all()


def test_train_erm_writes_predictions_and_manifest_for_every_split(tmp_path: Path) -> None:
    datasets = _datasets()
    config = _tiny_config(epochs=2)
    config_dict = config.model_dump()

    train_erm(config, config_dict, datasets, tmp_path / "run", run_id="predict-test")

    for split, dataset in datasets.items():
        predictions_path = tmp_path / "run" / "predictions" / f"{split}.parquet"
        assert predictions_path.exists()
        predictions = pd.read_parquet(predictions_path)
        assert len(predictions) == len(dataset)
        assert set(predictions.columns) == {
            "example_id",
            "y",
            "y_hat",
            "p_y",
            "p_max",
            "logit_0",
            "logit_1",
            "correct",
        }

    manifest = read_manifest(tmp_path / "run" / "manifest.json")
    assert manifest.stage == "train"
    assert manifest.precision_mode == "fp32"
    assert manifest.hardware["device"] == "cpu"


def test_resume_from_checkpoint_matches_uninterrupted_run(tmp_path: Path) -> None:
    datasets = _datasets()
    config = _tiny_config(epochs=6)
    config_dict = config.model_dump()

    # uninterrupted run
    train_erm(config, config_dict, datasets, tmp_path / "uninterrupted", run_id="a")

    # simulate an interrupted-then-resumed run: stop after epoch 2, then continue
    train_erm(
        config,
        config_dict,
        datasets,
        tmp_path / "resumed",
        run_id="b",
        stop_after_epoch=2,
    )
    train_erm(config, config_dict, datasets, tmp_path / "resumed", run_id="b")

    uninterrupted_checkpoint = torch.load(
        tmp_path / "uninterrupted" / "train" / "checkpoint_last.pt", map_location="cpu"
    )
    resumed_checkpoint = torch.load(
        tmp_path / "resumed" / "train" / "checkpoint_last.pt", map_location="cpu"
    )

    assert uninterrupted_checkpoint["epoch"] == resumed_checkpoint["epoch"] == 5
    for key, value in uninterrupted_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(value, resumed_checkpoint["model_state_dict"][key])


def test_resume_raises_on_config_hash_mismatch(tmp_path: Path) -> None:
    datasets = _datasets()
    config = _tiny_config(epochs=4)
    config_dict = config.model_dump()
    run_dir = tmp_path / "run"

    train_erm(config, config_dict, datasets, run_dir, run_id="a", stop_after_epoch=1)

    different_config_dict = {**config_dict, "seed": 999}
    with pytest.raises(ValueError, match="different config"):
        train_erm(config, different_config_dict, datasets, run_dir, run_id="a")


def test_train_erm_rejects_missing_split() -> None:
    config = _tiny_config(epochs=1)
    datasets = _datasets()
    del datasets["test"]
    with pytest.raises(ValueError, match="exactly the keys"):
        train_erm(config, config.model_dump(), datasets, Path("/tmp/unused"), run_id="x")
