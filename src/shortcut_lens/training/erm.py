"""Empirical risk minimisation (ordinary fine-tuning) with AMP, checkpointing and exact resume.

Concept: ERM just minimises average training loss -- it is expected to learn shortcuts when they
correlate with the label, which is exactly what this project measures. Checkpoint selection uses
average validation accuracy only, never worst-group accuracy, because the latter would require
group labels (D-004, the "no group labels for any choice" integrity rule). "Average validation
accuracy" means accuracy over `val_a` and `val_b` combined (D-032) -- selection is orthogonal to
the val_a/val_b split, which exists for the discover/confirm firewall (D-005), not for model
selection.

Device policy (D-025): `resolve_device("auto")` picks mps > cuda > cpu. Exact reproducibility
(the resume test) never runs on MPS, since MPS reductions are not bit-stable run to run
(CLAUDE.md §4) -- only forced `device="cpu"` runs are checked for exact resume. Reproducible
shuffling does not rely on snapshotting global RNG state (D-035): each epoch gets its own
`DataLoader` built with a fresh `torch.Generator` seeded by `seeding.make_rng(seed, "erm_shuffle",
epoch)`, so epoch k's shuffle order is a pure function of `(seed, epoch)` alone -- resuming from
any earlier checkpoint reproduces every later epoch's shuffle order exactly, with nothing to
snapshot.

Predictions are written and returned as plain `pandas.DataFrame`s, not the `Predictions` dataclass
ARCHITECTURE §3 sketches (D-034) -- D-031 expected this module to be where that dataclass would
finally earn its place; it turned out not to.

Pipeline position: label-free zone. Entry point for `slens train`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from shortcut_lens.config import StrictBaseModel, config_hash
from shortcut_lens.manifest import PrecisionMode, new_manifest, write_manifest
from shortcut_lens.models.backbones import Arch, build_backbone
from shortcut_lens.seeding import make_rng
from shortcut_lens.types import ImageDataset

_LOGGER = logging.getLogger(__name__)

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
_SPLITS = ("train", "val_a", "val_b", "test")


class OptimizerConfig(StrictBaseModel):
    """SGD is the only optimizer this project uses -- ARCHITECTURE §6's example config."""

    name: Literal["sgd"] = "sgd"
    lr: float
    momentum: float = 0.9
    weight_decay: float = 1.0e-4


class TrainConfig(StrictBaseModel):
    """`train:` section of an experiment config (ARCHITECTURE §6). Colocated with its consumer."""

    arch: Arch
    pretrained: str | None = "IMAGENET1K_V1"
    epochs: int
    batch_size: int
    optimizer: OptimizerConfig
    lr_schedule: Literal["constant"] = "constant"
    amp: bool = False
    select_by: Literal["val_avg_acc"] = "val_avg_acc"
    seed: int
    device: DeviceName = "auto"
    num_workers: int = 0


def resolve_device(device: DeviceName) -> str:
    """`auto` -> mps > cuda > cpu (D-025); any explicit value passes through unchanged."""
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_precision_mode(amp: bool, device: str) -> PrecisionMode:
    """`amp=true` only has an effect on CUDA (AMP + `GradScaler`); MPS/CPU always run `fp32`.

    Never raises for `amp=true` on a non-CUDA device -- logs a one-time warning instead, since the
    manifest's `precision_mode` field (not the log) is the thing later code and reports should
    trust (D-033, human instruction: don't silently no-op, don't raise either).
    """
    if amp and device != "cuda":
        _LOGGER.warning("amp=true has no effect on %s; running in float32", device)
    if amp and device == "cuda":
        return "fp16_amp"
    return "fp32"


def _predict(model: nn.Module, dataset: ImageDataset, device: str, batch_size: int) -> pd.DataFrame:
    """Run `model` over every example in `dataset`; return a predictions/{split}.parquet frame."""
    model.eval()
    loader = DataLoader(cast(Dataset[Any], dataset), batch_size=batch_size, shuffle=False)

    example_ids: list[str] = []
    ys: list[int] = []
    logits_rows: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).cpu()
            example_ids.extend(batch["example_id"])
            ys.extend(int(y) for y in batch["y"])
            logits_rows.append(logits)

    logits_all = torch.cat(logits_rows, dim=0)
    probs = torch.softmax(logits_all, dim=1)
    y_hat = probs.argmax(dim=1)
    y_tensor = torch.tensor(ys)
    p_y = probs[torch.arange(len(ys)), y_tensor]

    return pd.DataFrame(
        {
            "example_id": example_ids,
            "y": ys,
            "y_hat": y_hat.tolist(),
            "p_y": p_y.tolist(),
            "p_max": probs.max(dim=1).values.tolist(),
            "logit_0": logits_all[:, 0].tolist(),
            "logit_1": logits_all[:, 1].tolist(),
            "correct": (y_hat == y_tensor).tolist(),
        }
    )


def _accuracy(model: nn.Module, dataset: Dataset[Any], device: str, batch_size: int) -> float:
    """Fraction correct over `dataset`, without building a full predictions frame."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device)).cpu()
            y_hat = logits.argmax(dim=1)
            correct += int((y_hat == batch["y"]).sum())
            total += len(batch["y"])
    return correct / total


def _epoch_train_loader(
    dataset: ImageDataset, seed: int, epoch: int, batch_size: int, num_workers: int
) -> DataLoader[Any]:
    """A fresh shuffled `DataLoader` for one epoch, seeded only by `(seed, epoch)` (see module
    docstring) -- never reused across epochs, so nothing needs to be snapshotted to resume."""
    epoch_seed = int(make_rng(seed, "erm_shuffle", epoch).integers(0, 2**31 - 1))
    generator = torch.Generator().manual_seed(epoch_seed)
    return DataLoader(
        cast(Dataset[Any], dataset),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
    )


def train_erm(
    config: TrainConfig,
    config_dict: dict[str, Any],
    datasets: Mapping[str, ImageDataset],
    run_dir: Path,
    run_id: str,
    *,
    stop_after_epoch: int | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Train, checkpoint, predict on every split, and write a manifest. Returns the history table.

    `datasets` must have exactly the keys in `_SPLITS` (`train`, `val_a`, `val_b`, `test`) --
    injected by the composition root (`cli.py`), never built here (label-free zone, D-003).

    Resuming is automatic and idempotent, mirroring `build/tables.py`'s pattern: if
    `run_dir/train/checkpoint_last.pt` exists with a matching config hash and `force=False`,
    training continues from `checkpoint["epoch"] + 1`; `force=True` restarts from epoch 0.
    `stop_after_epoch` deliberately checkpoints and returns early without writing final
    predictions/manifest -- it simulates an interrupted GPU session (used by the exact-resume test
    and available to `run-jobs` for a clean checkpoint before a wall-clock limit).
    """
    if set(datasets) != set(_SPLITS):
        raise ValueError(f"datasets must have exactly the keys {_SPLITS}, got {sorted(datasets)}")

    start_time = time.monotonic()
    device = resolve_device(config.device)
    precision_mode = resolve_precision_mode(config.amp, device)
    this_config_hash = config_hash(config_dict)

    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_last_path = train_dir / "checkpoint_last.pt"
    checkpoint_best_path = train_dir / "checkpoint_best.pt"
    history_path = train_dir / "history.csv"

    # Deterministic init given `config.seed` alone (not process RNG state): required so a fresh
    # run and a resumed run's pre-checkpoint epochs are bit-identical (see module docstring).
    # `pretrained` weights themselves are already deterministic; this only matters for the
    # `Backbone.head` (always randomly initialised) and any `pretrained=None` trunk (tiny_cnn).
    torch.manual_seed(config.seed)
    model = build_backbone(config.arch, num_classes=2, pretrained=config.pretrained).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(precision_mode == "fp16_amp"))

    start_epoch = 0
    best_val_acc = -1.0
    history_rows: list[dict[str, float]] = []
    if checkpoint_last_path.exists() and not force:
        checkpoint = torch.load(checkpoint_last_path, map_location=device)
        if checkpoint["config_hash"] != this_config_hash:
            raise ValueError(
                f"{checkpoint_last_path} was written for a different config "
                f"({checkpoint['config_hash']} != {this_config_hash}); pass force=True to restart"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_acc = checkpoint["best_val_acc"]
        start_epoch = checkpoint["epoch"] + 1
        if history_path.exists():
            history_rows = cast(
                "list[dict[str, float]]", pd.read_csv(history_path).to_dict("records")
            )

    val_dataset: ConcatDataset[Any] = ConcatDataset(
        [cast(Dataset[Any], datasets["val_a"]), cast(Dataset[Any], datasets["val_b"])]
    )

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.monotonic()
        model.train()
        loader = _epoch_train_loader(
            datasets["train"], config.seed, epoch, config.batch_size, config.num_workers
        )
        running_loss, running_correct, n = 0.0, 0, 0

        for batch in loader:
            images = batch["image"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()

            if precision_mode == "fp16_amp":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * len(y)
            running_correct += int((logits.argmax(dim=1) == y).sum())
            n += len(y)

        val_avg_acc = _accuracy(model, val_dataset, device, config.batch_size)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / n,
                "train_acc": running_correct / n,
                "val_avg_acc": val_avg_acc,
                "duration_s": time.monotonic() - epoch_start,
            }
        )
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        checkpoint_state = {
            "epoch": epoch,
            "config_hash": this_config_hash,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": max(best_val_acc, val_avg_acc),
        }
        torch.save(checkpoint_state, checkpoint_last_path)
        if val_avg_acc > best_val_acc:
            best_val_acc = val_avg_acc
            torch.save(checkpoint_state, checkpoint_best_path)

        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            return pd.DataFrame(history_rows)

    best_checkpoint = torch.load(checkpoint_best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    for split in _SPLITS:
        predictions = _predict(model, datasets[split], device, config.batch_size)
        predictions.to_parquet(predictions_dir / f"{split}.parquet", index=False)

    manifest = new_manifest(
        run_id=run_id,
        stage="train",
        config=config_dict,
        seed=config.seed,
        inputs={},
        duration_s=time.monotonic() - start_time,
        device=device,
        precision_mode=precision_mode,
    )
    write_manifest(run_dir / "manifest.json", manifest)

    return pd.DataFrame(history_rows)
