"""Integration test behind `make smoke`: build -> train -> embed (model + a fake CLIP), CPU only.

The model-space path goes through the real CLI (`slens train`, `slens embed --space model`),
exactly as a user would run it. The CLIP-space path uses the `fake_clip` fixture (deterministic,
no download, docs/TESTING.md §2) instead of the real `slens embed --space clip` -- that command's
real-CLIP path is already covered by `tests/unit/test_clip_space.py`'s `network`-marked test; this
test's job is proving the *plumbing*, fast and offline, end to end (docs/PLAN.md M3's `make smoke`).
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shortcut_lens.cli import _resolve_and_build, _split_datasets, app
from shortcut_lens.data.transforms import eval_transform
from shortcut_lens.embeddings.clip_space import embed_clip_space
from shortcut_lens.embeddings.store import read_embeddings, write_embeddings

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = REPO_ROOT / "configs" / "experiments" / "smoke.yaml"


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    path = REPO_ROOT / "artifacts" / f"_test_scratch_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_smoke_build_train_embed_model_and_fake_clip(
    scratch_dir: Path, monkeypatch: pytest.MonkeyPatch, fake_clip: object
) -> None:
    monkeypatch.chdir(scratch_dir)

    train_result = runner.invoke(app, ["train", "--config", str(SMOKE_CONFIG)])
    assert train_result.exit_code == 0, train_result.output
    run_id = train_result.output.split(":")[0].removeprefix("Trained ")

    embed_result = runner.invoke(app, ["embed", "--config", str(SMOKE_CONFIG), "--space", "model"])
    assert embed_result.exit_code == 0, embed_result.output

    run_directory = scratch_dir / "artifacts" / "runs" / run_id
    for split in ("train", "val_a", "val_b", "test"):
        assert (run_directory / "predictions" / f"{split}.parquet").exists()
        vectors, ids = read_embeddings(run_directory / "embeddings" / "model", split)
        assert len(vectors) == len(ids) > 0

    # fake-CLIP path: same datasets, transformed with eval_transform (a stand-in for CLIP's own
    # preprocess -- the fake model doesn't care about CLIP-specific normalisation).
    _dataset, _hash, public, oracle, image_root, patch_spec, build_seed = _resolve_and_build(
        SMOKE_CONFIG, force=False
    )
    transform_by_split = {split: eval_transform() for split in ("train", "val_a", "val_b", "test")}
    datasets = _split_datasets(
        public, oracle, image_root, patch_spec, build_seed, transform_by_split
    )

    for split, dataset in datasets.items():
        vectors, ids = embed_clip_space(fake_clip, dataset, device="cpu", batch_size=16)
        write_embeddings(run_directory / "embeddings" / "clip", split, vectors, ids)
        assert len(vectors) == len(dataset)
        assert ids == [dataset[i]["example_id"] for i in range(len(dataset))]
