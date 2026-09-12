"""Integration test: `slens build` and `slens data report` for synthetic_shapes, end to end.

Runs the CLI exactly as a user would -- see docs/TESTING.md's integration layer. Uses a scratch
directory *inside* the repo (under the gitignored `artifacts/`) rather than a system tmp dir,
because manifest writing needs `git rev-parse HEAD` to succeed, which requires being inside the
repo's working tree.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from shortcut_lens.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def scratch_dir() -> Iterator[Path]:
    path = REPO_ROOT / "artifacts" / f"_test_scratch_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_config(path: Path) -> Path:
    config_path = path / "synthetic_shapes.yaml"
    config_path.write_text(
        "dataset: synthetic_shapes\n"
        "variant: dot\n"
        "rho: 0.95\n"
        "build_seed: 0\n"
        "n_train: 40\n"
        "n_val: 40\n"
        "n_test: 40\n"
    )
    return config_path


def test_build_then_data_report(scratch_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(scratch_dir)
    config_path = _write_config(scratch_dir)

    build_result = runner.invoke(app, ["build", "--config", str(config_path)])
    assert build_result.exit_code == 0, build_result.output
    assert "Built synthetic_shapes" in build_result.output

    report_result = runner.invoke(app, ["data", "report", "--config", str(config_path)])
    assert report_result.exit_code == 0, report_result.output

    assert (scratch_dir / "results" / "data_counts_synthetic_shapes.csv").exists()
    assert (scratch_dir / "reports" / "figures" / "samples_synthetic_shapes.png").exists()


def test_build_is_idempotent(scratch_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(scratch_dir)
    config_path = _write_config(scratch_dir)

    first = runner.invoke(app, ["build", "--config", str(config_path)])
    second = runner.invoke(app, ["build", "--config", str(config_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_hash = first.output.split("(")[1].split(")")[0]
    second_hash = second.output.split("(")[1].split(")")[0]
    assert first_hash == second_hash
