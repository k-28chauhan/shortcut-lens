"""Tests for shortcut_lens.jobs: loading, dry-run planning, stop-on-first-failure."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from shortcut_lens.jobs import JobsFile, JobStep, load_jobs, run_jobs, step_command


def _jobs_file() -> JobsFile:
    return JobsFile(
        jobs=[
            JobStep(stage="train", config="configs/experiments/e1.yaml", seed=0),
            JobStep(stage="embed", config="configs/experiments/e1.yaml", seed=0, space="model"),
            JobStep(stage="embed", config="configs/experiments/e1.yaml", seed=0, space="clip"),
            JobStep(stage="reliance", config="configs/experiments/e1.yaml", seed=0),
        ]
    )


def test_load_jobs_parses_yaml(tmp_path: Path) -> None:
    path = tmp_path / "jobs.yaml"
    path.write_text(
        yaml.dump(
            {
                "jobs": [
                    {"stage": "train", "config": "configs/experiments/e1.yaml", "seed": 0},
                ]
            }
        )
    )
    jobs_file = load_jobs(path)
    assert jobs_file.jobs == [JobStep(stage="train", config="configs/experiments/e1.yaml", seed=0)]


def test_step_command_includes_seed_and_space() -> None:
    step = JobStep(stage="embed", config="c.yaml", seed=1, space="clip")
    assert step_command(step) == [
        "slens",
        "embed",
        "--config",
        "c.yaml",
        "--seed",
        "1",
        "--space",
        "clip",
    ]


def test_step_command_omits_absent_optional_fields() -> None:
    step = JobStep(stage="train", config="c.yaml")
    assert step_command(step) == ["slens", "train", "--config", "c.yaml"]


def test_run_jobs_dry_run_returns_plan_without_running(tmp_path: Path) -> None:
    calls: list[Sequence[str]] = []
    commands = run_jobs(_jobs_file(), runner=lambda cmd: calls.append(cmd) or 0, dry_run=True)

    assert calls == []
    assert len(commands) == 4
    assert commands[0][:2] == ["slens", "train"]


def test_run_jobs_runs_every_step_in_order_when_all_succeed() -> None:
    calls: list[Sequence[str]] = []

    def runner(command: Sequence[str]) -> int:
        calls.append(command)
        return 0

    executed = run_jobs(_jobs_file(), runner=runner)

    assert len(executed) == 4
    assert [c[1] for c in calls] == ["train", "embed", "embed", "reliance"]


def test_run_jobs_stops_at_first_failure() -> None:
    calls: list[Sequence[str]] = []

    def runner(command: Sequence[str]) -> int:
        calls.append(command)
        return 1 if command[1] == "embed" and "--space" in command and "model" in command else 0

    with pytest.raises(RuntimeError, match="job step failed"):
        run_jobs(_jobs_file(), runner=runner)

    # train succeeded, first embed (model) failed -- reliance must never have been attempted
    assert len(calls) == 2
