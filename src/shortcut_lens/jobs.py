"""`run-jobs`: a sequential stage runner for GPU sessions, resumable after disconnection.

Concept: a GPU handoff runs a fixed list of stage commands (e.g. train then embed then reliance
for every seed of a config). Kaggle/Colab sessions disconnect; resuming just means re-running the
same jobs file. `run_jobs` itself tracks nothing about what is already done -- idempotency is each
stage's own responsibility (`training.erm.train_erm`'s checkpoint-driven resume; the `embed`/
`reliance` CLI commands' own config-hash/manifest skip-if-done check, mirroring
`build/tables.py`'s "already_built" pattern, ARCHITECTURE §5). What `run_jobs` does own: running
every step in order, stopping at the first failure rather than continuing past a broken stage
(docs/TESTING.md), and `--dry-run`, so a human can see the planned command sequence -- and its
cost -- before spending GPU quota on it (docs/RUNBOOK_GPU.md).

Pipeline position: shared (unconstrained by the group-label-firewall contract -- it only ever
shells out to other `slens` subcommands, never touches label-free or oracle data itself). Entry
point for `slens run-jobs jobs/<file>.yaml`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

from shortcut_lens.config import StrictBaseModel, load_yaml_composed


class JobStep(StrictBaseModel):
    """One `slens <stage> ...` invocation. `seed`/`space` are only meaningful for some stages."""

    stage: Literal["train", "embed", "reliance"]
    config: str
    seed: int | None = None
    space: Literal["model", "clip"] | None = None


class JobsFile(StrictBaseModel):
    """A jobs YAML: an ordered list of steps, run in sequence."""

    jobs: list[JobStep]


def load_jobs(path: str | Path) -> JobsFile:
    """Load and validate a jobs YAML file."""
    return JobsFile.model_validate(load_yaml_composed(path))


def step_command(step: JobStep) -> list[str]:
    """The `slens <stage> ...` argv this step maps to -- used for both `--dry-run` and execution."""
    command = ["slens", step.stage, "--config", step.config]
    if step.seed is not None:
        command += ["--seed", str(step.seed)]
    if step.space is not None:
        command += ["--space", step.space]
    return command


def run_jobs(
    jobs_file: JobsFile,
    runner: Callable[[Sequence[str]], int],
    *,
    dry_run: bool = False,
) -> list[list[str]]:
    """Run every step's command via `runner` (given the argv, returns the process exit code).

    Stops at the first step whose `runner` return code is non-zero, raising rather than
    continuing to later steps. With `dry_run=True`, no command is run at all -- the full planned
    list is returned as-is.

    Returns the commands that were (or, under `dry_run`, would be) run, in order.
    """
    commands = [step_command(step) for step in jobs_file.jobs]
    if dry_run:
        return commands

    executed: list[list[str]] = []
    for command in commands:
        executed.append(command)
        exit_code = runner(command)
        if exit_code != 0:
            raise RuntimeError(f"job step failed (exit {exit_code}): {' '.join(command)}")
    return executed
