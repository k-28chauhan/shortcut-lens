"""Composition root: the Typer CLI `slens`. The only module that may import from both zones.

Concept: `cli.py` builds datasets in the oracle zone (`build/`) and injects them into label-free
stages as plain `ImageDataset` objects exposing only `{"image", "y", "example_id"}` -- the
mechanism that makes the label-free/oracle firewall enforceable (D-003). Every subcommand
corresponds to one pipeline stage (docs/ARCHITECTURE.md §7); each is idempotent, writes a
manifest, and lands milestone by milestone (see docs/PLAN.md). At M0 every command is a
documented stub: `--help` lists the full interface before any of it is implemented.

Pipeline position: composition root. Entry point: `slens <command>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="slens",
    help="shortcut-lens: discover, confirm, name, verify and fix a classifier's failure slices.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Dataset inspection commands (oracle zone).")
app.add_typer(data_app, name="data")

ConfigOption = Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")]


def _not_implemented(command: str, milestone: str) -> None:
    """Fail loudly with the milestone a stub command is scheduled to land in."""
    typer.secho(
        f"'{command}' is not implemented yet -- lands in {milestone} (see docs/PLAN.md).",
        fg=typer.colors.RED,
        err=True,
    )
    raise typer.Exit(code=1)


@app.command()
def build(config: ConfigOption) -> None:
    """Construct dataset tables and caches (oracle zone). See `build/tables.py`."""
    _not_implemented("build", "M1")


@data_app.command("report")
def data_report(config: ConfigOption) -> None:
    """Write group counts and a sample image grid for a dataset (oracle zone)."""
    _not_implemented("data report", "M1")


@app.command()
def train(
    config: ConfigOption,
    seed: Annotated[int | None, typer.Option(help="Override the config's training seed.")] = None,
) -> None:
    """ERM fine-tune + predictions for every split (label-free zone). See `training/erm.py`."""
    _not_implemented("train", "M3")


@app.command()
def embed(
    config: ConfigOption,
    space: Annotated[str, typer.Option(help="Embedding space: 'model' or 'clip'.")] = "model",
) -> None:
    """Compute embeddings in the model or CLIP space (label-free zone)."""
    if space not in {"model", "clip"}:
        raise typer.BadParameter("space must be 'model' or 'clip'")
    _not_implemented("embed", "M3")


@app.command()
def reliance(config: ConfigOption) -> None:
    """Measure `R_net` for a planted run (oracle zone). See `verification/reliance.py`."""
    _not_implemented("reliance", "M3")


@app.command()
def discover(config: ConfigOption) -> None:
    """Fit slicers on val_a and score every split (label-free zone)."""
    _not_implemented("discover", "M4")


@app.command()
def confirm(config: ConfigOption) -> None:
    """Test discovered slices on val_b with BH correction and rank them (label-free zone)."""
    _not_implemented("confirm", "M4")


@app.command()
def name(
    config: ConfigOption,
    namer: Annotated[str, typer.Option(help="Naming method: 'vocab' or 'caption_keywords'.")] = (
        "vocab"
    ),
) -> None:
    """Name confirmed slices in plain English (label-free zone)."""
    if namer not in {"vocab", "caption_keywords"}:
        raise typer.BadParameter("namer must be 'vocab' or 'caption_keywords'")
    _not_implemented("name", "M5")


@app.command()
def verify(config: ConfigOption) -> None:
    """Run counterfactual interventions on confirmed slices (oracle zone)."""
    _not_implemented("verify", "M6")


@app.command()
def mitigate(
    config: ConfigOption,
    method: Annotated[str, typer.Option("--method", help="Mitigation method.")],
) -> None:
    """Last-layer retraining: label-free methods or the oracle reference (mixed zones)."""
    _not_implemented("mitigate", "M7")


@app.command()
def evaluate(
    config: ConfigOption,
    final: Annotated[
        bool, typer.Option("--final", help="Run on the test split; appends to the final eval log.")
    ] = False,
    rerun_reason: Annotated[
        str | None,
        typer.Option("--rerun-reason", help="Required to repeat a final eval of the same config."),
    ] = None,
) -> None:
    """Compute metrics tables against ground truth (oracle zone)."""
    _not_implemented("evaluate", "M2 (draft tables) / M7 (--final)")


@app.command()
def report() -> None:
    """Regenerate every table and figure from artefacts (oracle zone)."""
    _not_implemented("report", "M8")


@app.command()
def gradcam(config: ConfigOption) -> None:
    """Grad-CAM overlays for sampled slice members, labelled as illustration only."""
    _not_implemented("gradcam", "M8")


@app.command(name="export-demo")
def export_demo() -> None:
    """Build the demo bundle for the Gradio slice explorer (oracle zone)."""
    _not_implemented("export-demo", "M8")


@app.command(name="run-jobs")
def run_jobs(jobs_file: Annotated[Path, typer.Argument(help="Path to a jobs YAML file.")]) -> None:
    """Run a list of stage commands sequentially, resuming after interruption."""
    _not_implemented("run-jobs", "M3")


@app.command(name="pull-artifacts")
def pull_artifacts(
    run_ids: Annotated[list[str], typer.Option("--run-ids", help="Run ids to pull from the Hub.")],
) -> None:
    """Download finished run artefacts from the Hugging Face Hub store."""
    _not_implemented("pull-artifacts", "M3")


@app.command(name="validate-run")
def validate_run(run_id: Annotated[str, typer.Argument(help="Run id to validate.")]) -> None:
    """Check a run's manifest: commit, config hash, and that referenced files exist."""
    _not_implemented("validate-run", "M3")


if __name__ == "__main__":
    app()
