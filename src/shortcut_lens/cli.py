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

import time
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from shortcut_lens.artifacts import LocalStore, build_dir
from shortcut_lens.build.pets import PlantedPetsConfig, build_planted_pets
from shortcut_lens.build.synthetic_shapes import SyntheticShapesConfig, build_synthetic_shapes
from shortcut_lens.build.tables import write_build_tables, write_data_report
from shortcut_lens.build.waterbirds import WaterbirdsConfig, build_waterbirds
from shortcut_lens.config import load_yaml_composed, short_hash
from shortcut_lens.data.public import read_public_table
from shortcut_lens.oracle.groups import read_oracle_table

_CACHE_ROOT = Path.home() / ".cache" / "shortcut-lens"

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


def _resolve_and_build(
    config_path: Path, force: bool
) -> tuple[str, str, pd.DataFrame, pd.DataFrame, Path]:
    """Load a dataset config and build it, or reuse an existing build with a matching hash.

    Returns `(dataset, build_hash, public_table, oracle_table, image_root)`. `image_root` is
    where `image_ref` values resolve: the build's own image cache for synthetic_shapes/
    planted_pets, or the extracted download directory for waterbirds.
    """
    raw = load_yaml_composed(config_path)
    dataset = raw.get("dataset")
    if dataset not in {"synthetic_shapes", "planted_pets", "waterbirds"}:
        raise typer.BadParameter(
            "config must set dataset: one of synthetic_shapes/planted_pets/waterbirds, "
            f"got {dataset!r}"
        )
    build_config = {k: v for k, v in raw.items() if k != "dataset"}
    build_hash = short_hash(build_config)

    store = LocalStore()
    directory = build_dir(store, dataset, build_hash)
    already_built = not force and (directory / "public.parquet").exists()

    if dataset == "synthetic_shapes":
        shapes_cfg = SyntheticShapesConfig.model_validate(build_config)
        image_root = directory / "images"
        if already_built:
            public = read_public_table(directory / "public.parquet")
            oracle = read_oracle_table(directory / "oracle.parquet")
        else:
            start = time.monotonic()
            public, oracle = build_synthetic_shapes(shapes_cfg, image_root)
            write_build_tables(
                store,
                dataset,
                build_hash,
                public,
                oracle,
                config=build_config,
                seed=shapes_cfg.build_seed,
                duration_s=time.monotonic() - start,
            )
    elif dataset == "planted_pets":
        pets_cfg = PlantedPetsConfig.model_validate(build_config)
        image_root = directory / "images"
        if already_built:
            public = read_public_table(directory / "public.parquet")
            oracle = read_oracle_table(directory / "oracle.parquet")
        else:
            torchvision_root = _CACHE_ROOT / "torchvision"
            start = time.monotonic()
            public, oracle = build_planted_pets(pets_cfg, torchvision_root, image_root)
            write_build_tables(
                store,
                dataset,
                build_hash,
                public,
                oracle,
                config=build_config,
                seed=pets_cfg.build_seed,
                duration_s=time.monotonic() - start,
            )
    else:
        birds_cfg = WaterbirdsConfig.model_validate(build_config)
        image_root = directory / "images"
        if already_built:
            public = read_public_table(directory / "public.parquet")
            oracle = read_oracle_table(directory / "oracle.parquet")
        else:
            start = time.monotonic()
            public, oracle = build_waterbirds(birds_cfg, _CACHE_ROOT / "downloads", image_root)
            write_build_tables(
                store,
                dataset,
                build_hash,
                public,
                oracle,
                config=build_config,
                seed=birds_cfg.build_seed,
                duration_s=time.monotonic() - start,
            )

    return dataset, build_hash, public, oracle, image_root


ForceOption = Annotated[
    bool, typer.Option("--force", help="Rebuild even if a matching cached build exists.")
]


@app.command()
def build(config: ConfigOption, force: ForceOption = False) -> None:
    """Construct dataset tables and caches (oracle zone). See `build/tables.py`."""
    dataset, build_hash, public, _, _ = _resolve_and_build(config, force)
    typer.echo(
        f"Built {dataset} ({build_hash}): {len(public)} examples "
        f"-> artifacts/builds/{dataset}/{build_hash}/"
    )


@data_app.command("report")
def data_report(config: ConfigOption, force: ForceOption = False) -> None:
    """Write group counts and a sample image grid for a dataset (oracle zone)."""
    dataset, _, public, oracle, image_root = _resolve_and_build(config, force)
    counts_path, figure_path = write_data_report(
        public, oracle, image_root, dataset, Path("results"), Path("reports") / "figures"
    )
    typer.echo(f"Wrote {counts_path} and {figure_path}")


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
