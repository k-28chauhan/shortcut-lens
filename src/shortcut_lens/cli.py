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

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import torch
import typer

from shortcut_lens.artifacts import HFHubStore, LocalStore, build_dir, run_dir
from shortcut_lens.build.cifar_pets import CifarPetsConfig, build_cifar_pets
from shortcut_lens.build.datasets import RenderedImageDataset
from shortcut_lens.build.pets import PlantedPetsConfig, build_planted_pets
from shortcut_lens.build.planting import PatchSpec
from shortcut_lens.build.synthetic_shapes import SyntheticShapesConfig, build_synthetic_shapes
from shortcut_lens.build.tables import write_build_tables, write_data_report
from shortcut_lens.build.waterbirds import WaterbirdsConfig, build_waterbirds
from shortcut_lens.config import config_hash, load_yaml_composed, short_hash
from shortcut_lens.data.public import read_public_table
from shortcut_lens.data.transforms import eval_transform, train_transform
from shortcut_lens.embeddings.clip_space import ClipEmbedConfig, embed_clip_space, load_clip
from shortcut_lens.embeddings.model_space import embed_model_space
from shortcut_lens.embeddings.store import write_embeddings
from shortcut_lens.evaluation.core import group_metrics_table
from shortcut_lens.jobs import load_jobs
from shortcut_lens.jobs import run_jobs as execute_jobs
from shortcut_lens.manifest import new_manifest, read_manifest, validate_manifest, write_manifest
from shortcut_lens.models.backbones import build_backbone
from shortcut_lens.oracle.groups import read_oracle_table
from shortcut_lens.training.erm import TrainConfig, resolve_device, train_erm
from shortcut_lens.verification.reliance import compute_reliance

_CACHE_ROOT = Path.home() / ".cache" / "shortcut-lens"
_SPLITS = ("train", "val_a", "val_b", "test")
_NON_DATASET_CONFIG_KEYS = {"dataset", "train", "embed", "device", "experiment_id"}

app = typer.Typer(
    name="slens",
    help="shortcut-lens: discover, confirm, name, verify and fix a classifier's failure slices.",
    no_args_is_help=True,
)
data_app = typer.Typer(help="Dataset inspection commands (oracle zone).")
app.add_typer(data_app, name="data")

ConfigOption = Annotated[Path, typer.Option("--config", help="Path to a YAML config file.")]
SeedOption = Annotated[int | None, typer.Option(help="Override the config's training seed.")]


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
) -> tuple[str, str, pd.DataFrame, pd.DataFrame, Path, PatchSpec | None, int]:
    """Load a dataset config and build it, or reuse an existing build with a matching hash.

    Returns `(dataset, build_hash, public_table, oracle_table, image_root, patch_spec,
    build_seed)`. `image_root` is where `image_ref` values resolve. `patch_spec` is `None` for
    synthetic_shapes/waterbirds (no dynamically-applied patch, D-003's "pure file reader" case).

    `config_path` may be a bare dataset-build config (M1) or a full experiment config that also
    has `train:`/`embed:`/`device:`/`experiment_id:` sections (M3, ARCHITECTURE §6) -- only the
    dataset-construction keys feed `build_hash` and the per-dataset Pydantic model, so two
    experiment configs that only differ in `train:` reuse the same dataset build.
    """
    raw = load_yaml_composed(config_path)
    dataset = raw.get("dataset")
    if dataset not in {"synthetic_shapes", "planted_pets", "waterbirds", "planted_cifar_pets"}:
        raise typer.BadParameter(
            "config must set dataset: one of synthetic_shapes/planted_pets/waterbirds/"
            f"planted_cifar_pets, got {dataset!r}"
        )
    build_config = {k: v for k, v in raw.items() if k not in _NON_DATASET_CONFIG_KEYS}
    build_hash = short_hash(build_config)

    store = LocalStore()
    directory = build_dir(store, dataset, build_hash)
    already_built = not force and (directory / "public.parquet").exists()
    patch_spec: PatchSpec | None = None

    if dataset == "synthetic_shapes":
        shapes_cfg = SyntheticShapesConfig.model_validate(build_config)
        image_root = directory / "images"
        build_seed = shapes_cfg.build_seed
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
                seed=build_seed,
                duration_s=time.monotonic() - start,
            )
    elif dataset == "planted_pets":
        pets_cfg = PlantedPetsConfig.model_validate(build_config)
        image_root = directory / "images"
        build_seed = pets_cfg.build_seed
        patch_spec = pets_cfg.patch_spec
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
                seed=build_seed,
                duration_s=time.monotonic() - start,
            )
    elif dataset == "planted_cifar_pets":
        cifar_cfg = CifarPetsConfig.model_validate(build_config)
        image_root = directory / "images"
        build_seed = cifar_cfg.build_seed
        patch_spec = cifar_cfg.patch_spec
        if already_built:
            public = read_public_table(directory / "public.parquet")
            oracle = read_oracle_table(directory / "oracle.parquet")
        else:
            torchvision_root = _CACHE_ROOT / "torchvision"
            start = time.monotonic()
            public, oracle = build_cifar_pets(cifar_cfg, torchvision_root, image_root)
            write_build_tables(
                store,
                dataset,
                build_hash,
                public,
                oracle,
                config=build_config,
                seed=build_seed,
                duration_s=time.monotonic() - start,
            )
    else:
        birds_cfg = WaterbirdsConfig.model_validate(build_config)
        image_root = directory / "images"
        build_seed = birds_cfg.build_seed
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
                seed=build_seed,
                duration_s=time.monotonic() - start,
            )

    return dataset, build_hash, public, oracle, image_root, patch_spec, build_seed


def _split_datasets(
    public: pd.DataFrame,
    oracle: pd.DataFrame,
    image_root: Path,
    patch_spec: PatchSpec | None,
    build_seed: int,
    transform_by_split: Mapping[str, Callable[[Any], Any]],
) -> dict[str, RenderedImageDataset]:
    """One `RenderedImageDataset` per split, transformed and (for planted data) patch-aware."""
    oracle_by_id = oracle.set_index("example_id") if patch_spec is not None else None
    datasets: dict[str, RenderedImageDataset] = {}
    for split in _SPLITS:
        split_public = public[public["split"] == split].reset_index(drop=True)
        patch_flags = None
        if patch_spec is not None and oracle_by_id is not None:
            patch_flags = oracle_by_id.loc[split_public["example_id"], "has_patch"].to_dict()
        datasets[split] = RenderedImageDataset(
            split_public,
            image_root,
            patch_spec=patch_spec,
            patch_flags=patch_flags,
            build_seed=build_seed,
            transform=transform_by_split[split],
        )
    return datasets


def _run_id(raw: dict[str, Any], dataset: str, seed: int) -> str:
    """`<exp>-<dataset>-<cfghash8>-s<seed>` (ARCHITECTURE §5). `experiment_id` defaults to
    `dataset` for a bare train-only config (e.g. `make smoke`'s), not a full experiment config."""
    experiment_id = raw.get("experiment_id", dataset)
    return f"{experiment_id}-{dataset}-{short_hash(raw)}-s{seed}"


def _resolved_train_config(
    raw: dict[str, Any], seed: int | None
) -> tuple[TrainConfig, dict[str, Any]]:
    train_dict = dict(raw.get("train", raw))  # bare train config files have no "train:" wrapper
    if "device" not in train_dict and "device" in raw:
        train_dict["device"] = raw["device"]
    if seed is not None:
        train_dict["seed"] = seed
    return TrainConfig.model_validate(train_dict), train_dict


ForceOption = Annotated[
    bool, typer.Option("--force", help="Rebuild even if a matching cached build exists.")
]


@app.command()
def build(config: ConfigOption, force: ForceOption = False) -> None:
    """Construct dataset tables and caches (oracle zone). See `build/tables.py`."""
    dataset, build_hash, public, _, _, _, _ = _resolve_and_build(config, force)
    typer.echo(
        f"Built {dataset} ({build_hash}): {len(public)} examples "
        f"-> artifacts/builds/{dataset}/{build_hash}/"
    )


@data_app.command("report")
def data_report(config: ConfigOption, force: ForceOption = False) -> None:
    """Write group counts and a sample image grid for a dataset (oracle zone)."""
    dataset, _, public, oracle, image_root, _, _ = _resolve_and_build(config, force)
    counts_path, figure_path = write_data_report(
        public, oracle, image_root, dataset, Path("results"), Path("reports") / "figures"
    )
    typer.echo(f"Wrote {counts_path} and {figure_path}")


@app.command()
def train(
    config: ConfigOption,
    seed: SeedOption = None,
    restart: Annotated[
        bool, typer.Option("--restart", help="Ignore any existing checkpoint; train from epoch 0.")
    ] = False,
) -> None:
    """ERM fine-tune + predictions for every split (label-free zone). See `training/erm.py`."""
    raw = load_yaml_composed(config)
    dataset, _, public, oracle, image_root, patch_spec, build_seed = _resolve_and_build(
        config, force=False
    )
    train_cfg, train_dict = _resolved_train_config(raw, seed)

    transform_by_split = {
        "train": train_transform(),
        "val_a": eval_transform(),
        "val_b": eval_transform(),
        "test": eval_transform(),
    }
    datasets = _split_datasets(
        public, oracle, image_root, patch_spec, build_seed, transform_by_split
    )

    run_id = _run_id(raw, dataset, train_cfg.seed)
    run_directory = run_dir(LocalStore(), run_id)
    history = train_erm(train_cfg, train_dict, datasets, run_directory, run_id, force=restart)
    typer.echo(f"Trained {run_id}: {len(history)} epoch row(s) -> {run_directory}")


@app.command()
def embed(
    config: ConfigOption,
    space: Annotated[str, typer.Option(help="Embedding space: 'model' or 'clip'.")] = "model",
    seed: SeedOption = None,
    force: ForceOption = False,
) -> None:
    """Compute embeddings in the model or CLIP space (label-free zone)."""
    if space not in {"model", "clip"}:
        raise typer.BadParameter("space must be 'model' or 'clip'")

    raw = load_yaml_composed(config)
    dataset, _, public, oracle, image_root, patch_spec, build_seed = _resolve_and_build(
        config, force=False
    )
    train_cfg, _ = _resolved_train_config(raw, seed)
    run_id = _run_id(raw, dataset, train_cfg.seed)
    run_directory = run_dir(LocalStore(), run_id)

    embed_config = {"space": space, **raw.get("embed", {})}
    this_hash = config_hash(embed_config)
    embeddings_dir = run_directory / "embeddings" / space
    manifest_path = embeddings_dir / "manifest.json"
    if (
        not force
        and manifest_path.exists()
        and read_manifest(manifest_path).config_hash == this_hash
    ):
        typer.echo(f"{space} embeddings for {run_id} already computed -- skipping (use --force)")
        return

    start = time.monotonic()
    if space == "model":
        device = resolve_device(train_cfg.device)
        backbone = build_backbone(train_cfg.arch, num_classes=2, pretrained=train_cfg.pretrained)
        backbone = backbone.to(device)
        checkpoint = torch.load(run_directory / "train" / "checkpoint_best.pt", map_location=device)
        backbone.load_state_dict(checkpoint["model_state_dict"])
        transform_by_split: Mapping[str, Callable[[Any], Any]] = {
            split: eval_transform() for split in _SPLITS
        }
        datasets = _split_datasets(
            public, oracle, image_root, patch_spec, build_seed, transform_by_split
        )
        for split in _SPLITS:
            vectors, ids = embed_model_space(
                backbone, datasets[split], device, train_cfg.batch_size
            )
            write_embeddings(embeddings_dir, split, vectors, ids)
    else:
        clip_cfg = ClipEmbedConfig.model_validate(raw.get("embed", {}).get("clip", {}))
        device = resolve_device(raw.get("device", "auto"))
        model, preprocess = load_clip(clip_cfg, device)
        transform_by_split = {split: preprocess for split in _SPLITS}
        datasets = _split_datasets(
            public, oracle, image_root, patch_spec, build_seed, transform_by_split
        )
        for split in _SPLITS:
            vectors, ids = embed_clip_space(model, datasets[split], device, batch_size=32)
            write_embeddings(embeddings_dir, split, vectors, ids)

    manifest = new_manifest(
        run_id=run_id,
        stage=f"embed_{space}",
        config=embed_config,
        seed=train_cfg.seed,
        inputs={},
        duration_s=time.monotonic() - start,
        device=device,
    )
    write_manifest(manifest_path, manifest)
    typer.echo(f"Computed {space} embeddings for {run_id} -> {embeddings_dir}")


@app.command()
def reliance(config: ConfigOption, seed: SeedOption = None) -> None:
    """Measure `R_net` for a planted run (oracle zone). See `verification/reliance.py`."""
    raw = load_yaml_composed(config)
    dataset, _, public, oracle, image_root, patch_spec, build_seed = _resolve_and_build(
        config, force=False
    )
    if dataset not in {"planted_pets", "planted_cifar_pets"} or patch_spec is None:
        raise typer.BadParameter(
            "reliance only applies to a planted shortcut we control "
            "(planted_pets/planted_cifar_pets, FR-R1)"
        )

    train_cfg, _ = _resolved_train_config(raw, seed)
    device = resolve_device(train_cfg.device)
    run_id = _run_id(raw, dataset, train_cfg.seed)
    run_directory = run_dir(LocalStore(), run_id)

    backbone = build_backbone(train_cfg.arch, num_classes=2, pretrained=train_cfg.pretrained)
    backbone = backbone.to(device)
    checkpoint = torch.load(run_directory / "train" / "checkpoint_best.pt", map_location=device)
    backbone.load_state_dict(checkpoint["model_state_dict"])

    predictions = pd.read_parquet(run_directory / "predictions" / "test.parquet")
    test_public = public[public["split"] == "test"]
    joined = test_public.merge(oracle, on="example_id").merge(
        predictions[["example_id", "correct"]], on="example_id"
    )

    result = compute_reliance(
        backbone, device, joined, image_root, patch_spec, build_seed, seed=train_cfg.seed
    )
    result.to_parquet(run_directory / "reliance.parquet", index=False)
    r_net = result.loc[result["direction"] == "net", "break_rate"].iloc[0]
    typer.echo(f"R_net for {run_id}: {r_net:.3f} -> {run_directory / 'reliance.parquet'}")


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
    seed: SeedOption = None,
    final: Annotated[
        bool, typer.Option("--final", help="Run on the test split; appends to the final eval log.")
    ] = False,
    rerun_reason: Annotated[
        str | None,
        typer.Option("--rerun-reason", help="Required to repeat a final eval of the same config."),
    ] = None,
) -> None:
    """Compute metrics tables against ground truth (oracle zone). Draft (val only) as of M3;
    `--final` (test split, append-only log, rerun guard) lands in M7."""
    if final:
        _not_implemented("evaluate --final", "M7")

    raw = load_yaml_composed(config)
    dataset, _, public, oracle, _, _, _ = _resolve_and_build(config, force=False)
    train_cfg, _ = _resolved_train_config(raw, seed)
    run_id = _run_id(raw, dataset, train_cfg.seed)
    run_directory = run_dir(LocalStore(), run_id)

    tables = []
    for split in ("val_a", "val_b"):
        predictions = pd.read_parquet(run_directory / "predictions" / f"{split}.parquet")
        split_ids = public.loc[public["split"] == split, "example_id"]
        split_oracle = oracle[oracle["example_id"].isin(split_ids)]
        table = group_metrics_table(
            predictions[["example_id", "y", "y_hat"]],
            split_oracle[["example_id", "group", "group_name"]],
            seed=train_cfg.seed,
        )
        table.insert(0, "split", split)
        tables.append(table)

    result = pd.concat(tables, ignore_index=True)
    out_path = Path("results") / f"draft_eval_{run_id}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    typer.echo(f"Wrote draft evaluation (val_a/val_b only) -> {out_path}")


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
def run_jobs_command(
    jobs_path: Annotated[Path, typer.Argument(help="Path to a jobs YAML file.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the planned commands without running them.")
    ] = False,
) -> None:
    """Run a list of stage commands sequentially, resuming after interruption. See `jobs.py`."""
    jobs_file = load_jobs(jobs_path)

    def _subprocess_runner(command: Sequence[str]) -> int:
        return subprocess.run(command).returncode

    commands = execute_jobs(jobs_file, _subprocess_runner, dry_run=dry_run)
    if dry_run:
        for command in commands:
            typer.echo(" ".join(command))
    else:
        typer.echo(f"Ran {len(commands)} job step(s).")


@app.command(name="pull-artifacts")
def pull_artifacts(
    run_ids: Annotated[list[str], typer.Option("--run-ids", help="Run ids to pull from the Hub.")],
    repo_id: Annotated[str, typer.Option("--repo-id", help="HF Hub dataset repo id.")],
) -> None:
    """Download finished run artefacts from the Hugging Face Hub store."""
    store = HFHubStore(repo_id=repo_id)
    for one_run_id in run_ids:
        store.pull_run(one_run_id)
        typer.echo(f"Pulled {one_run_id}")


@app.command(name="validate-run")
def validate_run(run_id: Annotated[str, typer.Argument(help="Run id to validate.")]) -> None:
    """Check a run's manifest: commit, config hash, and that referenced files exist."""
    directory = run_dir(LocalStore(), run_id)
    manifest = read_manifest(directory / "manifest.json")
    problems = validate_manifest(manifest, run_dir=directory)
    if problems:
        for problem in problems:
            typer.secho(f"- {problem}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{run_id}: manifest valid.")


if __name__ == "__main__":
    app()
