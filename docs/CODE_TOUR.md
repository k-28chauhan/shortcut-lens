# CODE TOUR — shortcut-lens

A guided reading order through the codebase. Claude Code adds a section whenever a module lands.
Each section: what the module does, the concept behind it, key functions to read, and one question
the reader should be able to answer afterwards.

Suggested order (mirrors the pipeline):

1. Foundations — `config.py`, `seeding.py`, `manifest.py`, `artifacts.py`, `cli.py`
2. The firewall — `types.py`, the import-linter contract, `build/datasets.py`, `oracle/groups.py`
3. Data — `build/planting.py`, `build/pets.py`, `build/waterbirds.py`, `build/splits.py`
4. Measurement — `metrics.py`, `stats.py`, `evaluation/core.py`
5. Models — `models/backbones.py`, `training/erm.py`
6. Embeddings — `embeddings/*`
7. Discovery — `discovery/preprocess.py`, `confidence.py`, `error_kmeans.py`, `domino_em.py`,
   `failure_direction.py`, `confirm.py`
8. Naming — `naming/*`
9. Verification — `verification/*`
10. Mitigation — `mitigation/last_layer.py`, `label_free/*`, `selection.py`, `oracle_ref/*`
11. Reporting and app — `report/*`, `viz/*`, `app/*`

---

<!-- Template for each module section:

### `path/to/module.py`
**What it does:** …
**Concept:** … (plain English, 2–4 sentences)
**Read these functions:** `a()`, `b()`
**Check yourself:** …?
-->

## M0 — Foundations

### `src/shortcut_lens/config.py`
**What it does:** loads a YAML config, merges it with any `base:` file it points to, validates it
against a Pydantic model, and computes a canonical hash of the resolved config.
**Concept:** the config hash has to be *deterministic* -- the same logical config, however its
keys happen to be ordered, must always hash to the same value. That is what lets a later stage
ask "does an output already exist for this exact config?" (idempotency) and what makes a run id
reproducible. `StrictBaseModel` rejects unknown YAML keys so a typo in a config file fails loudly
at load time instead of being silently ignored.
**Read these functions:** `load_yaml_composed()`, `_deep_merge()`, `config_hash()`, `load_config()`.
**Check yourself:** why does `config_hash` operate on the resolved dict rather than on the
Pydantic model instance? (Hint: what would happen to old hashes if a model's defaults changed?)

### `src/shortcut_lens/seeding.py`
**What it does:** `set_all_seeds(seed)` seeds every global RNG a third-party library might reach
for (Python's `random`, numpy's legacy global state, torch CPU/CUDA). `make_rng(seed, *keys)`
returns this project's own `numpy.random.Generator`, deterministic in `(seed, *keys)`.
**Concept:** CLAUDE.md's rule "seeds are explicit parameters, never hidden globals" is really
about `make_rng`: instead of one global seed for the whole run, code asks for a generator scoped
to exactly what it's doing (e.g. `make_rng(build_seed, example_id)` for one image's patch
position), so two unrelated draws can never accidentally share a stream. `set_all_seeds` is the
one place a global seed is still needed, because we don't control what RNG torch/sklearn reach
for internally.
**Read these functions:** `set_all_seeds()`, `make_rng()`.
**Check yourself:** why does `make_rng` hash its inputs with `hashlib.sha256` instead of Python's
built-in `hash()`?

### `src/shortcut_lens/manifest.py`
**What it does:** builds, writes, reads and validates a `Manifest` -- the provenance record every
stage run writes (git commit, dirty flag, resolved config + hash, seed, inputs, package versions,
hardware, duration).
**Concept:** "every reported number is traceable" (CLAUDE.md §3 rule 7) is not a slogan here --
`validate_manifest()` actually recomputes the config hash from the stored config and compares it,
checks the git commit looks like a real sha, and checks every referenced input file still exists.
A manifest that fails these checks means the run behind it cannot be trusted.
**Read these functions:** `new_manifest()`, `write_manifest()` / `read_manifest()`,
`validate_manifest()`.
**Check yourself:** if you hand-edited a `manifest.json`'s `config` field to make a run look like
it used a different learning rate, which check in `validate_manifest` would catch it, and why?

### `src/shortcut_lens/artifacts.py`
**What it does:** `LocalStore` resolves paths under an artefact root (`artifacts/` by default);
`build_dir()` / `cache_dir()` / `run_dir()` are the fixed layout every stage reads and writes
under (see docs/ARCHITECTURE.md §5).
**Concept:** `ArtifactStore` is declared as a `Protocol`, not a base class, specifically so a
future `HFHubStore` (M3) can satisfy the same interface without inheriting from `LocalStore` --
`cli.py` and every stage only ever depend on the protocol, never on which backend is in use.
**Read these functions:** `LocalStore.path()` / `.ensure_dir()`, `build_dir()`, `run_dir()`.
**Check yourself:** what would break if `run_dir()` used the *config hash* instead of the *run id*
as its directory name? (Hint: what else besides the config determines a run id -- see
docs/ARCHITECTURE.md §5.)

### `src/shortcut_lens/cli.py`
**What it does:** the Typer app `slens`; one subcommand per pipeline stage. `build` and
`data report` are real as of M1 (`_resolve_and_build()` dispatches to the right dataset's builder
by config, is idempotent by build hash, and feeds `data report`'s group-count CSV + sample grid).
Every later-milestone command is still a documented stub that prints which milestone it lands in.
**Concept:** this module is the *composition root* -- the only place allowed to import both the
oracle zone (`build/`) and the label-free zone. Every other module lives entirely on one side of
the firewall; once label-free stages land (M3+), `cli.py` is where a dataset built with group
labels gets narrowed down to `{"image", "y", "example_id"}` before being handed to label-free code
(D-003) -- `build`/`data report` already show the oracle-zone half of that pattern.
**Read these functions:** `_resolve_and_build()`, `build()`, `data_report()`.
**Check yourself:** why must `cli.py`, and no other module, be the one that imports from both
`shortcut_lens.build` and (eventually) `shortcut_lens.discovery`? Why does `_resolve_and_build`
compute the build hash from the config *without* the `dataset` key?

## M1 — Data

### `src/shortcut_lens/build/planting.py`
**What it does:** `PatchSpec`, `add_patch`, `null_patch`, `remove_patch`, `patch_position`.
**Concept:** the position is a pure function of `(build_seed, example_id)` only -- not of colour or
alpha -- which is what lets `null_patch` (the occlusion control, D-008) always land on exactly the
same pixels as the real patch. `remove_patch` doesn't undo pixels; it returns the clean base that
was never patched, because undoing an alpha=1.0 paste is impossible -- see FR-D3.
**Read these functions:** `patch_position()`, `add_patch()`, `remove_patch()`.
**Check yourself:** why does `remove_patch` take only the clean base image as an argument, not the
patched image it's supposedly "removing" the patch from?

### `src/shortcut_lens/build/splits.py`
**What it does:** `stratified_half_split`, `class_balance_by_subsampling`,
`assign_by_target_fraction`, `realistic_subsample`.
**Concept:** two different tools for the same "realistic val mode" idea (D-006), depending on
whether the attribute can be *assigned* (PlantedPets/synthetic_shapes: the image already exists,
only the patch decision changes) or only *observed* (Waterbirds: the background is a fixed fact
about a real photo, so "realistic" means dropping examples, not reassigning them).
**Read these functions:** `assign_by_target_fraction()`, `realistic_subsample()`.
**Check yourself:** for PlantedPets, why is `val_a`/`val_b`'s patch balance allowed to differ
slightly from exactly 50/50 even in `balanced` mode, while the *combined* val split is not?

### `src/shortcut_lens/build/synthetic_shapes.py`, `pets.py`, `waterbirds.py`
**What they do:** one `build_*` function per dataset, each returning `(public_table,
oracle_table)`. `synthetic_shapes` renders images procedurally; `pets` downloads via torchvision
but parses its own raw annotation files rather than relying on the dataset object's internal
attributes; `waterbirds` downloads directly from CodaLab (D-027) and asserts its group counts
against docs/PRD.md §7 exactly, refusing to proceed on any mismatch.
**Concept:** despite three very different data sources, all three produce the same two tables with
the same `group_name` convention (`"<class>|<attribute>"`, matching `vocab/eval_keywords.yaml`'s
frozen keys) and compute `is_minority` from *realised training-set proportions*, never from the
config's `rho` directly (ARCHITECTURE §4) -- so a bug in patch/attribute assignment would show up
as a wrong minority label too, not be silently masked.
**Read these functions:** `build_synthetic_shapes()`, `build_planted_pets()`,
`build_waterbirds()`, and each module's own `_minority_*` / `minority_place_by_class` logic.
**Check yourself:** why is Waterbirds' expected-group-count check something the build *refuses to
proceed past* on a mismatch, rather than a warning?

### `src/shortcut_lens/build/datasets.py`
**What it does:** `RenderedImageDataset`, the only `ImageDataset` implementation.
**Concept:** `patch_flags` is handed in as plain data (already computed from the oracle table
elsewhere in the oracle zone) -- this class itself never reads `oracle.parquet` or imports
`oracle.groups`, which is what a `tests/contracts/` grep test checks independently of
import-linter.
**Read these functions:** `RenderedImageDataset.__getitem__()`.
**Check yourself:** why can the same class serve both "pass-through" datasets (synthetic_shapes,
waterbirds) and "patch overlay" datasets (planted_pets) without a subclass for each?

### `src/shortcut_lens/oracle/groups.py`, `src/shortcut_lens/data/public.py`
**What they do:** the only reader of `oracle.parquet`, and the reader of `public.parquet`,
respectively -- both validate required columns and reject duplicate `example_id`s on load.
**Check yourself:** which one of these two modules would a bug in `discovery/` be *physically
unable* to import, no matter how the code was written?
