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

## M2 — Measurement

### `src/shortcut_lens/metrics.py`
**What it does:** pure functions over plain numpy arrays -- `accuracy`, `group_accuracy`,
`worst_group_accuracy`, `mean_group_accuracy`, `weighted_average_accuracy`, `wga_gap`,
`precision_at_k`, `slice_auroc`, `jaccard`, `recovery`. Every metric returns a fraction in [0, 1]
(D-030); "N points" elsewhere in the docs means a fraction difference of N/100 on this scale.
**Concept:** this module never loads a group label itself -- every function takes `group` (or
`target_membership`) as a plain array the caller already has, which is what lets the exact same
functions serve both zones (label-free code computing plain `accuracy` for checkpoint selection,
and `evaluation/core.py` computing `group_accuracy` with true groups). `precision_at_k` raises
rather than silently shrinking `k` when too few examples are available, and `slice_auroc` returns
`NaN` (not an exception) for a degenerate all-one-class slice -- both verified against
`sklearn.metrics.roc_auc_score`'s own behaviour first (CLAUDE.md §5).
**Read these functions:** `group_accuracy()`, `weighted_average_accuracy()`, `recovery()`.
**Check yourself:** why does `weighted_average_accuracy` take the weights as a plain argument
instead of computing them itself from group sizes?

### `src/shortcut_lens/stats.py`
**What it does:** `percentile_bootstrap` (seeded, optionally stratified), `fisher_exact_one_sided`,
`benjamini_hochberg` (from scratch), `aggregate_over_seeds`.
**Concept:** `percentile_bootstrap` takes an explicit `numpy.random.Generator`, not a raw seed --
callers derive one with `seeding.make_rng(seed, *context_keys)` so e.g. `evaluation/core.py` can
give every group its own independent resampling stream from one run seed. `benjamini_hochberg`'s
q-value is a *reverse running minimum* of `p_(i) * m / i` over the sorted p-values -- that
monotonicity is what makes it match `statsmodels.stats.multitest.multipletests(method="fdr_bh")`
exactly (verified in `tests/unit/test_stats.py`), not just the reject/accept decision.
**Read these functions:** `percentile_bootstrap()`, `benjamini_hochberg()`.
**Check yourself:** re-derive the BH procedure on paper for `p = [0.01, 0.02, 0.03, 0.04, 0.20]`,
`q = 0.05` -- which are rejected, and what are their q-values? (Compare against
`test_benjamini_hochberg_matches_hand_worked_example`.)

### `src/shortcut_lens/evaluation/core.py`
**What it does:** `join_predictions_with_groups` (strict id-set join) and `group_metrics_table`
(tidy per-group accuracy with bootstrap CIs).
**Concept:** the join deliberately raises if `predictions` and `oracle_groups` don't cover exactly
the same `example_id` set, rather than keeping the intersection -- a silent partial join is exactly
the kind of bug that produces a plausible-looking wrong number (docs/TESTING.md). Works over plain
DataFrames rather than the `Predictions` dataclass ARCHITECTURE §3 sketches, since nothing produces
a real one until M3 (D-031).
**Read these functions:** `join_predictions_with_groups()`, `group_metrics_table()`.
**Check yourself:** why does each group in `group_metrics_table` get its own `make_rng` call
(`make_rng(seed, "group_metrics_table", group_id)`) instead of one shared `Generator` for the whole
table?

## M3 — Training, embeddings, reliance and GPU-handoff plumbing

### `src/shortcut_lens/models/backbones.py`
**What it does:** `build_backbone()` wraps torchvision resnet18/50 (dropping the original `fc`) or
a small conv net (`tiny_cnn`) in a `Backbone` with `.features()` (penultimate activations) and
`.forward()` (logits).
**Concept:** `.features()` always returns the same fixed-shape flattened trunk output regardless
of `arch` -- swapping the `.head` (as last-layer retraining does in M7) never changes what
`.features()` returns for the same input, which is what M7's frozen-backbone assumption depends on.
**Read these functions:** `Backbone.features()`, `build_backbone()`.
**Check yourself:** why does `tiny_cnn` ignore the `pretrained` argument entirely?

### `src/shortcut_lens/training/erm.py`
**What it does:** `train_erm()` -- the whole training loop, checkpointing, resume, and final
per-split predictions in one function; `resolve_device()` (D-025's mps > cuda > cpu); `resolve_precision_mode()`.
**Concept:** exact resume works without ever snapshotting RNG state: each epoch gets a brand new
`DataLoader` built with a fresh `torch.Generator` seeded only by `(seed, epoch)`
(`seeding.make_rng`), so epoch *k*'s shuffle order is a pure function of `(seed, epoch)` alone --
resuming from any earlier checkpoint reproduces every later epoch identically, with nothing to
restore. `resolve_precision_mode` never silently no-ops `amp=true` on non-CUDA devices: it logs a
warning *and* the manifest records `precision_mode` (D-033), so a report is never wrong about what
actually ran, even if nobody read the log.
**Read these functions:** `train_erm()`, `_epoch_train_loader()`, `resolve_precision_mode()`.
**Check yourself:** why does `train_erm` call `torch.manual_seed(config.seed)` unconditionally,
even on a resumed run where the loaded checkpoint immediately overwrites the freshly-initialised
weights anyway?

### `src/shortcut_lens/embeddings/model_space.py`, `clip_space.py`, `store.py`, `cache.py`
**What they do:** `embed_model_space()`/`embed_clip_space()` run a (already eval-transformed)
dataset through a trunk and return `(vectors, example_ids)` in iteration order; `store.py`
writes/reads them as float16 `.npy` + an aligned ids parquet; `cache.py` keys a CLIP embedding
cache by `RenderKey` (plain data -- size/colour/alpha/base image/build seed), not by rho.
**Concept:** CLIP needs its *own* preprocessing (different normalisation from the model-space
ImageNet one), verified by running code to be a geometric no-op on this project's already-224x224
images -- so the composition root attaches CLIP's own `preprocess` as the dataset's `transform`
rather than reusing `data.transforms`. `RenderKey` duplicates a few fields from
`build.planting.PatchSpec` as plain data instead of importing it, because this module is
label-free and `build` is oracle (D-003) -- only `cli.py` ever turns a real `PatchSpec` into a
`RenderKey`.
**Read these functions:** `embed_clip_space()`, `cache.render_hash()`.
**Check yourself:** why does the CLIP cache key on the *set* of render keys for a whole split,
not per image?

### `src/shortcut_lens/verification/reliance.py`
**What it does:** `compute_reliance()` -- FR-R1's `R_net`, from a trained model and a joined
predictions+oracle table.
**Concept:** the add/remove directions' null-patch controls isolate reliance on the patch's
*colour* from mere reliance on *something being painted there* (D-008's occlusion confound) --
`add_null`/`remove_null` both use `build.planting.null_patch`, just starting from different base
images (a never-patched dog vs. a cat's clean base after its real patch is stripped).
**Read these functions:** `compute_reliance()`, `_render()`.
**Check yourself:** why does `remove_null` call `null_patch()` on the *clean* base rather than on
the cat's actually-patched image?

### `src/shortcut_lens/artifacts.py` (`HFHubStore`), `src/shortcut_lens/jobs.py`
**What they do:** `HFHubStore` resolves paths exactly like `LocalStore` and adds `push_run()`/
`pull_run()` (one run's folder at a time, via `huggingface_hub`); `jobs.run_jobs()` runs an
ordered list of `slens <stage> ...` commands via an injected `runner` callable, stopping at the
first non-zero exit.
**Concept:** `run_jobs` owns *sequencing and failure-stopping* only -- idempotency (skip a
finished step) is deliberately left to each stage's own command, not tracked here, so there is
exactly one place (`cli.py`'s per-stage config-hash/manifest check, or `train_erm`'s checkpoint)
that can be wrong about whether a step is done.
**Read these functions:** `HFHubStore.push_run()`, `jobs.run_jobs()`.
**Check yourself:** what does `slens run-jobs jobs/h1_erm.yaml --dry-run` let a human check before
any GPU time is spent?
