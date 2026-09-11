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
**What it does:** the Typer app `slens`; one subcommand per pipeline stage. At M0 every command
is a documented stub that prints which milestone it lands in and exits non-zero.
**Concept:** this module is the *composition root* -- the only place allowed to import both the
oracle zone (`build/`) and the label-free zone. Every other module lives entirely on one side of
the firewall; `cli.py` is where a dataset built with group labels gets narrowed down to
`{"image", "y", "example_id"}` before being handed to label-free code (D-003). None of that
wiring exists yet at M0 -- what's here is only the interface (`--help` output) it will fill in.
**Read these functions:** `_not_implemented()`, and any one real command, e.g. `train()` or
`mitigate()`, to see the `--option` patterns that later milestones will keep.
**Check yourself:** why must `cli.py`, and no other module, be the one that imports from both
`shortcut_lens.build` and (eventually) `shortcut_lens.discovery`?
