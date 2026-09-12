# STATUS — shortcut-lens

Claude Code updates this file at the end of every session and every milestone.
The human reads it first when returning to the project.

## Current state

- **Current milestone:** M1 — Data layer and firewall (gate G1 passed; awaiting human checkpoint)
- **Last updated:** 2026-09-12 (all three datasets built, tested and verified against real data)
- **Blocked on:** M1's human checkpoint only (see docs/PLAN.md M1: open the sample grids, read
  `planting.py`/`pets.py`/`splits.py`, explain `val_a`/`val_b`/`test` and balanced vs realistic).
- **Dev machine:** MacBook Air M4, 24 GB unified memory, MPS-capable (confirmed:
  `torch.backends.mps.is_available()` → `True` after `uv sync --extra cpu`, D-026). Local
  training/embedding uses MPS by default from M3 onward; cloud GPU handoffs (Kaggle/Colab T4)
  remain the fallback for jobs too large locally (Waterbirds, the E3 sweep) and for non-Mac
  reproducibility.

## Next actions

1. Human: M1 checkpoint -- run `slens data report --config configs/datasets/<name>.yaml` for each
   dataset, open `reports/figures/samples_<dataset>.png`, confirm the patch/attribute and groups
   look right. Read `planting.py`, `pets.py`, `splits.py`. Approve moving on to M2, or push back.
2. Claude Code: once approved, `/next-milestone` for M2 (metrics and statistics core).

## Open questions for the human

- (none yet)

## Gate log

| Gate | Date | Result | Notes / link to evidence |
|---|---|---|---|
| G0 | 2026-09-12 | PASS | `make check` green locally (commit `8c8d0d7`): ruff, ruff format, mypy strict (76 files, 0 errors), lint-imports (1 contract kept), pytest (36 passed). `slens --help` lists all 16 ARCHITECTURE §7 commands plus `data report`. First push (commit `60dea6d`) failed CI at "Set up job": `astral-sh/setup-uv@v10` doesn't resolve (that action publishes no floating major tag, only exact tags). Fixed by pinning `@v10.1.0` (commit `40f88f6`); GitHub Actions run confirmed `success` (human-verified). |
| G1 | 2026-09-12 | PASS | All three dataset counts match expectations: synthetic_shapes and planted_pets by construction (train ρ within 0.5pp of target, test exactly ~50/50, ρ=0.5 control near-zero signal, verified against the real downloaded pet images); waterbirds against real data via a verified Hugging Face parquet mirror (D-029) -- all 12 frozen PRD §7 group-count cells match exactly across train/val/test. Sample grids generated via `slens data report` for all three. `make check` green (`pytest -m "not slow and not gpu and not network"` + the `network`-marked pets/waterbirds suites, both run for real this session). Human checkpoint (reading the sample grids and source files) still pending. |
| G2 | | | |
| G3 | | | |
| G4 | | | |
| G5 | | | |
| G6 | | | |
| G7 | | | |
| G8 | | | |

## GPU handoff log

| Handoff | Date | Commit | Jobs file | Run ids | Measured time | Validated |
|---|---|---|---|---|---|---|
| H1 | | | | | | |
| H2 | | | | | | |
| H3 | | | | | | |

## Milestone summaries

<!-- For each finished milestone, add:
### Mx — <name> (date)
- What was built:
- Files to read, in order:
- How to run:
- Gate result:
- Decisions made (D-ids):
- Open questions:
-->

### M0 — Scaffold and guardrails (2026-09-11)

- **What was built:** the full `src/shortcut_lens` package layout from ARCHITECTURE §2 (every
  module a docstring stub noting its future milestone); real logic for the five foundation
  modules (`config.py`, `seeding.py`, `manifest.py`, `artifacts.py`, `cli.py`); `pyproject.toml`
  with pinned dependencies and the torch cpu/cu126 extras; the import-linter group-label-firewall
  contract; ruff/mypy/pytest config; `.pre-commit-config.yaml`; `.github/workflows/ci.yml`;
  `Makefile` (real `setup`/`check`/`test`, loudly-failing stubs for the rest); unit tests for the
  five foundation modules and two contract tests (vocab freeze hash, grep-based oracle.parquet
  check).
- **Files to read, in order:** `pyproject.toml` (dependencies, tool config, the import-linter
  contract) → `src/shortcut_lens/config.py` → `seeding.py` → `manifest.py` → `artifacts.py` →
  `cli.py`. See docs/CODE_TOUR.md's new "M0 — Foundations" section for what to look for in each.
- **How to run:** `make setup` (once) → `make check`. `uv run slens --help` to see the CLI
  surface; every subcommand currently exits 1 with "not implemented yet -- lands in Mx".
- **Gate result:** G0 PASS, fully closed out (local `make check` + a real green GitHub Actions
  run on commit `40f88f6`, after fixing an `astral-sh/setup-uv@v10` tag-resolution bug found by
  the first CI run). Also verified by hand: a temporary forbidden `discovery -> oracle.groups`
  import made `lint-imports` fail, then was reverted.
- **Decisions made:** D-020 (uv torch cpu/cu126 extras; also pinned `numpy==2.4.6` /
  `scipy==1.17.1`, one minor version behind latest, because newer releases require Python ≥3.12),
  D-021 (Python 3.11.16 pin, CI on ubuntu-24.04 via astral-sh/setup-uv), D-022 (pre-registered
  ResNet-18 sweep fallback above 6 GPU-hours), D-023 (`IMAGENET1K_V1` pretrained weights), D-024 /
  FR-C4 (label-free method/space selection rule for naming/verification/`dfr_discovered`), D-025
  (device policy: mps > cuda > cpu), D-026 (macOS `cpu` extra already resolves an MPS-capable
  wheel).
- **Open questions:** none. M0 fully closed; proceeding to M1.

### M1 — Data layer and firewall (2026-09-12)

- **What was built:** all three datasets, each returning `(public_table, oracle_table)` via a
  `build_*` function: `build/synthetic_shapes.py` (procedural, no download), `build/pets.py`
  (Oxford-IIIT Pet via torchvision, patched shortcut), `build/waterbirds.py` (natural benchmark,
  no shortcut we control). Shared infrastructure: `build/planting.py` (patch mechanism + occlusion
  control), `build/splits.py` (stratified half-split, class balance, two different "realistic
  mode" tools depending on whether an attribute can be assigned or only observed),
  `build/datasets.py` (`RenderedImageDataset`, the one `ImageDataset` implementation),
  `build/tables.py` (writes `public.parquet`/`oracle.parquet`/manifest + the data report CSV and
  sample grid PNG), `data/public.py` + `oracle/groups.py` (schema-validated table readers).
  `slens build` and `slens data report` are wired up for real and idempotent by build hash.
- **Files to read, in order:** `build/planting.py` → `build/splits.py` → `build/synthetic_shapes.py`
  → `build/pets.py` → `build/waterbirds.py` → `build/datasets.py` → `build/tables.py` →
  `oracle/groups.py` / `data/public.py`. See docs/CODE_TOUR.md's "M1 — Data" section.
- **How to run:** `uv run slens build --config configs/datasets/<name>.yaml` then
  `uv run slens data report --config configs/datasets/<name>.yaml` for `synthetic_shapes`,
  `planted_pets` or `waterbirds`. `uv run pytest -m network` runs the real-data tests (downloads
  Oxford-IIIT Pet the first time; Waterbirds' parquet files are already cached locally).
- **Gate result:** G1 PASS -- see gate log above.
- **Decisions made:** D-027 (Waterbirds: rejected the `wilds` PyPI package for an unmaintained
  transitive dependency; first tried the official CodaLab tarball directly), D-028
  (synthetic_shapes' `background` variant intentionally has no naming-eval keywords; its
  discovery-only test is deferred to M4), D-029 (switched Waterbirds to a verified Hugging Face
  parquet mirror after the CodaLab download proved impractically slow -- verified by exact match
  of all 12 group-count cells against the frozen PRD §7 values, not just trusted).
- **Open questions:** none blocking M2. Waiting on the human M1 checkpoint (see "Next actions").
