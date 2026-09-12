# STATUS — shortcut-lens

Claude Code updates this file at the end of every session and every milestone.
The human reads it first when returning to the project.

## Current state

- **Current milestone:** M1 — Data layer and firewall (in progress; waterbirds pending real data)
- **Last updated:** 2026-09-12 (planting/synthetic_shapes/pets/splits/datasets/tables done and
  tested; waterbirds.py code-complete but unverified against real data)
- **Blocked on:** an unusually slow download from CodaLab for the Waterbirds tarball (~30-40KB/s;
  ~470MB total). Everything else in M1 is done, tested and green. Once the download finishes:
  fill in `tarball_sha256` in `configs/datasets/waterbirds.yaml` (currently `[TBD]`), run
  `build_waterbirds()` against the real `metadata.csv`, confirm the frozen PRD §7 group counts
  actually match, and add the `network`-marked test module for it (mirroring `tests/unit/test_pets.py`).
- **Dev machine:** MacBook Air M4, 24 GB unified memory, MPS-capable (confirmed:
  `torch.backends.mps.is_available()` → `True` after `uv sync --extra cpu`, D-026). Local
  training/embedding uses MPS by default from M3 onward; cloud GPU handoffs (Kaggle/Colab T4)
  remain the fallback for jobs too large locally (Waterbirds, the E3 sweep) and for non-Mac
  reproducibility.

## Next actions

1. Claude Code: once the Waterbirds download finishes, pin its checksum, verify the build for
   real, add its test module, tick the remaining M1 checkbox, and run gate G1.
2. Human: nothing blocking yet -- M1's human checkpoint (sample grids, `planting.py`/`pets.py`/
   `splits.py`) is best done once G1 passes in full, including Waterbirds.

## Open questions for the human

- (none yet)

## Gate log

| Gate | Date | Result | Notes / link to evidence |
|---|---|---|---|
| G0 | 2026-09-12 | PASS | `make check` green locally (commit `8c8d0d7`): ruff, ruff format, mypy strict (76 files, 0 errors), lint-imports (1 contract kept), pytest (36 passed). `slens --help` lists all 16 ARCHITECTURE §7 commands plus `data report`. First push (commit `60dea6d`) failed CI at "Set up job": `astral-sh/setup-uv@v10` doesn't resolve (that action publishes no floating major tag, only exact tags). Fixed by pinning `@v10.1.0` (commit `40f88f6`); GitHub Actions run confirmed `success` (human-verified). |
| G1 | pending | IN PROGRESS | synthetic_shapes and planted_pets fully built, tested (incl. against real downloaded pet images) and wired through `slens build`/`slens data report`; counts match expectations by construction (verified: train ρ within 0.5pp of target, test exactly 50/50, control ρ=0.5 near-zero signal). Waterbirds pending real-data verification (see "Blocked on" above). |
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
