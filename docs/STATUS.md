# STATUS — shortcut-lens

Claude Code updates this file at the end of every session and every milestone.
The human reads it first when returning to the project.

## Current state

- **Current milestone:** M0 — scaffold complete, gate G0 run, awaiting human checkpoint
- **Last updated:** 2026-09-11 (M0 implemented and locally verified; dev-machine hardware
  corrected to Apple Silicon/MPS, not CPU-only -- D-025, D-026)
- **Blocked on:** human checkpoint for M0 (see docs/PLAN.md M0); gate G0's "green in CI" clause
  is unverified because the repo has not been pushed to `origin` yet (see gate log below)
- **Dev machine:** MacBook Air M4, 24 GB unified memory, MPS-capable (confirmed:
  `torch.backends.mps.is_available()` → `True` after `uv sync --extra cpu`, D-026). Local
  training/embedding uses MPS by default from M3 onward; cloud GPU handoffs (Kaggle/Colab T4)
  remain the fallback for jobs too large locally (Waterbirds, the E3 sweep) and for non-Mac
  reproducibility.

## Next actions

1. Human: read `config.py`, `manifest.py`, the import-linter contract (M0 human checkpoint,
   docs/PLAN.md). Approve moving on to M1, or push back.
2. Human: `git push` to `origin` (https://github.com/k-28chauhan/shortcut-lens.git) so GitHub
   Actions CI actually runs once, closing out the one unverified part of gate G0.
3. Claude Code: once approved, start a fresh session (`/clear`) and run `/next-milestone` for M1.

## Open questions for the human

- (none yet)

## Gate log

| Gate | Date | Result | Notes / link to evidence |
|---|---|---|---|
| G0 | 2026-09-11 | PASS (local); CI unverified | `make check` green locally (commit `8c8d0d7`): ruff, ruff format, mypy strict (76 files, 0 errors), lint-imports (1 contract kept), pytest (36 passed). `slens --help` lists all 16 ARCHITECTURE §7 commands plus `data report`. `.github/workflows/ci.yml` mirrors `make check` step-for-step and is valid YAML, but has never run (no push to `origin` yet) — CI-green is unverified, not failed. |
| G1 | | | |
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
- **Gate result:** G0 PASS locally; CI step never run (repo not pushed to `origin` yet) — see
  gate log above. Everything CI would check was run identically by hand (`make check`'s exact
  steps), plus the manual import-linter tamper check from docs/PLAN.md's M0 test list
  (a temporary forbidden `discovery -> oracle.groups` import made `lint-imports` fail, then was
  reverted).
- **Decisions made:** D-020 (uv torch cpu/cu126 extras; also pinned `numpy==2.4.6` /
  `scipy==1.17.1`, one minor version behind latest, because newer releases require Python ≥3.12),
  D-021 (Python 3.11.16 pin, CI on ubuntu-24.04 via astral-sh/setup-uv), D-022 (pre-registered
  ResNet-18 sweep fallback above 6 GPU-hours), D-023 (`IMAGENET1K_V1` pretrained weights), D-024 /
  FR-C4 (label-free method/space selection rule for naming/verification/`dfr_discovered`).
- **Open questions:** none blocking M1. The CI-green verification will only close out once the
  human pushes to `origin` — flagged above, not treated as a gate failure since every step CI runs
  was independently verified locally.
