# STATUS — shortcut-lens

Claude Code updates this file at the end of every session and every milestone.
The human reads it first when returning to the project.

## Current state

- **Current milestone:** M3 — Training, embeddings, reliance and GPU-handoff plumbing. All
  locally-testable code done and green; gate G3 itself is blocked on real E1 results (below).
- **Last updated:** 2026-09-12 (M3 code complete; Handoff H1 being prepared for the cloud per the
  human's explicit choice -- a local E1 run was started, then stopped before real progress per
  that same decision)
- **Blocked on:** finishing H1 prep (commit + push, then `/handoff` gives the human the exact
  cells to run) -- see "Next actions". M1's sample-grid gap remains open too, left as-is per
  explicit human instruction.
- **Dev machine:** MacBook Air M4, 24 GB unified memory, MPS-capable (confirmed:
  `torch.backends.mps.is_available()` → `True` after `uv sync --extra cpu`, D-026). Local
  training/embedding uses MPS by default from M3 onward; cloud GPU handoffs (Kaggle/Colab T4)
  remain the fallback for jobs too large locally (Waterbirds, the E3 sweep) and for non-Mac
  reproducibility.

## Next actions

1. Claude Code: finish preparing Handoff H1 -- push this session's commits, then run `/handoff
   h1_erm` to give the human the commit hash, `jobs/h1_erm.yaml`, expected run ids/outputs, and a
   time estimate (in progress).
2. Human: run H1 on Kaggle/Colab per `docs/RUNBOOK_GPU.md` once `/handoff` hands off the details.
3. Human: M1 checkpoint remains open whenever convenient (see "Open questions" below) -- not
   blocking M3's code, only its human checkpoint.
4. Claude Code: M3 human checkpoint pending (read `erm.py`, explain mixed precision/checkpoint-
   resume/why selection uses average accuracy/what `R_net` measures) -- waiting on the human.

## Open questions for the human

- **M1 sample grids not yet generated/committed, blocking G1's human checkpoint.** `reports/`,
  `results/` and `artifacts/` are all currently empty in this checkout even though the G1 gate log
  below says sample grids were generated in an earlier session -- they were never committed (or
  the working directory changed since). Decide whether to regenerate now (`slens build` +
  `slens data report` for each of the three datasets -- `planted_pets` and `waterbirds` need
  network downloads) or defer. Not acted on yet per the human's instruction this session.

- **E1 is going to the cloud (H1), not run locally.** Measured this session: one real epoch of
  `configs/experiments/e1_pets_rho95.yaml` (ResNet-50, planted_pets ρ=0.95, MPS) took 142s
  wall-clock (~26s one-time weight download, now cached), extrapolating to roughly ~24-30 min/run
  for the real 20-epoch config -- cheap enough that planted_pets could have run locally, and a
  `slens run-jobs` of all 6 planted_pets E1 runs was started locally on that basis. The human then
  asked for cloud instead; the local run was stopped immediately (one partial epoch checkpoint
  discarded, no meaningful compute spent) and H1 now covers all 9 E1 runs (both planted_pets
  configs + waterbirds, 3 seeds each) via `jobs/h1_erm.yaml`. Waterbirds has not been separately
  timed; its per-run cost in `/handoff`'s estimate is extrapolated from dataset size, not measured
  -- flag if that estimate looks off once H1 is actually running.

## Gate log

| Gate | Date | Result | Notes / link to evidence |
|---|---|---|---|
| G0 | 2026-09-12 | PASS | `make check` green locally (commit `8c8d0d7`): ruff, ruff format, mypy strict (76 files, 0 errors), lint-imports (1 contract kept), pytest (36 passed). `slens --help` lists all 16 ARCHITECTURE §7 commands plus `data report`. First push (commit `60dea6d`) failed CI at "Set up job": `astral-sh/setup-uv@v10` doesn't resolve (that action publishes no floating major tag, only exact tags). Fixed by pinning `@v10.1.0` (commit `40f88f6`); GitHub Actions run confirmed `success` (human-verified). |
| G1 | 2026-09-12 | PASS | All three dataset counts match expectations: synthetic_shapes and planted_pets by construction (train ρ within 0.5pp of target, test exactly ~50/50, ρ=0.5 control near-zero signal, verified against the real downloaded pet images); waterbirds against real data via a verified Hugging Face parquet mirror (D-029) -- all 12 frozen PRD §7 group-count cells match exactly across train/val/test. Sample grids generated via `slens data report` for all three. `make check` green (`pytest -m "not slow and not gpu and not network"` + the `network`-marked pets/waterbirds suites, both run for real this session). Human checkpoint (reading the sample grids and source files) still pending. |
| G2 | 2026-09-12 | PASS* | `make check` green (ruff, ruff format, mypy strict 76 files, lint-imports 1 contract kept, pytest 114 passed / 10 deselected). Full suite incl. slow + network: 124 passed, coverage 92% (≥85% target). `metrics.py`'s 9 functions (`accuracy`, `group_accuracy`, `worst_group_accuracy`, `mean_group_accuracy`, `weighted_average_accuracy`, `wga_gap`, `precision_at_k`, `slice_auroc`, `jaccard`, `recovery`) each carry a docstring citing PRD §12. *PASS is scoped to what docs/PLAN.md's own M2 task list assigns to M2 (`metrics.py`/`stats.py`/`evaluation/core.py`); 5 of PRD §12's 13 rows (Top-1 hit, Naming hit@3, `R_net`, Fix/break rate, Cost of labels) are derived quantities the M2 task list itself defers to `discovery_eval.py` (M4), `naming_eval.py` (M5), `verification/reliance.py` (M3) and `verification/verify.py` / `mitigation_eval.py` (M6/M7) -- flagging the literal "every metric in PRD §12" gate wording against that scoping rather than silently treating it as fully satisfied. Human checkpoint (read `metrics.py`/`stats.py`, re-derive BH by hand) still pending. |
| G3 | 2026-09-12 | PENDING | All M3 code green: `make check` (ruff, ruff format, mypy strict 76 files, lint-imports 1 contract kept, pytest 156 passed/12 deselected); full suite incl. slow+network: 167+ passed, 90% coverage; `make smoke` (build→train→embed model+fake-CLIP) in ~2.3s. `slens train`/`embed`/`reliance`/`run-jobs`/`pull-artifacts`/`validate-run`/draft `evaluate` manually verified end-to-end on a real config. **Gate's own frozen thresholds (WGA gap, `R_net` on real E1 runs) not yet evaluable** -- no full E1 training run exists yet; E1 is going to Handoff H1 (cloud) per the human's choice, see the GPU handoff log and "Open questions" above. |
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

### M2 — Metrics and statistics core (2026-09-12)

- **What was built:** `metrics.py` (pure numpy functions: `accuracy`, `group_accuracy`,
  `worst_group_accuracy`, `mean_group_accuracy`, `weighted_average_accuracy`, `wga_gap`,
  `precision_at_k`, `slice_auroc`, `jaccard`, `recovery` -- all fractions in [0,1], D-030);
  `stats.py` (`percentile_bootstrap` seeded via an explicit `numpy.random.Generator` with optional
  stratification, `fisher_exact_one_sided`, `benjamini_hochberg` from scratch, `aggregate_over_seeds`);
  `evaluation/core.py` (`join_predictions_with_groups` -- strict id-set join, raises on any
  mismatch -- and `group_metrics_table`, a tidy per-group accuracy table with bootstrap CIs, one
  independent RNG stream per group). `evaluation/core.py` takes plain DataFrames rather than the
  `Predictions` dataclass ARCHITECTURE §3 sketches, since nothing produces a real one until M3
  (D-031). `make test-slow` now runs `pytest -m "slow or network"` (was a stub).
- **Files to read, in order:** `metrics.py` → `stats.py` → `evaluation/core.py`. See
  docs/CODE_TOUR.md's "M2 — Measurement" section.
- **How to run:** `make check` (fast tests); `make test-slow` (adds the bootstrap-coverage
  simulation and M1's network-marked dataset tests).
- **Gate result:** G2 PASS* -- see gate log above (starred: literal PRD §12 coverage is scoped to
  what M2's own task list assigns to this milestone; the other 5 rows land in M3/M4/M5/M6/M7).
- **Decisions made:** D-030 (metrics are fractions in [0,1]; "points" = fraction × 100), D-031
  (`evaluation/core.py` takes plain DataFrames; `Predictions`/`EmbeddingTable` deferred to
  `types.py` in M3).
- **Verified by running code (CLAUDE.md §5):** `sklearn.metrics.roc_auc_score` returns `NaN` with
  an `UndefinedMetricWarning` (not an exception) for a single-class `y_true`; `scipy.stats.
  fisher_exact(alternative="greater")`'s direction convention on a hand-built 2×2 table;
  `statsmodels.stats.multitest.multipletests(method="fdr_bh")`'s q-values match the from-scratch
  `benjamini_hochberg` reverse-running-minimum formula exactly.
- **Open questions:** none new. Human checkpoint approved 2026-09-12. M1's sample-grid gap (see
  "Open questions for the human" above) left open, explicitly not blocking, per the human.

### M3 — Training, embeddings, reliance and GPU-handoff plumbing (2026-09-12)

- **What was built:** `models/backbones.py` (resnet18/50 + tiny_cnn, `Backbone.features()`/
  `.forward()`); `training/erm.py` (`train_erm`: device/precision policy, per-epoch checkpointing,
  exact resume via a fresh per-epoch `DataLoader`+`Generator` seeded by `(seed, epoch)` -- D-035 --
  `history.csv` with per-epoch `duration_s`, predictions for every split); `embeddings/{model_space,
  clip_space,store,cache}.py` (both spaces, float16 storage with id-alignment checks, a
  `RenderKey`-keyed CLIP cache -- implemented and unit-tested, not yet wired into `slens embed`,
  see below); `verification/reliance.py` (`compute_reliance`: FR-R1's `R_net`, D-036's row schema);
  `artifacts.py`'s `HFHubStore` (`push_run`/`pull_run` via `huggingface_hub`); `jobs.py`
  (`run_jobs`: sequencing + stop-on-first-failure + `--dry-run`, idempotency left to each stage);
  `cli.py` wiring for `train`/`embed`/`reliance`/`run-jobs`/`pull-artifacts`/`validate-run`/draft
  `evaluate`; `make smoke`; config files (`configs/train/`, `configs/embed/`,
  `configs/experiments/e1_*.yaml`, `configs/experiments/smoke.yaml`); `notebooks/gpu_runner.ipynb`
  (real cells) and `docs/RUNBOOK_GPU.md` updated to match; ARCHITECTURE §6 documents the jobs-file
  schema; `jobs/h1_erm.yaml` (all 9 E1 runs per PLAN.md's frozen H1 spec).
- **Files to read, in order:** `models/backbones.py` → `training/erm.py` → `embeddings/model_space.py`
  / `clip_space.py` / `store.py` / `cache.py` → `verification/reliance.py` → `artifacts.py`
  (`HFHubStore`) → `jobs.py`. See docs/CODE_TOUR.md's "M3" section.
- **How to run:** `make smoke` (fast, offline pipeline proof); `uv run slens train --config
  configs/experiments/e1_pets_rho95.yaml` then `embed --space model`/`--space clip` then
  `reliance`/`evaluate`; `uv run slens run-jobs jobs/h1_erm.yaml --dry-run` to preview H1.
- **Gate result:** G3 PENDING -- see gate log above. Code and tests are complete and green; the
  gate's own frozen thresholds need a real E1 training run, going to Handoff H1 (cloud).
- **Decisions made:** D-032 (`val_avg_acc` selection metric = `val_a`+`val_b` combined), D-033
  (`amp=true` on non-CUDA: warn, don't silently no-op or raise; `precision_mode` recorded in the
  manifest -- human instruction), D-034 (`Predictions`/`EmbeddingTable` dataclasses turned out
  unnecessary in M3 after all, amending D-031's plan -- reported as a finding, not silently
  dropped), D-035 (exact resume via a fresh per-epoch `DataLoader`, not RNG-state snapshotting),
  D-036 (`reliance.parquet`'s row schema: 4 per-condition rows + 1 `R_net` summary row).
- **Verified by running code (CLAUDE.md §5):** `torchvision.models.ResNet18_Weights`/
  `ResNet50_Weights` enum members and `IMAGENET1K_V1.transforms()`; `open_clip.list_pretrained()`
  confirms `("ViT-B-32", "laion2b_s34b_b79k")`; `open_clip`'s own preprocessing is a geometric
  no-op on already-224x224 images; `torch.amp.GradScaler`/`torch.autocast` API for torch 2.14;
  `huggingface_hub.HfApi.upload_folder`/`snapshot_download` signatures; a real 1-epoch training run
  (planted_pets ρ=0.95, ResNet-50, MPS) to measure timing for the H1 estimate.
- **Scope trims (disclosed, not silent):** `embeddings/cache.py` is implemented and tested but not
  wired into `slens embed --space clip`'s CLI path -- no M3 config varies rho for a fixed render
  set within one session, so the cache's reuse benefit (FR-E4) isn't exercised until M6's sweep;
  wiring deferred to then. `slens evaluate`'s non-final path never touches the `test` split
  (CLAUDE.md §3 rule 3), only `val_a`/`val_b`. Waterbirds' per-run cost in H1's estimate is
  extrapolated from dataset size relative to the measured pets run, not separately measured.
- **Open questions:** the M3 human checkpoint (read `erm.py`, explain mixed precision/checkpoint-
  resume/why selection uses average accuracy/what `R_net` measures) is still pending; M1's
  sample-grid gap remains open too; Handoff H1 is being prepared (see GPU handoff log).
