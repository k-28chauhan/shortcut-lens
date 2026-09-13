# STATUS — shortcut-lens

Claude Code updates this file at the end of every session and every milestone.
The human reads it first when returning to the project.

## Current state

- **Current milestone:** M3 — Training, embeddings, reliance and GPU-handoff plumbing. Code
  complete and green; gate G3 **failed on planted_pets** (D-037, kept as documented evidence);
  D-002 step 3 (`planted_cifar_pets`) implemented and locally sanity-checked (D-038), real run
  going to Handoff H1c.
- **Last updated:** 2026-09-13 (`build/cifar_pets.py` implemented, tested, 1-epoch local sanity
  check passed; Handoff H1c prepared for the real 20-epoch/3-seed run)
- **Blocked on:** the human running Handoff H1c (`jobs/h1c_cifar_pets.yaml`, ~1.5-2h estimate,
  needs a push first) -- see "Next actions". M1's sample-grid gap remains open too.
- **Dev machine:** MacBook Air M4, 24 GB unified memory, MPS-capable (confirmed:
  `torch.backends.mps.is_available()` → `True` after `uv sync --extra cpu`, D-026). Local
  training/embedding uses MPS by default from M3 onward; cloud GPU handoffs (Kaggle/Colab T4)
  remain the fallback for jobs too large locally (Waterbirds, the E3 sweep) and for non-Mac
  reproducibility.

## Next actions

1. Human: run Handoff H1c (`jobs/h1c_cifar_pets.yaml`) on Kaggle/Colab once pushed; report back.
2. Claude Code: pull + validate H1c, evaluate gate G3 on `planted_cifar_pets`. If it also fails,
   the next unconfounded step is `pretrained=None`/`tiny_cnn` on the same CIFAR base (D-038).
3. Human: once a planted dataset passes G3 (or a final decision is made), decide when to run the
   deferred waterbirds half (`jobs/h1_waterbirds.yaml`).
4. Human: M1 checkpoint remains open whenever convenient (see "Open questions" below).
5. Claude Code: M3 human checkpoint pending (read `erm.py`, explain mixed precision/checkpoint-
   resume/why selection uses average accuracy/what `R_net` measures) -- waiting on the human.

## Open questions for the human

- **M1 sample grids not yet generated/committed, blocking G1's human checkpoint.** `reports/`,
  `results/` and `artifacts/` are all currently empty in this checkout even though the G1 gate log
  below says sample grids were generated in an earlier session -- they were never committed (or
  the working directory changed since). Decide whether to regenerate now (`slens build` +
  `slens data report` for each of the three datasets -- `planted_pets` and `waterbirds` need
  network downloads) or defer. Not acted on yet per the human's instruction this session.

- **A real Hugging Face token was pasted in plaintext into this chat session on 2026-09-13.** The
  human was told to revoke and rotate it immediately on huggingface.co/settings/tokens. Confirm
  this was done -- flagging here so it isn't forgotten by the next session.

## Gate log

| Gate | Date | Result | Notes / link to evidence |
|---|---|---|---|
| G0 | 2026-09-12 | PASS | `make check` green locally (commit `8c8d0d7`): ruff, ruff format, mypy strict (76 files, 0 errors), lint-imports (1 contract kept), pytest (36 passed). `slens --help` lists all 16 ARCHITECTURE §7 commands plus `data report`. First push (commit `60dea6d`) failed CI at "Set up job": `astral-sh/setup-uv@v10` doesn't resolve (that action publishes no floating major tag, only exact tags). Fixed by pinning `@v10.1.0` (commit `40f88f6`); GitHub Actions run confirmed `success` (human-verified). |
| G1 | 2026-09-12 | PASS | All three dataset counts match expectations: synthetic_shapes and planted_pets by construction (train ρ within 0.5pp of target, test exactly ~50/50, ρ=0.5 control near-zero signal, verified against the real downloaded pet images); waterbirds against real data via a verified Hugging Face parquet mirror (D-029) -- all 12 frozen PRD §7 group-count cells match exactly across train/val/test. Sample grids generated via `slens data report` for all three. `make check` green (`pytest -m "not slow and not gpu and not network"` + the `network`-marked pets/waterbirds suites, both run for real this session). Human checkpoint (reading the sample grids and source files) still pending. |
| G2 | 2026-09-12 | PASS* | `make check` green (ruff, ruff format, mypy strict 76 files, lint-imports 1 contract kept, pytest 114 passed / 10 deselected). Full suite incl. slow + network: 124 passed, coverage 92% (≥85% target). `metrics.py`'s 9 functions (`accuracy`, `group_accuracy`, `worst_group_accuracy`, `mean_group_accuracy`, `weighted_average_accuracy`, `wga_gap`, `precision_at_k`, `slice_auroc`, `jaccard`, `recovery`) each carry a docstring citing PRD §12. *PASS is scoped to what docs/PLAN.md's own M2 task list assigns to M2 (`metrics.py`/`stats.py`/`evaluation/core.py`); 5 of PRD §12's 13 rows (Top-1 hit, Naming hit@3, `R_net`, Fix/break rate, Cost of labels) are derived quantities the M2 task list itself defers to `discovery_eval.py` (M4), `naming_eval.py` (M5), `verification/reliance.py` (M3) and `verification/verify.py` / `mitigation_eval.py` (M6/M7) -- flagging the literal "every metric in PRD §12" gate wording against that scoping rather than silently treating it as fully satisfied. Human checkpoint (read `metrics.py`/`stats.py`, re-derive BH by hand) still pending. |
| G3 | 2026-09-13 | PENDING on planted_cifar_pets; FAIL on planted_pets (documented) | `planted_pets`: FAIL at both ρ=0.95 (mean WGA gap 1.7±0.9pts, `R_net` 0.011±0.003) and ρ=0.99 (4.3±2.6pts, 0.024±0.018) -- need ≥10pts / ≥0.20. Control ρ=0.5 correctly passes (`R_net` 0.001±0.001, need ≤0.05). Root cause confirmed genuine (not a bug, checked both times): an ImageNet-pretrained ResNet-50 solves cat-vs-dog almost immediately regardless of correlation strength (PRD §16 risk 1). D-002's ρ fallback exhausted; kept as documented evidence. D-038: implemented `planted_cifar_pets` (D-002 step 3) -- new build module, tests (7 network-marked, all pass), 1-epoch local sanity check passed (train_acc 0.975, val_avg_acc 0.531 after 1 epoch -- a much bigger train/val gap than pets showed, consistent with the harder-task hypothesis). Handoff H1c prepared for the real 20-epoch/3-seed run. Gate thresholds unchanged throughout. |
| G4 | | | |
| G5 | | | |
| G6 | | | |
| G7 | | | |
| G8 | | | |

## GPU handoff log

| Handoff | Date | Commit | Jobs file | Run ids | Measured time | Validated |
|---|---|---|---|---|---|---|
| H1 | 2026-09-12 | `d589f5d` | `jobs/h1_erm.yaml` (planted_pets ρ=0.95+control, 6 runs); `jobs/h1_waterbirds.yaml` (waterbirds, 3 runs, deferred) | `E1-planted_pets-e7a2c0c8-s{0,1,2}` (ρ=0.95), `E1-planted_pets-a08bcaa2-s{0,1,2}` (control); `E1-waterbirds-a9772376-s{0,1,2}` (deferred half) | **measured**: ~406s/run (~6.8 min), Tesla T4, `fp16_amp` -- much faster than the pre-run MPS-based estimate | yes, all 6 (`slens validate-run`) -- see gate G3 (ρ=0.95 failed; control passed as expected) |
| H1b | 2026-09-13 | `f520176` | `jobs/h1_pets_rho99.yaml` (planted_pets ρ=0.99, 3 runs) | `E1-planted_pets-50f71fb8-s{0,1,2}` | done | yes, all 3 (`slens validate-run`) -- see gate G3 (also failed) |
| H1c | 2026-09-13 | `9defc5f` | `jobs/h1c_cifar_pets.yaml` (planted_cifar_pets ρ=0.99, 3 runs) | `E1-planted_cifar_pets-ef0cc3ab-s{0,1,2}` | prepared, estimate ~1.5-2h total (scaled from H1's measured per-image cost x ~4.2 for CIFAR's larger 10k-image train split -- not directly measured on cloud hardware) | not yet |
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
- **Gate result:** G3 -- code side fully green throughout. Threshold side: **FAIL on
  `planted_pets`** at both ρ=0.95 and ρ=0.99 (D-037, root cause confirmed genuine, kept as
  documented evidence); D-002's fallback (`planted_cifar_pets`, D-038) implemented, tested, and
  locally sanity-checked, real result pending Handoff H1c.
- **Decisions made:** D-032 (`val_avg_acc` selection metric = `val_a`+`val_b` combined), D-033
  (`amp=true` on non-CUDA: warn, don't silently no-op or raise; `precision_mode` recorded in the
  manifest -- human instruction), D-034 (`Predictions`/`EmbeddingTable` dataclasses turned out
  unnecessary in M3 after all, amending D-031's plan -- reported as a finding, not silently
  dropped), D-035 (exact resume via a fresh per-epoch `DataLoader`, not RNG-state snapshotting),
  D-036 (`reliance.parquet`'s row schema: 4 per-condition rows + 1 `R_net` summary row), D-037
  (G3 fails on `planted_pets` at ρ=0.95 and ρ=0.99, genuine finding), D-038 (`planted_cifar_pets`,
  D-002's fallback: same construction, patch painted after upscaling 32px→224px).
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
  sample-grid gap remains open too; Handoff H1c (`planted_cifar_pets`) is being prepared next.
