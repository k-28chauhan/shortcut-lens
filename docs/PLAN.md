# PLAN — shortcut-lens

Nine milestones (M0–M8), each ending in an exit gate (G0–G8) and a human checkpoint.
Claude Code ticks boxes as tasks complete. **Gate thresholds are frozen** (see CLAUDE.md §3, rule 4).

## Principles

1. Build the whole pipeline on `synthetic_shapes` on CPU first; real data only once the plumbing works.
2. Pure logic before I/O. Tests before or with every module.
3. GPU work is batched into four handoffs (H1–H4) so the human runs few, well-defined sessions.
4. At every gate, stop. The human reads the listed files and approves before the next milestone.

## Timeline (4 weeks + buffer)

| Week | Milestones | GPU handoffs |
|---|---|---|
| 1 | M0 scaffold · M1 data · M2 metrics | — |
| 2 | M3 training/embeddings · M4 discovery | H1 (ERM + embeddings), H2 (captions, early) |
| 3 | M5 naming · M6 verification + sweep · M7 mitigation | H3 (sweep) |
| 4 | M8 report, app, release, write-up | H4 (only if re-runs needed) |
| +1 | Buffer / stretch goals | — |

---

## M0 — Scaffold and guardrails

**Goal:** a repo where the rules are enforced by tooling before any ML exists.

- [x] Create the layout from ARCHITECTURE §2 with empty packages and module docstrings.
- [x] `pyproject.toml` with uv, Python 3.11, pinned runtime and dev dependencies; `uv.lock`.
- [x] ruff (lint + format), mypy strict on `src/shortcut_lens`, pytest with markers
      `slow`, `gpu`, `network`; pre-commit hooks.
- [x] import-linter contract from ARCHITECTURE §1 (works on stub modules).
- [x] GitHub Actions CI: `uv sync` → ruff → mypy → lint-imports → `pytest -m "not slow and not gpu and not network"`.
- [x] Makefile targets from CLAUDE.md §7 (stubs allowed for later ones).
- [x] `config.py`: Pydantic models, YAML loader with composition, canonical config hash.
- [x] `seeding.py`: `set_all_seeds(seed)`, `make_rng(seed, *keys)`.
- [x] `manifest.py`: write, read, validate.
- [x] `artifacts.py`: path helpers, `LocalStore` (HF Hub backend comes in M3).
- [x] `cli.py`: Typer app `slens` with every subcommand stubbed and documented in `--help`.
- [x] `docs/STATUS.md` and `docs/CODE_TOUR.md` initialised.
- [x] A test that recomputes the sha256 of `vocab/*.yaml` and asserts it matches `vocab/FROZEN.sha256`.

**Tests:** config hash independent of key order and stable across runs; manifest round-trip;
`set_all_seeds` makes torch/numpy draws repeatable; contracts pass; a deliberately forbidden import
in a temporary test module makes lint-imports fail (then removed).

**Gate G0:** `make check` green locally and in CI; `slens --help` lists all commands.

**Human checkpoint:** read `config.py`, `manifest.py`, the import-linter contract. Be able to explain
why config hashes and manifests exist and what the firewall protects.

---

## M1 — Data layer and firewall

**Goal:** all three datasets built reproducibly, with labels split into public and oracle tables.

- [ ] `build/planting.py`: `PatchSpec`, `add_patch`, `null_patch`, deterministic positions; remove = re-render.
- [ ] `build/synthetic_shapes.py`: circle vs square, attribute variants `dot` and `background`, params ρ, n, seed.
- [ ] `build/pets.py`: torchvision Oxford-IIIT Pet download; derive cat/dog (verify the `binary-category`
      target type; fallback: species from the dataset annotation list, or capitalised filenames = cats);
      preprocess to 224 JPEG cache; class balance; splits per PRD §7; patch assignment per split.
- [ ] `build/waterbirds.py`: download (WILDS or direct tarball — decide, log), parse `metadata.csv`
      (verify columns), map official splits, `val_a`/`val_b` by class, realistic mode.
- [ ] `build/splits.py`: stratified 50/50 splitting by class; balanced/realistic assignment.
- [ ] `build/tables.py` + `slens build`: write `public.parquet`, `oracle.parquet`, build manifest.
- [ ] `build/datasets.py`: `RenderedImageDataset` (images rendered on the fly, returns only the three keys).
- [ ] `data/public.py`, `data/transforms.py`; `oracle/groups.py`.
- [ ] `slens data report`: `results/data_counts_<dataset>.csv` and `reports/figures/samples_<dataset>.png`
      (a grid with one row per group).

**Tests:** `remove(add(x)) == x` exactly; positions in bounds and deterministic; null patch differs
from the real patch only in colour; splits disjoint by `example_id` and base image; class balance;
planted proportions within ±1 image of target per split; Waterbirds counts equal PRD §7 values
(network-marked); every dataset type returns exactly `{"image", "y", "example_id"}`; oracle table
never read outside the oracle zone (contract).

**Gate G1:** all counts tables match expectations; sample grids generated; tests green.

**Human checkpoint:** open the sample grids and confirm the patch and groups look right. Read
`planting.py`, `pets.py`, `splits.py`. Explain the role of `val_a`, `val_b` and `test`, and why
val has balanced and realistic modes.

---

## M2 — Metrics and statistics core

**Goal:** trustworthy measurement code, fully tested on CPU, before any model exists.

- [ ] `metrics.py`: accuracy, group accuracy, WGA, mean-group accuracy, weighted average accuracy,
      precision@k, slice AUROC (wrap sklearn), Jaccard, recovery (NaN below a 2-point denominator).
- [ ] `stats.py`: percentile bootstrap (seeded, optionally stratified by group), one-sided Fisher exact
      test wrapper, Benjamini–Hochberg (own implementation), seed aggregation (mean, std).
- [ ] `evaluation/core.py`: join predictions with oracle groups; tidy group-metrics table with CIs.

**Tests:** property tests (Hypothesis): permutation invariance, bounds, `WGA ≤ mean-group ≤ max-group`;
metric values equal sklearn/scipy references on random inputs; BH equals a hand-worked example and
statsmodels; bootstrap CI coverage ≈ 95% ± 3% over 500 simulated datasets (slow); recovery guard.

**Gate G2:** all tests green; every metric in PRD §12 exists with a docstring stating the definition.

**Human checkpoint:** read `metrics.py` and `stats.py` line by line. Re-derive the BH procedure on
paper for five p-values.

---

## M3 — Training, embeddings and the first GPU handoff

**Goal:** trained models, predictions and embeddings for the headline configs, validated.

- [ ] `models/backbones.py`: resnet18, resnet50 (verify torchvision weights enums), tiny CNN;
      2-class head; `features()` returns penultimate activations.
- [ ] `training/erm.py`: constant LR default, checkpoint every epoch, exact resume (on CPU, see
      below), `history.csv`, selection by average val accuracy, predictions for all splits.
      Device policy (D-025): `auto` → mps > cuda > cpu; CUDA branch uses AMP + GradScaler, MPS
      branch uses float32 with no GradScaler, CPU branch uses float32.
      `PYTORCH_ENABLE_MPS_FALLBACK=1` when invoked locally with `device=mps`. DataLoader
      `num_workers=0` for smoke/test/CI configs, small default on macOS otherwise.
- [ ] `embeddings/model_space.py`, `embeddings/clip_space.py` (verify open_clip tags),
      `embeddings/store.py`, `embeddings/cache.py` (planted CLIP cache keyed by render spec).
      Verify open_clip's preprocessing is a no-op resize/crop on already-224×224 inputs (run code,
      not memory).
- [ ] `verification/reliance.py` + `slens reliance`: `R_net` with CIs (PRD FR-R1).
- [ ] `artifacts.py` HFHubStore; `slens pull-artifacts`; `slens validate-run`.
- [ ] `jobs.py` + `slens run-jobs`; `notebooks/gpu_runner.ipynb`; `docs/RUNBOOK_GPU.md` updated with exact cells.
- [ ] `make smoke`: synthetic build → train → embed (model space + a deterministic fake CLIP) on CPU.
- [ ] `slens evaluate` (non-final, val splits) → draft T1.
- [ ] **Handoff H1** (via `/handoff`): E1 configs — planted_pets ρ=0.95 size 32, seeds 0–2;
      planted_pets control ρ=0.5 size 32, seeds 0–2; waterbirds seeds 0–2. Train + both embeddings
      + reliance. First job measures seconds per epoch and writes it to the manifest.
- [ ] Pull, validate, and write measured timings into STATUS.

**Tests:** a tiny model overfits a batch of 8; resume from epoch k gives identical weights to an
uninterrupted run on CPU; feature dimensions; embedding id alignment; CLIP cache hit/miss;
reliance on synthetic data with a known answer; `validate-run` rejects a tampered manifest.

**Gate G3 (frozen thresholds):**
- planted_pets ρ=0.95, size 32 (mean over seeds): WGA gap ≥ 10 points on the balanced test split
  and `R_net` ≥ 0.20.
- planted_pets control ρ=0.5: `R_net` ≤ 0.05.
- waterbirds: WGA gap ≥ 10 points.
- If the planted condition fails: check for bugs first; then apply the fallback in DECISIONS D-002
  (raise ρ to 0.99 for the headline, then CIFAR-10 base) and tell the human. Do not change thresholds.

**Human checkpoint:** read `erm.py` thoroughly. Explain mixed precision, checkpoint/resume,
why selection uses average accuracy, and what `R_net` measures.

---

## M4 — Slice discovery and confirmation

**Goal:** the core tool works, and is proven on the planted shortcut.

- [ ] `discovery/base.py`, `discovery/preprocess.py` (L2 + PCA on val_a).
- [ ] `confidence.py`, `error_kmeans.py`, `domino_em.py` (from scratch), `failure_direction.py`.
- [ ] `discovery/confirm.py`: Fisher + BH + ranking on val_b.
- [ ] `discovery/select_combo.py` (FR-C4, D-024): label-free selection of the (method, space)
      combination that feeds naming, verification and `dfr_discovered`.
- [ ] `slens discover`, `slens confirm`.
- [ ] `evaluation/discovery_eval.py`: AUROC, precision@10/25, top-1 hit, best-match reference, on val_b and test.
- [ ] Run E2 and E4 on CPU from H1 artefacts, balanced and realistic val modes.

**Tests:** `domino_em` with `w=0` matches sklearn diagonal GMM log-likelihood within 1e-4 relative
(same initialisation); EM log-likelihood never decreases; seeded determinism for every method;
`failure_direction` recovers a planted direction in synthetic Gaussian data; under a null (random
labels of "incorrect") the false-confirmation rate ≤ `fdr_q` in simulation; **CI integration gate:**
synthetic_shapes `dot` ρ=0.95 with the tiny CNN → top confirmed slice precision@25 ≥ 0.8.

**Gate G4 (frozen thresholds):**
- planted_pets ρ=0.95 size 32, balanced val, in ≥ 2 of 3 seeds: for class *dog*, the top-ranked
  confirmed slice of at least one (method, space) has test precision@25 ≥ 0.8 and AUROC ≥ 0.8 for
  group (dog, patch). This gate deliberately checks the best of all method × space combinations
  against ground truth (oracle-selected) as a sanity check that the tool *can* find the shortcut at
  all — it is not the rule used to pick what feeds naming/verification/mitigation downstream. That
  is a separate, label-free rule, FR-C4 (D-024): implemented in `discovery/select_combo.py`.
- control ρ=0.5: no confirmed slice reaches precision@25 ≥ 0.8 for any patch group.
- Draft T2 produced (with all method × space combinations in an appendix, and the FR-C4-selected
  combination marked).

**Human checkpoint:** read `domino_em.py` and `confirm.py`. Derive the EM updates on paper. Explain
why confirmation happens on a different split than discovery.

---

## M5 — Naming

**Goal:** confirmed slices get human-readable names, with honest measures of reliability.

- [ ] `naming/freeze.py`: verify `vocab/FROZEN.sha256` (shipped with the spec package, checked in M0);
      refuse to run on mismatch unless `--unfreeze` with a DECISIONS id.
- [ ] `naming/clip_text.py`: templates + prompt ensembling + cache.
- [ ] `naming/vocabulary.py`: contrastive scores; top-5 positive and negative; variants `full`, `no_artifacts`.
- [ ] `naming/caption_keywords.py`: BLIP captions (verify model id), YAKE keywords, CLIP scoring vs
      incorrect/correct sets, support counts.
- [ ] **Handoff H2:** BLIP captions for `val_a` of planted_pets (each patch size) and waterbirds.
      (Can be scheduled together with H1 if the code is ready.)
- [ ] `evaluation/naming_eval.py`: hit@3 per group, including absence groups via negative phrases.
- [ ] Run E5 → T3.

**Tests:** on synthetic embeddings where the slice centroid equals a phrase vector plus noise, the
namer ranks that phrase first; negative-direction logic; support counts exact; YAKE deterministic;
freeze check refuses a modified vocab file.

**Gate G5:** T3 produced with support counts; frozen hash recorded in every naming manifest.
(No numeric pass bar — naming quality is a measured result, not a requirement.)

**Human checkpoint:** read `vocabulary.py`. Explain why the vocabulary namer can only find what
the phrase list contains, and how the `no_artifacts` variant measures that.

---

## M6 — Verification and the sensitivity sweep

**Goal:** causal checks for named slices, and the headline sensitivity map.

- [ ] `verification/interventions.py`: `patch_remove`, `patch_add`, `null_patch_add`; keyword → intervention registry.
- [ ] `verification/verify.py`: fix/break rates with CIs vs control and size-matched random sample; verdicts.
- [ ] Sweep config `configs/sweeps/planted_strength.yaml` (ARCHITECTURE §6) and `jobs/h3_sweep.yaml`.
- [ ] Estimate sweep cost from H1 timings. Present it with the ResNet-18 fallback option. **Ask before H3.**
- [ ] **Handoff H3:** sweep training + model-space embeddings + reliance (CLIP from cache).
- [ ] CPU: discover → confirm → evaluate for every sweep run, both val modes.
- [ ] F1 sensitivity map: x = `R_net`, y = precision@25 and AUROC of the top-ranked confirmed slice;
      marker = patch size; colour = space; facets = val mode; points = seeds.
- [ ] E6 verification table and F2.

**Tests:** interventions exact; verification logic on synthetic data with a known causal attribute;
run-jobs skips completed runs and resumes partial ones.

**Gate G6:** every sweep cell has ≥ 3 seeds; F1 and F2 generated; E6 table complete.

**Human checkpoint:** interpret F1 out loud. Explain why the x-axis is measured reliance rather than ρ.

---

## M7 — Mitigation and final evaluation

**Goal:** label-free fixes compared honestly against the oracle reference, on the test split, once.

- [ ] `mitigation/last_layer.py` (PyTorch L-BFGS, L1/L2, pull-to-init, weights, subset averaging).
- [ ] `label_free/ll_balanced.py`, `label_free/afr.py`, `label_free/dfr_discovered.py`, `selection.py`.
- [ ] `oracle_ref/dfr_oracle.py`, `oracle_ref/oracle_tuning.py`.
- [ ] `slens mitigate`; `slens evaluate --final` with the append-only log and rerun guard.
- [ ] Run E7 for both datasets, both val modes, 3 seeds → T4 (main) and T5 (cost of labels).

**Tests:** logistic regression matches sklearn on unweighted L2 problems; sample weights respected;
subset averaging correct; selection code has no oracle import (contract) and receives no group
columns (runtime assert); final log is append-only and the rerun guard works.

**Gate G7 (frozen thresholds):**
- `dfr_oracle` on waterbirds, balanced val: WGA ≥ 85% (sanity check against the literature; if it
  fails, investigate features/standardisation/regularisation — never tune on test).
- T4 and T5 complete with CIs for all cells.

**Human checkpoint:** read `last_layer.py`, `afr.py`, `selection.py`. Explain why last-layer
retraining works at all, and why balanced-val results flatter every method.

---

## M8 — Reporting, app, release, write-up

- [ ] `report/`: every table and figure from artefacts; `reports/RESULTS.md` auto-generated.
- [ ] UMAP plots (F3) and Grad-CAM overlays (F4), labelled as illustrations.
- [ ] Stability analysis E8 → T6 (if time allows).
- [ ] `app/explorer.py` + `slens export-demo`; licence check for any bundled images (DECISIONS).
- [ ] Release artefact bundle to Hugging Face Hub; `make reproduce-cpu`.
- [ ] README: one-paragraph pitch, demo GIF, quickstart, auto-inserted results table, reproduction,
      limitations. No numbers that are not in `results/`.
- [ ] `reports/writeup.md` (4–6 pages): every claim cites a table or figure; limitations section.
- [ ] Final `docs/CODE_TOUR.md`; STATUS closed out.

**Gate G8:** fresh clone → `make setup && make reproduce-cpu` reproduces committed tables exactly;
app runs locally and on Spaces; README complete; all boxes ticked or explicitly deferred.

**Human checkpoint:** walk through the whole repo using CODE_TOUR and LEARNING_PATH.

---

## Stretch (only after G7)

- **S1** Waterbirds `background_grey` intervention using CUB-200-2011 segmentation masks
  (verify pixel alignment with Waterbirds images first).
- **S2** Two simultaneous planted shortcuts (patch + colour tint): does fixing one amplify the other?
- **S3** Full JTT retraining as an additional baseline.
- **S4** CelebA replication.
- **S5** CLIP ViT-L/14 as a third embedding space.
