# TESTING — shortcut-lens

Tests are how we know a number is real. In this project a silent bug (misaligned ids, a leaked
label, a wrong split) produces a plausible-looking wrong result, so tests carry more weight than
in a typical app.

## 1. Test layers

| Layer | Folder | What it checks | Speed |
|---|---|---|---|
| Unit | `tests/unit/` | one function at a time, known inputs → known outputs | ms |
| Property | `tests/property/` | invariants over many random inputs (Hypothesis) | ms–s |
| Contract | `tests/contracts/` | zone boundaries, dataset key sets, schemas | s |
| Integration | `tests/integration/` | stages chained on `synthetic_shapes` on CPU | < 3 min total |
| Slow | marker `slow` | real CLIP/BLIP on a few images, coverage simulations | minutes |
| Network | marker `network` | dataset downloads, expected Waterbirds counts | minutes |
| GPU | marker `gpu` | never in CI; optional sanity checks in the GPU notebook | — |

CI runs: `pytest -m "not slow and not gpu and not network"` plus ruff, mypy, lint-imports.
`make test-slow` runs slow and network tests locally.

## 2. Fixtures (in `tests/conftest.py`)

- `tiny_synthetic_build`: synthetic_shapes with 400 images per split, ρ=0.95, `dot` variant.
- `fake_clip`: deterministic embedder (fixed random projection of downsampled pixels) so pipelines
  run without downloading CLIP. Real CLIP is used only in slow tests.
- `fake_text_embedder`: maps phrases to fixed vectors; used for naming unit tests.
- `random_predictions(n, groups, seed)`: synthetic prediction tables for metric tests.
- `tmp_artifacts`: temporary artefact root.

## 3. Required tests by module

| Module | Must test |
|---|---|
| `config.py` | hash stable and key-order independent; composition; validation errors are readable |
| `manifest.py` | round-trip; `validate-run` rejects mismatched commit, config hash, missing files |
| `seeding.py` | repeatable draws across numpy and torch |
| `build/planting.py` | `remove(add(x)) == x`; positions in bounds; deterministic; null patch differs only in colour |
| `build/pets.py`, `waterbirds.py`, `splits.py` | split disjointness; class balance; planted proportions; expected Waterbirds counts (network) |
| `build/datasets.py` | returns exactly `{"image","y","example_id"}`; image shape and dtype |
| `metrics.py` | vs sklearn/scipy references; bounds; permutation invariance; recovery guard |
| `stats.py` | BH vs hand example and statsmodels; bootstrap coverage ≈ 95% (slow); Fisher wrapper direction |
| `training/erm.py` | overfit a batch; exact resume on CPU; selection never reads group columns |
| `embeddings/*` | id alignment; float16 round-trip tolerance; cache hit/miss |
| `discovery/domino_em.py` | `w=0` equals sklearn diag GMM; log-likelihood non-decreasing; determinism |
| `discovery/failure_direction.py` | recovers planted direction on synthetic Gaussians |
| `discovery/confirm.py` | null simulation false-confirmation rate ≤ `fdr_q`; ranking order |
| `naming/*` | planted phrase ranked first on synthetic embeddings; negative logic; support counts; freeze check |
| `verification/*` | interventions exact; known-cause synthetic verification; reliance on synthetic |
| `mitigation/last_layer.py` | matches sklearn on unweighted L2; weights respected; subset averaging |
| `mitigation/selection.py` | receives no group columns (runtime assert) |
| `evaluation/final.py` | append-only log; rerun guard |
| `jobs.py` | skips completed runs; resumes partial ones; stops on first failure |

## 4. The integration gate (runs in CI)

`tests/integration/test_synthetic_pipeline.py` runs build → train (tiny CNN) → embed (model space
and fake CLIP) → discover → confirm → name (fake text embedder) → mitigate → evaluate on
synthetic_shapes with ρ=0.95 and asserts:

- the top confirmed slice of the affected class has precision@25 ≥ 0.8 for the planted group;
- with ρ=0.5 no slice reaches that bar;
- `dfr_oracle` WGA ≥ ERM WGA;
- every stage wrote a valid manifest.

This is the single most important test: it proves the plumbing end to end on every commit.

## 5. Contract tests

- `lint-imports` passes (the zone contract).
- A test enumerates every `ImageDataset` implementation and checks its key set.
- A test asserts no module in the label-free zone reads a file named `oracle.parquet`
  (grep-style check over source files as a second line of defence).

## 6. Coverage and style

- Coverage ≥ 85% on `src/shortcut_lens` excluding `app/`.
- Test names describe behaviour: `test_remove_patch_restores_original_image_exactly`.
- No network, GPU or large downloads in default tests.
