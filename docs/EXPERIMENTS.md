# EXPERIMENTS — shortcut-lens

**Pre-registration rule.** Each experiment below is specified before it runs. Once any result for an
experiment exists, changes to its spec go in the *Amendments* log at the bottom with a date and a
reason. Hypotheses are predictions, not requirements — a refuted hypothesis is a result, not a bug.

Shared defaults unless stated: ResNet-50, 3 seeds (0, 1, 2), metrics as defined in PRD §12,
95% bootstrap CIs for single runs, mean ± std across seeds.

---

## E1 — ERM baselines and measured reliance

- **Question:** Does ERM learn the planted shortcut, how strongly, and does Waterbirds show the expected gap?
- **Configs:** planted_pets (ρ=0.95, size 32); planted_pets control (ρ=0.5, size 32); waterbirds.
- **Measure:** average, weighted-average (waterbirds), group accuracies, WGA, WGA gap; `R_net` (planted).
- **Hypothesis H1:** ρ=0.95 shows a WGA gap ≥ 10 points and `R_net` ≥ 0.20; control shows `R_net` ≈ 0.
- **Outputs:** Table **T1** (ERM results), reliance table.
- **Gate:** G3.

## E2 — Planted discovery gate

- **Question:** Does the tool find a shortcut we know is there, and stay quiet when it is not?
- **Configs:** E1 planted runs, balanced val; all four methods × two spaces.
- **Measure:** for the top-ranked confirmed slice per class: precision@10/25, AUROC, top-1 hit; number
  of confirmed slices in the control.
- **Hypothesis H2:** ρ=0.95 → top-1 hit for (dog, patch) in at least one method/space; control → no hit.
- **Outputs:** part of **T2**.
- **Gate:** G4.

## E3 — Sensitivity map (headline experiment)

- **Question (RQ1, RQ2, RQ3):** How does discovery quality depend on measured reliance, embedding
  space, patch visibility and validation composition?
- **Grid:** ρ ∈ {0.5, 0.75, 0.9, 0.95, 0.99} × patch size ∈ {8, 16, 32} px × seeds {0, 1, 2} = 45 runs.
  Val modes: balanced and realistic. Methods: all four. Spaces: model, CLIP.
  **Pre-registered fallback (D-022):** after H1 measured timings are in, if the extrapolated cost of
  the full 45-run sweep (training + embeddings + reliance) exceeds 6 GPU-hours, run the sweep on
  ResNet-18 instead of ResNet-50; also run ResNet-18 bridge runs at the headline config (ρ=0.95,
  size 32, seeds 0–2) for comparability. Headline experiments (E1, E2, E4, E6, E7) stay ResNet-50.
- **Measure:** x = `R_net`; y = precision@25 and AUROC of the top-ranked confirmed slice for the dog
  class; also number of confirmed slices.
- **Hypotheses:**
  - **H3a** discovery quality increases monotonically with `R_net`, with a threshold-like transition.
  - **H3b** CLIP space degrades faster than model space as patch size shrinks.
  - **H3c** realistic val mode shifts the transition to higher reliance than balanced mode.
- **Outputs:** Figure **F1** (sensitivity map), CSV of all points.
- **Gate:** G6.

## E4 — Method × space comparison at headline configs

- **Question:** Which discovery methods and spaces work, on both datasets and both val modes?
- **Configs:** planted_pets (ρ=0.95, size 32) and waterbirds; balanced and realistic val.
- **Measure:** AUROC, precision@10/25, top-1 hit (top-ranked confirmed slice) and best-match reference.
- **Hypothesis H4:** error-aware methods (`domino_em`, `failure_direction`) beat `error_kmeans` and
  `confidence` in precision@25; the gap between top-ranked and best-match slices is larger in realistic mode.
- **Outputs:** Table **T2**.

## E5 — Naming

- **Question (RQ4):** How often are confirmed slices named correctly, and how vocabulary-dependent is it?
- **Slices named:** confirmed slices from the FR-C4-selected (method, space) combination per run
  (D-024) — the label-free choice, not the oracle-best-of-all used only for gate G4.
- **Namers:** `vocab` (full), `vocab` (no_artifacts), `caption_keywords`.
- **Measure:** hit@3 per target group; support counts for caption keywords; example names per slice.
- **Hypotheses:**
  - **H5a** full vocab names the patch slice; no_artifacts vocab does not.
  - **H5b** caption keywords name Waterbirds backgrounds more often than the planted patch.
- **Outputs:** Table **T3**.
- **Gate:** G5.

## E6 — Counterfactual verification

- **Question (RQ5):** Does removing the named cause fix the errors, beyond a same-size control edit?
- **Configs:** planted runs from E1 (and selected E3 cells); interventions `patch_remove`, `patch_add`,
  `null_patch_add`; size-matched random control sample. Waterbirds only in stretch S1. Verified slices
  are the top confirmed slice of the FR-C4-selected (method, space) combination per run (D-024).
- **Measure:** fix and break rates with CIs; verdicts.
- **Hypothesis H6:** fix rate for the top patch slice exceeds the null-patch control with non-overlapping CIs.
- **Outputs:** Figure **F2**, verification table.
- **Gate:** G6.

## E7 — Mitigation and the cost of labels

- **Question (RQ6):** How much of the oracle DFR gain can label-free last-layer methods recover?
- **Methods:** ERM; `ll_balanced`; `afr` (label-free tuned); `afr` (oracle tuned);
  `dfr_discovered` (label-free tuned); `dfr_discovered` (oracle tuned); `dfr_oracle`.
  `dfr_discovered`'s pseudo-groups use confirmed slices from the FR-C4-selected (method, space)
  combination per run (D-024).
- **Configs:** planted_pets (ρ=0.95, size 32) and waterbirds; balanced and realistic val; 3 seeds.
- **Measure (test, via `evaluate --final`):** average, WGA, recovery, cost of labels.
- **Hypotheses:**
  - **H7a** in balanced val mode, even `ll_balanced` improves WGA substantially (the balanced data does the work).
  - **H7b** in realistic mode, `dfr_discovered` and `afr` beat `ll_balanced`; the ordering between
    them is uncertain.
  - **H7c** cost of labels is larger in realistic mode than in balanced mode.
- **Outputs:** Tables **T4** (main) and **T5** (cost of labels).
- **Gate:** G7.

## E8 — Stability (if time allows)

- **Question:** Are discovered slices stable across settings?
- **Grid:** k ∈ {2, 4, 8}, PCA dim ∈ {32, 64, 128}, 3 seeds, headline configs.
- **Measure:** Jaccard overlap of top slices across settings; variation in precision@25.
- **Outputs:** Table **T6**.

---

## Figures and tables index

| ID | Content | Produced by |
|---|---|---|
| T1 | ERM results + reliance | E1 |
| T2 | Discovery quality by method × space × dataset × val mode | E2, E4 |
| T3 | Naming results with support counts | E5 |
| T4 | Mitigation: average, WGA, recovery | E7 |
| T5 | Cost of labels | E7 |
| T6 | Stability | E8 |
| F1 | Sensitivity map | E3 |
| F2 | Verification: fix/break rates vs control | E6 |
| F3 | UMAP of embeddings with slices (illustration) | M8 |
| F4 | Grad-CAM overlays per slice (illustration) | M8 |

---

## Compute budget (estimates — replace with measured values after H1)

**Local vs cloud.** The dev machine has an Apple Silicon GPU (MPS, 24 GB unified memory), not a
CUDA GPU. planted_pets ERM (both the headline and control configs) and its embeddings are
expected to run locally on MPS -- no cloud handoff needed for E1/E2/E4/E6/E7's planted_pets cells.
Waterbirds and the E3 sweep keep the cloud (Kaggle/Colab T4) estimates below as the expected path,
since 45 sweep runs plus Waterbirds are likely to exceed comfortable local session length even if
they would technically fit in 24 GB. Actual local (MPS) timings are re-measured at M3 and recorded
in `docs/STATUS.md` alongside the cloud numbers below, replacing "estimate" with "measured".

| Job | Estimate on one T4 (cloud) | Notes |
|---|---|---|
| planted_pets ERM, ResNet-50, ~2.4k train images, 20 epochs | ~5–10 min per run | 6 runs; expected to run locally on MPS instead, re-measured at M3 |
| waterbirds ERM, ResNet-50, 30 epochs | ~20–30 min per run | 3 runs in H1 (cloud) |
| Model-space embeddings, all splits | ~1–3 min per run | planted_pets expected locally on MPS |
| CLIP ViT-B/32 embeddings (planted cache per patch size, both versions of each val/test image) | ~5 min per patch size | Reused across ρ; expected locally on MPS |
| BLIP captions for val_a sets | ~10–20 min total | H2 |
| E3 sweep: 45 runs | ~4–8 h (ResNet-50); ~1–2 h (ResNet-18) | H3 (cloud); ask before launching |
| **Total (cloud fallback path)** | **~8–14 GPU hours** | Check current Kaggle/Colab quotas; local MPS runs reduce this in practice |

---

## Amendments log

| Date | Experiment | Change | Reason |
|---|---|---|---|
| — | — | — | — |
