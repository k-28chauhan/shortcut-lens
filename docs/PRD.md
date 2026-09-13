# PRD — shortcut-lens

**Version** 1.0 · **Date** 2026-09-11 · **Owner** GS · **Builder** Claude Code (with human review at every milestone)

---

## 1. Summary

shortcut-lens is an end-to-end, tested pipeline that:

1. **Discovers** groups of images where a classifier fails ("slices") without using group labels.
2. **Confirms** each slice on held-out data, so random clusters are not reported as failures.
3. **Names** each confirmed slice in plain English (e.g. "birds on land", "a small coloured square").
4. **Verifies** the named cause with counterfactual edits (remove the suspected cause, see if the error disappears).
5. **Fixes** the failure by retraining only the model's last layer, using discovered slices instead of true group labels.
6. **Measures** all of the above against ground truth, including on shortcuts we plant ourselves with controlled strength.

The research question that makes it more than a re-implementation:
**when does automatic failure discovery work, and when does it break?**

---

## 2. Background (plain English)

**Shortcut learning.** Models often learn an easy cue that happens to correlate with the label in
training data instead of the real concept. In the Waterbirds benchmark, 95% of waterbirds appear on
water backgrounds and 95% of landbirds on land. A model trained on it learns "water background →
waterbird" and fails on waterbirds photographed on land.

**Why it hides.** Average accuracy stays high because most test images follow the majority pattern.
The failure only shows up when you measure accuracy per group (bird type × background) and look at
the worst group — **worst-group accuracy (WGA)**.

**Slice discovery.** In practice nobody hands you group labels. Slice-discovery methods look for
coherent clusters of errors in an embedding space (the model's own features, or CLIP's) and
report them as candidate failure groups.

**Naming.** A cluster is useless to a human unless it can be described. Vision-language models such
as CLIP let us score text phrases against a cluster, or caption its images and mine keywords.

**Mitigation.** Deep Feature Reweighting (DFR) showed that retraining only the final linear layer on
group-balanced data restores much of the worst-group accuracy, because the backbone usually already
learned the real features. Variants such as AFR do this without group labels by upweighting
examples the model gets wrong.

**The gap we target.** Existing tools are mostly evaluated on a handful of benchmarks, in conditions
where the validation set happens to contain many minority examples, rarely with a causal check of
the named cause, and often tune hyperparameters with group labels. We measure discovery under
controlled shortcut strength, in realistic validation conditions, with causal verification, under a
strict no-group-label protocol.

---

## 3. Problem statement

A practitioner has a trained image classifier, a validation set with class labels only, and no
knowledge of which hidden attributes the model relies on. They need to know: *where does my model
fail, why, and can I fix it without labelling attributes by hand?* They also need to know **how much
to trust** the answer a discovery tool gives them.

---

## 4. Goals and non-goals

### Goals
- **G1** A reusable Python library + CLI that runs discover → confirm → name → verify → mitigate on
  any model/dataset that fits the interfaces.
- **G2** A ground-truth evaluation harness: planted shortcuts with controlled strength and visibility,
  plus Waterbirds.
- **G3** A sensitivity map: discovery quality as a function of how much the model actually relies on
  the shortcut, per embedding space and validation-set composition.
- **G4** A causal verification step for named slices.
- **G5** A label-free mitigation comparison, including the measured cost of not having group labels.
- **G6** Public artefacts: repo with tests and CI, reproducible tables from released artefacts,
  a Gradio slice explorer on Hugging Face Spaces, and a 4–6 page write-up.

### Non-goals
- Inventing a new state-of-the-art discovery or debiasing method.
- Large-scale training, training from scratch, or architectures beyond ResNet-18/50.
- Text, tabular, audio or medical data.
- Generative image editing (inpainting, diffusion) for counterfactuals.
- A production web service, authentication, or user uploads in the app.
- Anything related to LLM inference or KV caches (separate long-term project).

---

## 5. Users and use cases

| User | Use case |
|---|---|
| ML practitioner | Point the CLI at a model + validation set; get ranked, named, confirmed failure slices and a last-layer fix. |
| Reviewer / recruiter | Open the Spaces demo, browse slices, see the planted-shortcut proof and results table in the README. |
| GS (owner) | Learn the full stack: data engineering, training, embeddings, clustering, statistics, evaluation design. |

---

## 6. Research questions

- **RQ1 (sensitivity).** How does discovery quality change with the model's *measured reliance* on a
  planted shortcut (not just its correlation strength ρ)? Is there a threshold below which discovery fails?
- **RQ2 (embedding space).** Does discovery work better in the model's own feature space or in CLIP
  space, and does that depend on shortcut visibility (patch size)?
- **RQ3 (validation composition).** How much worse is discovery when the validation set has realistic
  (training-like, rare-minority) composition instead of the group-balanced composition common in benchmarks?
- **RQ4 (naming).** How often do vocabulary-based and caption-keyword namers produce a correct name,
  and how dependent is the vocabulary approach on the phrase list we supply?
- **RQ5 (verification).** Do counterfactual interventions confirm the named cause, relative to a
  control intervention of the same size?
- **RQ6 (mitigation).** How much of the worst-group gain from DFR with true group labels can label-free
  methods recover, and what does the no-labels rule cost?

Hypotheses are stated per experiment in `docs/EXPERIMENTS.md` before results exist.

---

## 7. Datasets

| ID | Role | Source | Classes | Spurious attribute | Notes |
|---|---|---|---|---|---|
| `synthetic_shapes` | CI fixture; CPU gate | Procedurally generated, 32×32 | circle vs square | small red dot, or background tint | Tiny CNN learns the shortcut in seconds on CPU. |
| `planted_pets` | Ground-truth testbed (real-photo resolution) | Oxford-IIIT Pet (torchvision), binary cat vs dog | cat, dog | coloured square patch we paint | Controlled ρ, patch size, colour, position. Natural images at usable resolution; licence reported as CC BY-SA 4.0 (verify). **Failed gate G3 at ρ=0.95 and ρ=0.99** (D-037): an ImageNet-pretrained backbone solves cat-vs-dog almost immediately, so the patch is never needed. Kept as a documented null result. |
| `planted_cifar_pets` | Main ground-truth testbed (D-002/D-037 fallback) | CIFAR-10 (torchvision), binary cat vs dog | cat, dog | coloured square patch we paint | Same planting code and construction as `planted_pets`; images upscaled 32×32→224×224 *before* the patch is painted, so the patch stays exactly as crisp/large -- isolating "does lower real-image information restore shortcut reliance" as the only changed variable. |
| `waterbirds` | Natural benchmark | Waterbirds (Sagawa et al.), via WILDS download or original tarball | landbird, waterbird | land vs water background | Official splits. Validation set is group-balanced within class — we also build a realistic variant. |

**Group encoding** (all datasets): `group = 2 * y + attribute`. Minority groups are determined from
training-set proportions and stored in the oracle table, not hardcoded.

**PlantedPets construction.**
- Train: official `trainval` split, class-balanced by subsampling the larger class.
  Patch assignment: `P(patch | cat) = ρ`, `P(patch | dog) = 1 − ρ`.
  So the minority (failure) groups are *cat without patch* and *dog with patch*.
- Validation and test: official `test` split, class-balanced, split 50/50 into val and test.
  Val is split 50/50 into `val_a` and `val_b`, stratified by class only.
- Test is always group-balanced within class (patch on 50% of each class).
- Val is either `balanced` (50/50 within class) or `realistic` (same proportions as train).
- Images are preprocessed once (resize shorter side 256 → centre crop 224) and cached.
  The patch is painted at load time, after decoding, so JPEG compression never touches it.
- Defaults: patch size 32 px, colour magenta (255, 0, 255), alpha 1.0, uniformly random position
  per image (deterministic from example id + seed).
- Fallback if the model does not learn the shortcut strongly enough (see gate G3): CIFAR-10 cat vs
  dog with the same planting code (D-002) -- **triggered** at both ρ=0.95 and ρ=0.99 (D-037); see
  `planted_cifar_pets` below.
- **Control condition:** ρ = 0.5 (no patch–class correlation), same patch settings.

**PlantedCifarPets construction (D-002/D-037 fallback).** Identical to PlantedPets above, except:
- Source is CIFAR-10 (torchvision), official train split (5000 cat / 5000 dog, already
  class-balanced) and test split (1000 cat / 1000 dog), used the same way as PlantedPets' `trainval`
  and `test` splits.
- Images are upscaled from native 32×32 to 224×224 *before* the patch is painted (same
  `build/planting.py` functions, same defaults), so the patch is exactly as large/crisp as in
  PlantedPets -- the only changed variable is how much real information the underlying photo
  carries.

**Waterbirds.**
- Official train/val/test splits. Expected group counts (verify at build time):
  train 3498 / 184 / 56 / 1057; val 467 / 466 / 133 / 133; test 2255 / 2255 / 642 / 642
  (landbird-land / landbird-water / waterbird-land / waterbird-water).
- Val is split into `val_a` / `val_b` stratified by class only.
- `realistic` val mode subsamples minority groups to training-like proportions (configurable
  `minority_fraction`, default = training proportion).
- Report both plain average test accuracy and training-distribution-weighted average accuracy
  (the test set is balanced, so plain average overstates minority importance). WGA is unaffected.

**Split roles (all datasets).**

| Split | Used for | Never used for |
|---|---|---|
| `train` | ERM training | anything else |
| `val_a` | fitting slice discovery; last-layer retraining data | final reporting |
| `val_b` | slice confirmation; label-free hyperparameter selection; ERM checkpoint selection (with val_a, average accuracy only) | final reporting |
| `test` | final reporting via `evaluate --final`; discovery-quality scoring | any selection or tuning |

---

## 8. System overview

```
build → train (ERM) → embed (model + CLIP) → reliance* → discover → confirm → name
      → verify* → mitigate → evaluate* → report → export-demo
                                     (* = oracle zone: may read ground-truth groups)
```

Every stage is a CLI command, reads and writes artefacts under `artifacts/`, writes a manifest,
and is idempotent (skips if outputs exist with a matching config hash, unless `--force`).

---

## 9. Functional requirements

### FR-D Data
- **FR-D1** `slens build --dataset {synthetic_shapes|planted_pets|waterbirds}` produces
  `public.parquet` (example_id, dataset, split, y, class_name, image_ref) and `oracle.parquet`
  (example_id, attribute, group, group_name, is_minority, plus planting parameters for planted data).
- **FR-D2** Planting is deterministic: same config + seed → identical images, bit for bit.
- **FR-D3** Patch operations: `add`, and `remove` (re-render from the clean base). `remove(add(x)) == x`.
- **FR-D4** Datasets handed to label-free code return exactly `{"image", "y", "example_id"}`.
- **FR-D5** `slens data report` writes group counts per split and a sample image grid per dataset.

### FR-T Training
- **FR-T1** ERM fine-tuning of ImageNet-pretrained ResNet-50 (ResNet-18 allowed for sweeps; tiny CNN for synthetic).
- **FR-T2** Mixed precision on GPU, checkpoint every epoch, exact resume after interruption.
- **FR-T3** Checkpoint selection by average validation accuracy only.
- **FR-T4** Writes per-split predictions: y, y_hat, p_y (probability of true class), p_max, logits, correct.

### FR-E Embeddings
- **FR-E1** Model space: penultimate-layer features (after global average pooling).
- **FR-E2** CLIP space: open_clip ViT-B/32 image embeddings, L2-normalised. ViT-L/14 optional.
- **FR-E3** Stored as float16 `.npy` plus an aligned id file; alignment is validated on load.
- **FR-E4** CLIP embeddings for planted data are cached per (image, patch spec, has_patch), so sweeps
  over ρ reuse them.

### FR-S Slice discovery (label-free)
Operates per class on `val_a`. Every method outputs ranked `Slice` objects plus a scoring function
that gives a continuous membership score for any example (needed for AUROC).
- **FR-S1** `confidence`: the misclassified / low-confidence set (JTT-style baseline).
- **FR-S2** `error_kmeans`: k-means on embeddings of all class examples; clusters ranked by error rate.
- **FR-S3** `domino_em`: error-aware diagonal Gaussian mixture (Domino-style, per-class simplification):
  each component also models the probability of being misclassified, with a weight `w` on that
  likelihood term. Implemented from scratch with EM.
- **FR-S4** `failure_direction`: per-class linear SVM separating correct from incorrect examples;
  the slice is the direction of the SVM normal (Jain et al.).
- **FR-S5** Preprocessing: L2 normalisation and PCA fitted on `val_a` only.
- **FR-S6** Number of slices per class and PCA dimension are configs, chosen without group labels.

### FR-C Confirmation (label-free)
- **FR-C1** Apply each slice's membership rule to `val_b`; compare the slice's error rate with the rest
  of its class using a one-sided Fisher exact test.
- **FR-C2** Benjamini–Hochberg correction across all candidate slices; confirmed if q ≤ `fdr_q`
  (default 0.10) and size ≥ `min_size` (default 20).
- **FR-C3** Global ranking of confirmed slices by q-value, then effect size.
- **FR-C4** Label-free selection of the (method, space) combination that feeds naming, verification
  and `dfr_discovered`: the combination whose top confirmed slice has the largest error-rate lift
  over the rest of its class on `val_b`, tie-break by smallest q-value (D-024). Uses only `val_b`
  error rates and q-values, no group labels. All combinations still appear in an appendix table.

### FR-N Naming (label-free)
- **FR-N1** `vocab` namer: CLIP text embeddings of phrases from `vocab/generic_phrases.yaml` (prompt
  ensembling over templates). Contrastive score: similarity to slice centroid minus similarity to
  rest-of-class centroid. Reports top-5 positive and top-5 negative phrases.
- **FR-N2** Two vocabulary variants: `full` and `no_artifacts` (drops artefact and colour phrases),
  to measure vocabulary dependence.
- **FR-N3** `caption_keywords` namer (B2T-style): caption slice and class images with BLIP, extract
  keywords (YAKE), rank keywords by CLIP score against misclassified vs correctly classified images,
  and report support count N for every keyword.
- **FR-N4** Vocabulary files are frozen: their sha256 is stored in `vocab/FROZEN.sha256`, recorded in
  every naming manifest, and the CLI refuses to run if they changed without an explicit override.

### FR-V Verification (oracle zone, analyst step)
- **FR-V1** Interventions: `patch_remove`, `patch_add`, `null_patch_add` (grey square of the same size,
  to control for occlusion). Stretch: `background_grey` for Waterbirds using CUB segmentation masks.
- **FR-V2** A small registry maps name keywords to interventions.
- **FR-V3** Metrics: fix rate (fraction of misclassified slice members that become correct after the
  intervention) and break rate (fraction of correct non-members that become wrong), each with bootstrap
  CIs, against the control intervention and a size-matched random sample.
- **FR-V4** Verdict `confirmed` if the lower CI bound of the fix rate exceeds the upper CI bound of the control.

### FR-R Reliance measurement (oracle zone)
- **FR-R1** For planted data: break rate when adding the patch to correctly classified dogs without
  a patch, and when removing it from correctly classified cats with one; minus the same rates for the
  null patch. `R_net` is the mean of the two net rates. This is the x-axis of the sensitivity map.

### FR-M Mitigation
Last-layer retraining on frozen model-space embeddings, trained on `val_a`:
- **FR-M1** `ll_balanced` (label-free): class-balanced logistic regression. Control.
- **FR-M2** `afr` (label-free): weights `∝ exp(−γ · p_y)` within class, L2 pull towards the original head.
- **FR-M3** `dfr_discovered` (label-free): DFR-style group-balanced subsampling using pseudo-groups
  (class × confirmed-slice membership), averaged over several random subsets.
- **FR-M4** `dfr_oracle` (oracle reference): the same with true groups.
- **FR-M5** Label-free hyperparameter selection: worst pseudo-group accuracy on `val_b`, tie-break by
  average accuracy. Oracle-tuned variants (true WGA on `val_b`) are reported as separate reference rows.

### FR-X Evaluation (oracle zone)
- **FR-X1** Group metrics for every model and mitigation, per split, with bootstrap CIs.
- **FR-X2** Discovery quality: AUROC and precision@k of the top-ranked confirmed slice against the
  true minority group of its class; also the best-matching slice (oracle-chosen upper reference).
- **FR-X3** Naming hit@3 against `vocab/eval_keywords.yaml`.
- **FR-X4** `--final` runs on test are append-logged; re-runs need a written reason.

### FR-P Reporting and app
- **FR-P1** `slens report` regenerates every table (CSV + Markdown) and figure from artefacts, and
  writes `reports/RESULTS.md` with numbers pulled from the CSVs.
- **FR-P2** UMAP plots and Grad-CAM overlays for a sample of slice members, labelled as illustrations.
- **FR-P3** Gradio slice explorer reading only a precomputed demo bundle (≤ 200 MB), deployable on a
  free CPU Hugging Face Space. Default bundle uses PlantedPets.

### FR-O Operations
- **FR-O1** Typer CLI `slens`, Pydantic-validated YAML configs, stable config hashes.
- **FR-O2** Manifests for every stage run; `slens validate-run` checks them.
- **FR-O3** Artifact store with `local` and `hf_hub` backends; `slens pull-artifacts`.
- **FR-O4** `slens run-jobs jobs/<file>.yaml` runs a list of stage commands with resume, for GPU sessions.
- **FR-O5** `make reproduce-cpu` regenerates all tables and figures from released artefacts on a CPU.

---

## 10. Non-functional requirements

- **Reproducibility:** fixed seeds, deterministic CPU paths, manifests, pinned dependencies,
  results regenerated from artefacts. GPU runs may differ at floating-point level; conclusions must not.
- **Compute:** every GPU job fits a single 16 GB T4 (fp16; no bfloat16, no FlashAttention).
  Total GPU budget target ≤ 15 hours. Everything after embeddings runs on a laptop CPU.
- **Resilience:** any GPU job can be interrupted and resumed with no lost epochs beyond the current one.
- **Performance:** `make smoke` < 3 min on CPU; `make check` < 5 min; CI < 10 min.
- **Quality:** mypy strict, ruff clean, test coverage ≥ 85% on `src/` excluding `app/`.
- **Security:** no secrets in the repo; tokens via environment variables only.
- **Licensing:** dataset licences checked before publishing any images in the demo; decision logged.

---

## 11. Integrity requirements

See `CLAUDE.md` §3. In summary: group-label firewall enforced by import-linter; no group labels in
any label-free choice; test split only via `evaluate --final`; frozen vocab, thresholds and metric
definitions; no fabricated numbers; surprising results investigated; everything traceable to a manifest.

---

## 12. Metric definitions (frozen)

| Metric | Definition |
|---|---|
| Average accuracy | Fraction correct over a split. |
| Weighted average accuracy (Waterbirds) | Per-group test accuracies weighted by training-set group proportions. |
| Group accuracy | Accuracy within one group. |
| Worst-group accuracy (WGA) | Minimum group accuracy over the four groups. |
| WGA gap | Average accuracy − WGA. |
| Precision@k | Among the k examples of a class with the highest slice score, the fraction in the target minority group. k ∈ {10, 25}. |
| Slice AUROC | AUROC of the slice score for identifying target-group membership among examples of that class. |
| Top-1 hit | The top-ranked confirmed slice of the class has precision@25 ≥ 0.8 and AUROC ≥ 0.8. |
| Naming hit@3 | Any of the top-3 names contains a keyword from `vocab/eval_keywords.yaml` for the target group (negative-phrase list for "absence" groups). |
| Reliance `R_net` | See FR-R1. |
| Fix / break rate | See FR-V3. |
| Recovery | `(WGA_method − WGA_ERM) / (WGA_dfr_oracle − WGA_ERM)`; reported as NaN if the denominator is under 2 points. |
| Cost of labels | `WGA_oracle_tuned − WGA_label_free_tuned` for the same method. |

Uncertainty: 95% percentile bootstrap CIs (1,000 resamples, stratified by group) for single runs;
mean ± std across 3 seeds for multi-seed results.

---

## 13. Success criteria (project level)

1. All milestone gates G0–G8 in `docs/PLAN.md` pass.
2. Experiments E1–E7 complete with uncertainty estimates; E8 if time allows.
3. Fresh clone → `make setup && make reproduce-cpu` regenerates the committed tables exactly.
4. The Spaces demo loads and shows ranked, named, confirmed slices with Grad-CAM illustrations.
5. The write-up states every claim with the table or figure that supports it, and a limitations section.
6. GS can explain every module, metric and decision without notes (see `docs/LEARNING_PATH.md`).

---

## 14. Deliverables

- Repository `shortcut-lens` with library, CLI, tests, CI, docs.
- Released artefact bundle (embeddings, predictions, slices, names, metrics) on Hugging Face Hub.
- Tables T1–T6 and figures F1–F4 (see `docs/EXPERIMENTS.md`).
- `reports/writeup.md` (4–6 pages).
- Hugging Face Space with the slice explorer; demo GIF in README.

---

## 15. Scope

| In scope | Out of scope | Stretch (only after G7 passes) |
|---|---|---|
| Binary image classification | Multi-label, detection, segmentation | S1 Waterbirds background counterfactual (CUB masks) |
| ResNet-18/50 fine-tuning | Training from scratch; ViTs as the classifier | S2 Two simultaneous planted shortcuts (whack-a-mole) |
| 4 discovery methods × 2 spaces | New discovery algorithms | S3 Full JTT retraining |
| Vocab + caption namers | LLM-generated hypotheses | S4 CelebA replication |
| Patch interventions | Generative image editing | S5 CLIP ViT-L/14 as a third space |
| Last-layer mitigation | Group DRO, full retraining methods | |
| Gradio demo on bundle | User uploads, auth, hosted training | |

---

## 16. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Pets model barely uses the patch | Medium | Gate G3 measures `R_net`; raise ρ or size; fallback to CIFAR-10 cat vs dog (logged). |
| Waterbirds download / WILDS API issues | Medium | Thin own loader over `metadata.csv`; direct tarball fallback. |
| BLIP captions ignore the patch | High | That is a finding; the vocab namer and verification still run. |
| Colab/Kaggle disconnects | High | Epoch checkpoints, idempotent stages, run-jobs resume. |
| Discovery fails in realistic val mode | Medium | That is a finding (RQ3); report it. |
| AFR beats `dfr_discovered` | Medium | That is a finding (RQ6); report it. |
| Scope creep | High | PRD non-goals; stretch only after G7. |
| Hallucinated library APIs | Medium | CLAUDE.md rule: verify by running code; tests. |

---

## 17. Glossary

- **Slice:** a subset of examples defined by a membership rule, e.g. a cluster in embedding space.
- **Spurious attribute / shortcut:** a feature correlated with the label in training but not causally related.
- **ERM:** empirical risk minimisation — ordinary training that minimises average loss.
- **Oracle:** anything that uses ground-truth group labels; allowed only for evaluation and references.
- **Pseudo-group:** a group defined by discovered slice membership instead of true labels.
- **FDR / BH:** false discovery rate; Benjamini–Hochberg procedure for controlling it across many tests.

---

## 18. References (reading guide — verify venue, year and title before citing)

- Geirhos et al., "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence 2020.
- Sagawa et al., "Distributionally Robust Neural Networks for Group Shifts" (Waterbirds), ICLR 2020.
- Koh et al., "WILDS: A Benchmark of in-the-Wild Distribution Shifts", ICML 2021.
- Liu et al., "Just Train Twice" (JTT), ICML 2021.
- Kirichenko, Izmailov, Wilson, "Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations" (DFR), ICLR 2023.
- Qiu et al., "Simple and Fast Group Robustness by Automatic Feature Reweighting" (AFR), ICML 2023.
- LaBonte et al., "Towards Last-layer Retraining for Group Robustness with Fewer Annotations", NeurIPS 2023.
- Sohoni et al., "No Subclass Left Behind" (GEORGE), NeurIPS 2020.
- Eyuboglu et al., "Domino: Discovering Systematic Errors with Cross-Modal Embeddings", ICLR 2022.
- d'Eon et al., "The Spotlight", FAccT 2022.
- Jain et al., "Distilling Model Failures as Directions in Latent Space", ICLR 2023.
- Kim et al., "Discovering and Mitigating Visual Biases through Keyword Explanation" (B2T), CVPR 2024.
- Yenamandra et al., "FACTS: First Amplify Correlations and Then Slice", ICCV 2023.
- Li et al., "A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others", CVPR 2023.
- Radford et al., CLIP, ICML 2021. Li et al., BLIP, ICML 2022.
- Selvaraju et al., Grad-CAM, ICCV 2017. Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018.
- Parkhi et al., Oxford-IIIT Pet, CVPR 2012. Wah et al., CUB-200-2011, 2011.
- Benjamini & Hochberg, "Controlling the False Discovery Rate", JRSS-B 1995.
- McInnes et al., UMAP, 2018.
