# LEARNING PATH — for GS

Claude Code builds the code; this file is how you make it *yours*. Work through it milestone by
milestone as the code lands, not all at the end. The goal: explain every module, metric and decision
without notes, and rebuild the core pieces yourself.

---

## Phase 0 — Concepts to understand before reading code (week 1)

| Concept | What to be able to say | Where to learn |
|---|---|---|
| CNNs and fine-tuning | What a ResNet's layers do; what "penultimate features" are; why we start from ImageNet weights | CS231n notes (CNNs, transfer learning); PyTorch transfer-learning tutorial |
| Shortcut learning | Why models take shortcuts; the Waterbirds example | Geirhos et al. 2020 (sections 1–3) |
| Group robustness | Average vs worst-group accuracy; why balanced data helps | Sagawa et al. 2020 (intro, setup); DFR paper sections 1–4 |
| Embeddings and CLIP | What an embedding space is; how CLIP puts images and text in one space | OpenAI CLIP blog post; CLIP paper sections 1–2 |
| Clustering and mixtures | k-means vs Gaussian mixtures; what EM does | scikit-learn user guide (clustering, GMM); a StatQuest-style EM explainer |
| Hypothesis testing | p-values; why testing many slices needs FDR control | Benjamini–Hochberg explainer; Fisher's exact test intro |
| Bootstrap | Resampling to get confidence intervals | Any intro to bootstrap CIs |
| Slice discovery | What Domino and B2T do and what they don't claim | Domino paper §3; B2T paper §3 |

**Check yourself:** explain to a friend, in two minutes each, "why a 97%-accurate model can be
broken" and "how you'd find where it's broken without labels".

---

## Phase 1 — Read the code in pipeline order (as milestones land)

Use `docs/CODE_TOUR.md` for the order. For each module:

1. Read the module docstring and its tests first — tests tell you what the code promises.
2. Read the code top to bottom. Write down anything you can't explain.
3. Run it: every stage has a CLI command and a smoke config.
4. Answer the check-yourself question from CODE_TOUR out loud.
5. Ask Claude Code `/explain <path>` for anything still unclear.

Milestone-specific questions:

- **M0:** Why does a config need a hash? What breaks if two runs share a run id?
  What exactly would happen if `discovery/` imported `oracle/`?
- **M1:** Why is the patch painted at load time and not baked into JPEGs? Why is test always
  balanced but val has two modes? What is `is_minority` computed from?
- **M2:** Why is WGA ≤ mean-group accuracy always true? What does a bootstrap CI's coverage mean?
  Walk through BH on 5 p-values.
- **M3:** What does GradScaler do in mixed precision? Why must resume reproduce identical weights?
  What does `R_net` measure, and why subtract the null patch?
- **M4:** Derive the EM E-step and M-step for the error-aware mixture. Why confirm on `val_b`?
  What would happen to the false discovery rate without BH?
- **M5:** Why can't the vocabulary namer name something missing from its list?
  What does support count N protect against?
- **M6:** Read the sensitivity map: what does a point at high `R_net` but low precision mean?
- **M7:** Why does retraining only the last layer fix anything? Why do balanced-val results flatter
  every method? What is "cost of labels"?

---

## Phase 2 — Rebuild the core pieces yourself (the part that makes it yours)

Do these in a scratch folder without looking at the implementation, then compare against the
repo's tests (run the repo's tests against your version).

1. `worst_group_accuracy` and a stratified bootstrap CI.
2. Benjamini–Hochberg.
3. `add_patch` / `remove_patch` with deterministic positions.
4. The error-aware EM (start with plain diagonal GMM EM, then add the error term).
5. Class-balanced and group-balanced logistic regression on embeddings.
6. The contrastive CLIP naming score.

**Break things on purpose** (then revert):
- Import `oracle` from `discovery/` and watch CI fail.
- Shuffle embedding rows without shuffling ids and see which test catches it.
- Select the ERM checkpoint by worst-group accuracy and see how the "label-free" results inflate.

---

## Phase 3 — Interview preparation

Be ready to answer each in under two minutes, with a concrete example from your results:

1. Walk me through the project end to end.
2. How do you know your discovery tool works? (planted shortcut, control, confirmation split)
3. What's the difference between your project and Domino / B2T? (end-to-end, validated, sensitivity map, verification, strict no-labels protocol)
4. Why does the model learn the shortcut at all?
5. Why last-layer retraining? When would it fail?
6. How did you prevent label leakage? (firewall, import-linter, selection by average accuracy)
7. What's a result that surprised you, and how did you check it wasn't a bug?
8. What would you do with more compute or time? (S1–S5, multiple shortcuts, other modalities)
9. How would this run in production? (a slice report on every new model version, like a test report)
10. What are the limitations? (validation data must contain the failure; CLIP blind spots; vocabulary dependence; correlation ≠ causation without interventions; Grad-CAM reliability)

Keep a one-page cheat sheet with your final numbers from `results/` — fill it only from the CSVs.
