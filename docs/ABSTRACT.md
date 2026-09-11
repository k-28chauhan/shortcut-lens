# ABSTRACT (draft v1 · 2026-09-11)

Reread this at the start of every week. If the work no longer matches it, either rewrite it
with a dated note explaining what we learned, or recognise drift and go back.

---

Image classifiers often rely on shortcuts that correlate with the label in training data, and the
resulting failures stay hidden behind high average accuracy. Automatic slice-discovery tools promise
to find these failure groups without group labels, but they are usually evaluated on a few benchmarks
whose validation sets conveniently contain many minority examples, rarely check whether the named
cause is real, and often tune hyperparameters using the very group labels they claim not to need.

We build an end-to-end pipeline — discover, confirm, name, verify, mitigate — under a strict
no-group-label protocol enforced in code, and evaluate it against shortcuts we plant ourselves with
controlled strength and visibility, as well as on Waterbirds. We ask **when automatic discovery works
and when it breaks**: how discovery quality depends on the model's measured reliance on the shortcut,
on the embedding space (the model's own features vs CLIP), and on whether the validation set is
group-balanced or realistic; whether counterfactual edits confirm the named cause; and how much of
the oracle last-layer-retraining gain can be recovered using discovered slices instead of true labels.

**What would falsify our claims.**
- If discovery quality does not increase with measured reliance, there is no threshold to report (H3a fails).
- If the confirmation step reports planted-patch slices when no shortcut was planted, the pipeline
  is not trustworthy (G4 control fails).
- If slice-driven retraining does not beat class-balanced last-layer retraining in realistic mode,
  discovery adds nothing for mitigation (H7b fails) — which we would report as a negative result.

**Contribution in one sentence.** The individual components exist (Domino, B2T, DFR, AFR); we
contribute a validated, label-free end-to-end pipeline and a measurement of where automatic failure
discovery stops working.
