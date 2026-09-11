"""Statistics: percentile bootstrap CIs, one-sided Fisher exact test, Benjamini-Hochberg.

Concept: this is how the project turns a single-run number into an honest interval, and how it
controls the false discovery rate when testing many candidate slices at once (docs/EXPERIMENTS.md
pre-registers `fdr_q`). Benjamini-Hochberg is implemented from scratch (checked against
`statsmodels`) because it is small, central to the confirmation step, and a good learning
exercise (docs/LEARNING_PATH.md Phase 2).

Pipeline position: shared, pure. Used by `evaluation/core.py`, `discovery/confirm.py`,
`verification/verify.py`.

Status: stub -- implemented in M2 (see docs/PLAN.md).
"""
