"""Stratified train/val_a/val_b/test splitting; balanced vs realistic validation composition.

Concept: `val_a` (discovery, last-layer retraining) and `val_b` (confirmation, selection) must be
disjoint so that discovering and confirming on the same data cannot let random clusters look like
real failures (D-005). `balanced` validation is 50/50 within class; `realistic` matches the
training distribution's minority proportions, which is harder and more true-to-practice (D-006).

Pipeline position: oracle zone. Used by every `build/<dataset>.py` module.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
