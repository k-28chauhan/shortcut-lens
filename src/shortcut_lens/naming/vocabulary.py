"""`vocab` namer: contrastive CLIP score of slice centroid vs rest-of-class centroid.

Concept: a slice can only be named with a phrase that is in the vocabulary -- this method's
central limitation, and the reason for the `full` vs `no_artifacts` variants that measure how
much naming quality depends on the phrase list itself (FR-N1/N2, RQ4).

Pipeline position: label-free zone (FR-N1). Entry point for `slens name --namer vocab`.

Status: stub -- implemented in M5 (see docs/PLAN.md).
"""
