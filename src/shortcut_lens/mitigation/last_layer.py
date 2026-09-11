"""Shared last-layer logistic regression trainer: full-batch L-BFGS on standardised embeddings.

Concept: Deep Feature Reweighting (Kirichenko et al. 2023) showed the backbone usually already
learned the right features -- retraining only the final linear layer, on the right data, restores
most of the worst-group accuracy. This trainer is the shared engine every mitigation method
(`ll_balanced`, `afr`, `dfr_discovered`, `dfr_oracle`) configures differently (sample weights,
L1/L2, pull-to-init). Checked against sklearn on unweighted L2 problems.

Pipeline position: shared, pure -- no I/O, no group-label reads. Used by both zones.

Status: stub -- implemented in M7 (see docs/PLAN.md).
"""
