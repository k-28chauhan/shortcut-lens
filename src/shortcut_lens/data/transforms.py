"""Image transforms: train = horizontal flip + normalise; eval = normalise only.

Concept: random crops are deliberately excluded because they could cut a planted patch out of
frame, silently lowering the effective rho (D-009). One augmentation policy for every dataset
keeps method comparisons clean.

Pipeline position: label-free zone. Used by `training/erm.py` and `embeddings/model_space.py`.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
