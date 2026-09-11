"""Planted CLIP embedding cache, keyed by (image, patch spec, has_patch).

Concept: CLIP embeddings for planted data only need recomputing when the render (patch size,
colour, position) changes, not for every rho -- caching by render spec lets the rho sweep (E3)
reuse embeddings across its 5 rho values (FR-E4).

Pipeline position: label-free zone. Used by `embeddings/clip_space.py`.

Status: stub -- implemented in M3 (see docs/PLAN.md).
"""
