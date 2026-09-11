"""CLIP text embeddings with prompt templates, ensembling and a cache.

Concept: scoring a phrase against a slice requires a text embedding in the same space as the
image embeddings -- prompt ensembling (averaging several template sentences per phrase) makes
that score less sensitive to the exact wording of any one template.

Pipeline position: label-free zone. Used by `naming/vocabulary.py`.

Status: stub -- implemented in M5 (see docs/PLAN.md).
"""
