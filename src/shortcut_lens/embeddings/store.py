"""Stores embeddings as float16 `.npy` plus an aligned id file; validates alignment on load.

Concept: a silent id misalignment between an embedding table and the predictions/oracle table it
is joined with would produce a plausible-looking wrong result -- this is exactly the kind of bug
docs/TESTING.md exists to catch.

Pipeline position: label-free zone. Used by `embeddings/model_space.py` and `clip_space.py`.

Status: stub -- implemented in M3 (see docs/PLAN.md).
"""
