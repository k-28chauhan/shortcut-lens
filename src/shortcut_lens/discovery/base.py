"""Shared protocols for slice discovery: `SliceDiscoverer`, `FittedSlicer`.

Concept: every discovery method fits on a `ClassView` (one class's embeddings, correctness and
p_y on val_a) and returns a `FittedSlicer` that can score and assign membership for *any*
examples -- this is what lets confirmation and evaluation apply the rule to val_b and test
(see docs/ARCHITECTURE.md §3).

Pipeline position: label-free zone.

Status: stub -- implemented in M4 (see docs/PLAN.md).
"""
