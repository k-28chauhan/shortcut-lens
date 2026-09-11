"""`failure_direction`: a linear SVM direction separating correct from incorrect examples.

Concept: instead of clustering, this method (Jain et al. 2023) finds the single direction in
embedding space that best separates correct from incorrect predictions; the "slice" is examples
far along that direction, at the top q% by decision value.

Pipeline position: label-free zone (FR-S4).

Status: stub -- implemented in M4 (see docs/PLAN.md).
"""
