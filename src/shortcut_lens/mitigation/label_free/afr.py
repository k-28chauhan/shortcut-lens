"""`afr` (Automatic Feature Reweighting, Qiu et al. 2023): upweight examples the model gets wrong.

Concept: sample weight `w_i = exp(-gamma * p_y,i)`, normalised so each class has equal total
weight, plus an L2 pull toward the original ERM head -- entirely label-free, using only the
model's own predicted probabilities to decide which examples matter more (FR-M2).

Pipeline position: label-free zone.

Status: stub -- implemented in M7 (see docs/PLAN.md).
"""
