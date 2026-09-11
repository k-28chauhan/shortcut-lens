"""`dfr_discovered`: DFR-style group-balanced resampling using pseudo-groups from discovered slices.

Concept: true DFR balances by *true* group; this variant balances by class x confirmed-slice
membership instead (the FR-C4-selected combination, D-024) -- averaged over several random
balanced subsets, exactly as DFR does, but without ever reading a ground-truth group label
(FR-M3).

Pipeline position: label-free zone.

Status: stub -- implemented in M7 (see docs/PLAN.md).
"""
