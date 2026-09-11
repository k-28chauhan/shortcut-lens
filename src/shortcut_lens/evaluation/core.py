"""Joins predictions with oracle groups; tidy group-metrics table with bootstrap CIs.

Concept: the shared join-and-score step underneath every other evaluation module -- getting the
join right (matching ids, not silently dropping rows) is exactly the kind of thing that produces
a plausible-looking wrong number if it is wrong.

Pipeline position: oracle zone (FR-X1).

Status: stub -- implemented in M2 (see docs/PLAN.md).
"""
