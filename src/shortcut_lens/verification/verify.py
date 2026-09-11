"""Fix/break rates for interventions on confirmed slices, with bootstrap CIs vs a control.

Concept: fix rate = fraction of misclassified slice members that become correct after the
intervention; break rate = fraction of correct non-members that become wrong. A slice's cause is
`confirmed` only if the fix-rate CI's lower bound exceeds the control intervention's CI upper
bound (FR-V3/V4) -- a real effect, not one explainable by occlusion alone.

Pipeline position: oracle zone. Entry point for `slens verify`.

Status: stub -- implemented in M6 (see docs/PLAN.md).
"""
