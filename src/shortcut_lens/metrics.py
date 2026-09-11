"""Pure metric functions: accuracy, group accuracy, worst-group accuracy, precision@k, AUROC.

Concept: metrics are computed here from arrays the caller already has -- no group labels are
loaded inside this module. That is what lets the exact same functions be called from both zones:
`evaluation/core.py` (oracle zone, with true groups) and, later, anywhere label-free code needs a
metric that does not require groups at all (e.g. average accuracy for checkpoint selection, D-004).
Every metric here has a frozen definition in docs/PRD.md §12 -- changing one requires a
DECISIONS entry (CLAUDE.md §3 rule 4).

Pipeline position: shared, pure. No I/O, no dataset access.

Status: stub -- implemented in M2 (see docs/PLAN.md).
"""
