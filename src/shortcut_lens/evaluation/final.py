"""`slens evaluate --final`: append-only test-split evaluation with a rerun-reason guard.

Concept: the test split is for final reporting only -- nothing is ever selected or tuned on it
(CLAUDE.md §3 rule 3). Every final run appends to `results/final_eval_log.csv`; re-running the
same config requires a written `--rerun-reason`, so it is always visible if a number changed
between runs and why.

Pipeline position: oracle zone (FR-X4).

Status: stub -- implemented in M7 (see docs/PLAN.md).
"""
