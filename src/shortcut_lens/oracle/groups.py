"""The only reader of `oracle.parquet`: exposes group labels to oracle-zone code.

Concept: centralising the one place that opens `oracle.parquet` makes the label-free/oracle
boundary enforceable by a simple grep-based contract test in addition to import-linter
(see docs/TESTING.md §5).

Pipeline position: oracle zone. Used by `evaluation/`, `verification/`, `mitigation/oracle_ref/`.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
