"""Naming hit@3 against `vocab/eval_keywords.yaml`.

Concept: a name is scored "correct" if any of its top-3 phrases contains a keyword for the target
group -- including a negative-phrase list for "absence" groups (e.g. naming what's *missing*)
(FR-X3).

Pipeline position: oracle zone.

Status: stub -- implemented in M5 (see docs/PLAN.md).
"""
