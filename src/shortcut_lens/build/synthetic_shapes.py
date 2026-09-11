"""Procedurally generated circle-vs-square dataset with a planted shortcut, for CPU CI.

Concept: the smallest possible testbed for the whole pipeline. Images are generated (not
downloaded), so `make smoke` and CI can run the full discover-confirm-name-verify-mitigate loop
in minutes on a laptop with no GPU and no network. Two attribute variants: a small red `dot` or
a background colour tint, either of which can be planted with controlled rho.

Pipeline position: oracle zone. Feeds `build/tables.py`.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
