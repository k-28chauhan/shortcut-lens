"""Oracle zone: dataset construction, shortcut planting, table writing.

Concept: this package is the only place ground-truth group labels are created. It renders
images (optionally painting a planted shortcut), decides train/val_a/val_b/test splits, and
writes `public.parquet` (image/class/split only) and `oracle.parquet` (group labels).

Pipeline position: oracle zone. Label-free code must never import from `build/` directly or
indirectly; datasets built here are injected into label-free stages only via `cli.py`
(see docs/ARCHITECTURE.md §1).
"""
