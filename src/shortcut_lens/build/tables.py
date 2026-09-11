"""Writes `public.parquet`, `oracle.parquet` and the build manifest for a dataset.

Concept: this is the boundary artefact of the oracle zone -- `public.parquet` exposes only
`example_id, dataset, split, y, class_name, image_ref`; `oracle.parquet` (group labels) is never
read outside `oracle/groups.py` (see docs/ARCHITECTURE.md §4 for exact schemas).

Pipeline position: oracle zone. Entry point for `slens build`.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
