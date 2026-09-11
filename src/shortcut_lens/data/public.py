"""Reads `public.parquet`: example_id, dataset, split, y, class_name, image_ref.

Concept: this is the only table label-free code may read. Loading validates required columns,
no duplicate example_id, and valid split names (see docs/ARCHITECTURE.md §4).

Pipeline position: label-free zone. Used by every label-free stage that needs class labels.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
