"""Shared dataclasses and Protocols used across both zones: `ImageDataset`, `Slice`, `Predictions`.

Concept: these types are the data contracts at the seams between stages (see docs/ARCHITECTURE.md
§3-4) -- e.g. every `ImageDataset` implementation, regardless of which real dataset it wraps, must
return exactly `{"image", "y", "example_id"}`, which is what makes it safe to pass into label-free
code (D-003).

Pipeline position: shared, pure. Imported by both zones; contains no logic and no I/O.

Status: stub -- populated as each type is first needed, starting with `ImageDataset` in M1.
"""
