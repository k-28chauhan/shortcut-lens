"""Shared dataclasses and Protocols used across both zones: `ImageDataset`, `Slice`, `Predictions`.

Concept: these types are the data contracts at the seams between stages (see docs/ARCHITECTURE.md
§3-4) -- e.g. every `ImageDataset` implementation, regardless of which real dataset it wraps, must
return exactly `{"image", "y", "example_id"}`, which is what makes it safe to pass into label-free
code (D-003).

Pipeline position: shared, pure. Imported by both zones; contains no logic and no I/O.

Status: stub -- populated as each type is first needed, starting with `ImageDataset` in M1.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ImageDataset(Protocol):
    """A dataset that returns exactly `{"image", "y", "example_id"}` per item.

    Every real dataset (synthetic_shapes, planted_pets, waterbirds) is wrapped by
    `build.datasets.RenderedImageDataset`, which implements this protocol. That is the only shape
    label-free code is ever allowed to see -- no group labels, no dataset-specific fields.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, Any]: ...
