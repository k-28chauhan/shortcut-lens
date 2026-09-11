"""`RenderedImageDataset`: the only `ImageDataset` implementation, images rendered on the fly.

Concept: images (including any planted patch) are rendered at `__getitem__` time from the cached
base image plus a patch spec, not baked into files on disk -- this keeps a full ablation sweep
over patch size/rho cheap. Every instance returns exactly `{"image", "y", "example_id"}`,
regardless of dataset, which is what makes it safe to hand to label-free code (D-003).

Pipeline position: the seam between the oracle zone and the label-free zone. Built in `build/`,
injected into label-free stages by `cli.py`.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
