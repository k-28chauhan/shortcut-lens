"""Shortcut patch planting: `PatchSpec`, `add_patch`, `null_patch`, deterministic positions.

Concept: a "planted shortcut" is a synthetic spurious feature (a coloured square patch) painted
onto an image with controlled correlation to the label (rho) and controlled visibility (patch
size). Because we choose rho and the patch ourselves, we know the ground-truth answer a
discovery tool should find (see docs/ABSTRACT.md). Patch position is deterministic per image,
derived from a hash of (example_id, build_seed). `null_patch` paints a grey square of the same
size, used as an occlusion control (D-008). Painting happens at load time on decoded images, so
JPEG compression never touches the patch; `remove_patch` re-renders from the clean base rather
than editing pixels, so `remove(add(x)) == x` exactly.

Pipeline position: oracle zone. Used by `build/pets.py`, `build/waterbirds.py`,
`build/synthetic_shapes.py` and `build/datasets.py`.

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
