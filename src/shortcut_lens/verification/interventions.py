"""Counterfactual interventions: `patch_remove`, `patch_add`, `null_patch_add`.

Concept: a name is only evidence, not proof, of the real cause -- an intervention edits the
image to remove (or add) the suspected cause and checks whether the model's prediction changes.
`null_patch_add` (a grey square, same size) is the occlusion control (D-008): it isolates the
effect of the patch's colour from the effect of covering part of the image at all.

Pipeline position: oracle zone (FR-V1/V2). Used by `verification/verify.py`.

Status: stub -- implemented in M6 (see docs/PLAN.md).
"""
