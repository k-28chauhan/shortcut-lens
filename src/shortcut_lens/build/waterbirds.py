"""Waterbirds (Sagawa et al. 2020): landbird/waterbird x land/water background, natural benchmark.

Concept: 95% of waterbirds appear on water and 95% of landbirds on land in training data, so a
model can learn "background predicts label" instead of the bird itself. Unlike PlantedPets, we
did not create this shortcut -- it lets us check whether the pipeline's findings on a shortcut
we control generalise to one we did not (see PRD §7 for expected group counts).

Pipeline position: oracle zone. Parses the official `metadata.csv`; builds both `balanced` and
`realistic` validation modes (D-006).

Status: stub -- implemented in M1 (see docs/PLAN.md).
"""
