"""CLIP-space embeddings: open_clip ViT-B/32 image embeddings, L2-normalised.

Concept: the second embedding space -- a general-purpose vision-language representation not
trained on this task at all, which may see a planted patch (or Waterbirds' background) more or
less clearly than the model's own features (RQ2). Exact open_clip model/pretrained tags are
verified by running code at M3, not assumed from memory (CLAUDE.md §5).

Pipeline position: label-free zone. Entry point for `slens embed --space clip`.

Status: stub -- implemented in M3 (see docs/PLAN.md).
"""
