"""`caption_keywords` namer (B2T-style): BLIP captions, YAKE keywords, CLIP-scored, with support
counts.

Concept: an alternative to a fixed vocabulary -- caption the slice's images with BLIP, extract
candidate keywords with YAKE, then rank them by how much more they show up in misclassified vs
correctly classified images. The support count N (how many images actually mention the keyword)
protects against a keyword that "sounds right" but only came from one caption.

Pipeline position: label-free zone (FR-N3). Entry point for `slens name --namer caption_keywords`.

Status: stub -- implemented in M5 (see docs/PLAN.md).
"""
