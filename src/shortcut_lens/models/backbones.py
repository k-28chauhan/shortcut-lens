"""resnet18, resnet50 (ImageNet-pretrained) and a tiny CNN; penultimate features.

Concept: `features()` returns the penultimate-layer activations (after global average pooling) --
the representation last-layer retraining (DFR/AFR) operates on. Pretrained weights are
`IMAGENET1K_V1` for comparability with the Waterbirds/DFR literature (D-023).

Pipeline position: label-free zone. Used by `training/erm.py` and `embeddings/model_space.py`.

Status: stub -- implemented in M3 (see docs/PLAN.md).
"""
