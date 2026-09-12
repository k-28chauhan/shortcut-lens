"""Image transforms: train = horizontal flip + normalise; eval = normalise only.

Concept: random crops are deliberately excluded because they could cut a planted patch out of
frame, silently lowering the effective rho (D-009). One augmentation policy for every dataset
keeps method comparisons clean. A horizontal flip is a bijection, so a patch fully inside the
canvas stays fully inside after flipping (see docs/ARCHITECTURE.md §8) -- flipping is safe where
cropping would not be.

Pipeline position: label-free zone. Used by `training/erm.py` and `embeddings/model_space.py`
(both M3); verified here with `torchvision.transforms.v2` since that API is new enough to check
by running code rather than trusting memory (CLAUDE.md §5).
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as T

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform() -> T.Compose:
    """Horizontal flip (p=0.5) + normalise. No random crop -- see module docstring (D-009)."""
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


def eval_transform() -> T.Compose:
    """Normalise only -- no augmentation, so evaluation is deterministic given the same image."""
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )
