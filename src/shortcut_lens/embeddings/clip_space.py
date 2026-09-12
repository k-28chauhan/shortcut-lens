"""CLIP-space embeddings: open_clip ViT-B/32 image embeddings, L2-normalised.

Concept: the second embedding space -- a general-purpose vision-language representation not
trained on this task at all, which may see a planted patch (or Waterbirds' background) more or
less clearly than the model's own features (RQ2). Model/pretrained tags verified by running code
(CLAUDE.md §5): `open_clip.list_pretrained()` confirms `("ViT-B-32", "laion2b_s34b_b79k")` is a
real, available tag pair (ARCHITECTURE §6's example config). Its own preprocessing
(`Resize(224, bicubic) -> CenterCrop(224,224) -> Normalize(CLIP mean/std)`) is a geometric no-op on
this project's images, since every dataset is already exactly 224x224 at build time -- only the
CLIP-specific normalisation actually changes pixel values, which is why the dataset handed to
`embed_clip_space` must use CLIP's own `preprocess` as its transform, not
`data.transforms`'s ImageNet one (see `embeddings/model_space.py` for the model-space equivalent).

Pipeline position: label-free zone. Entry point for `slens embed --space clip`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader, Dataset

from shortcut_lens.config import StrictBaseModel
from shortcut_lens.types import ImageDataset


class ClipEmbedConfig(StrictBaseModel):
    """`embed.clip` section of an experiment config (ARCHITECTURE §6)."""

    model: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"


def load_clip(config: ClipEmbedConfig, device: str) -> tuple[Any, Callable[..., Any]]:
    """Load the CLIP model and its own preprocessing transform, in eval mode on `device`.

    Returned as `Any` (not `nn.Module`): open_clip ships no type stubs, and its model class adds
    `encode_image()`, which a plain `nn.Module` annotation would hide from mypy.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.model, pretrained=config.pretrained
    )
    model = model.to(device).eval()
    return model, preprocess


def embed_clip_space(
    model: Any, dataset: ImageDataset, device: str, batch_size: int
) -> tuple[np.ndarray, list[str]]:
    """Run `model.encode_image()` over `dataset` (already transformed by CLIP's own `preprocess`).

    Returns `(vectors [N, D], example_ids)`, L2-normalised, in the order `dataset` is iterated.
    """
    loader = DataLoader(cast(Dataset[Any], dataset), batch_size=batch_size, shuffle=False)

    example_ids: list[str] = []
    feature_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = model.encode_image(batch["image"].to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            feature_batches.append(features.cpu())
            example_ids.extend(batch["example_id"])

    vectors = torch.cat(feature_batches, dim=0).numpy()
    return vectors, example_ids
