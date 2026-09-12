"""Model-space embeddings: the trained classifier's own penultimate features.

Concept: one of the two embedding spaces slice discovery can operate in -- the model's own
learned representation, as opposed to CLIP's (see RQ2 in docs/PRD.md).

Pipeline position: label-free zone. Entry point for `slens embed --space model`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from shortcut_lens.models.backbones import Backbone
from shortcut_lens.types import ImageDataset


def embed_model_space(
    backbone: Backbone, dataset: ImageDataset, device: str, batch_size: int
) -> tuple[np.ndarray, list[str]]:
    """Run `backbone.features()` over every example in `dataset` (already eval-transformed).

    Returns `(vectors [N, D], example_ids)`, in the order `dataset` is iterated -- callers write
    both together via `embeddings/store.py` so they can never drift apart.
    """
    backbone.eval()
    loader = DataLoader(cast(Dataset[Any], dataset), batch_size=batch_size, shuffle=False)

    example_ids: list[str] = []
    feature_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = backbone.features(batch["image"].to(device)).cpu()
            feature_batches.append(features)
            example_ids.extend(batch["example_id"])

    vectors = torch.cat(feature_batches, dim=0).numpy()
    return vectors, example_ids
