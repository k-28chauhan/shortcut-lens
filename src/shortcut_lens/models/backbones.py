"""resnet18, resnet50 (ImageNet-pretrained) and a tiny CNN; penultimate features.

Concept: `features()` returns the penultimate-layer activations (after global average pooling) --
the representation last-layer retraining (DFR/AFR) operates on. Pretrained weights are
`IMAGENET1K_V1` for comparability with the Waterbirds/DFR literature (D-023; enum verified by
running code -- `ResNet18_Weights` has only `IMAGENET1K_V1`, `ResNet50_Weights` also has
`IMAGENET1K_V2`, not used here).

Pipeline position: label-free zone. Used by `training/erm.py` and `embeddings/model_space.py`.
"""

from __future__ import annotations

from typing import Literal, cast

import torch
from torch import nn
from torchvision.models import resnet18, resnet50

Arch = Literal["resnet18", "resnet50", "tiny_cnn"]


class Backbone(nn.Module):
    """A trunk (`features()`) plus a linear head (`forward()` -> logits) for `num_classes`.

    `features()` always returns the flattened penultimate activation, regardless of `arch` -- that
    fixed shape is what lets last-layer retraining (M7) treat every backbone the same way.
    """

    def __init__(self, trunk: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.feature_dim = feature_dim
        self.head = nn.Linear(feature_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.trunk(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.features(x)))


def _tiny_cnn_trunk() -> tuple[nn.Module, int]:
    """A small conv net for `smoke`/CI: three conv blocks + global average pool. Feature dim 32."""
    trunk = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
    )
    return trunk, 32


def build_backbone(
    arch: Arch, num_classes: int = 2, pretrained: str | None = "IMAGENET1K_V1"
) -> Backbone:
    """Build a `Backbone` for `arch`. `pretrained` names a torchvision weights enum member.

    `resnet18`/`resnet50` drop the original `fc` layer (their features feed `Backbone.head`
    instead); `tiny_cnn` ignores `pretrained` (there are no pretrained weights for it).
    """
    if arch == "resnet18":
        model = resnet18(weights=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return Backbone(model, feature_dim, num_classes)
    if arch == "resnet50":
        model = resnet50(weights=pretrained)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return Backbone(model, feature_dim, num_classes)
    trunk, feature_dim = _tiny_cnn_trunk()
    return Backbone(trunk, feature_dim, num_classes)
