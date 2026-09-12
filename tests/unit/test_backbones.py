"""Tests for shortcut_lens.models.backbones: shapes, and that features() is head-independent."""

from __future__ import annotations

import pytest
import torch

from shortcut_lens.models.backbones import build_backbone

_BATCH = 4


@pytest.mark.parametrize(
    ("arch", "expected_feature_dim"),
    [("resnet18", 512), ("resnet50", 2048), ("tiny_cnn", 32)],
)
def test_backbone_shapes(arch: str, expected_feature_dim: int) -> None:
    backbone = build_backbone(arch, num_classes=2, pretrained=None)  # type: ignore[arg-type]
    x = torch.randn(_BATCH, 3, 224, 224)

    features = backbone.features(x)
    logits = backbone.forward(x)

    assert features.shape == (_BATCH, expected_feature_dim)
    assert logits.shape == (_BATCH, 2)
    assert backbone.feature_dim == expected_feature_dim


def test_features_unaffected_by_head() -> None:
    backbone = build_backbone("tiny_cnn", num_classes=2, pretrained=None)
    x = torch.randn(_BATCH, 3, 224, 224)

    features_before = backbone.features(x)
    # replacing the head must not change what features() returns for the same input
    backbone.head = torch.nn.Linear(backbone.feature_dim, 5)
    features_after = backbone.features(x)

    torch.testing.assert_close(features_before, features_after)


@pytest.mark.network
def test_resnet18_imagenet_pretrained_downloads_and_has_expected_feature_dim() -> None:
    backbone = build_backbone("resnet18", num_classes=2, pretrained="IMAGENET1K_V1")
    assert backbone.feature_dim == 512
