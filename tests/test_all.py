"""Unit tests covering core SSL components:

- encoder structure and output shapes
- contrastive and consistency losses
- prototype memory behavior
- dual-agent model outputs
"""

import torch

from config.config import Config
from src.models import ResNet18SSL, DualAgentModel
from src.losses import (
    contrastive_loss,
    cross_agent_consistency_loss,
    PrototypeMemory,
)


def test_resnet18ssl_output_shape() -> None:
    """ResNet18SSL should return feature and embedding tensors with expected shapes."""
    cfg = Config()
    model = ResNet18SSL(cfg.embedding_dim)

    x = torch.randn(4, 3, 128, 128)
    h, z = model(x)

    assert h.shape == (4, model.encoder.feat_dim)
    assert z.shape == (4, cfg.embedding_dim)


def test_contrastive_loss_runs_and_is_positive() -> None:
    """Contrastive loss should run without error and return a finite positive scalar."""
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)

    loss = contrastive_loss(z1, z2, temperature=0.2)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_cross_agent_consistency_loss_runs_and_is_bounded() -> None:
    """Consistency loss should run and be within a reasonable range."""
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)

    loss = cross_agent_consistency_loss(z1, z2)

    assert torch.isfinite(loss)
    # 1 - cosine similarity averaged; typical values roughly in [0, 2]
    assert 0.0 <= loss.item() <= 2.0


def test_prototype_memory_update_creates_prototypes() -> None:
    """PrototypeMemory should allocate at least one prototype after an update."""
    mem = PrototypeMemory(128, max_prototypes=10, momentum=0.95)

    z = torch.randn(5, 128)
    mem.update_or_create(z, novelty_threshold=0.5)

    assert mem.active_k > 0
    assert mem.prototypes[: mem.active_k].shape[0] == mem.active_k


def test_dual_agent_output_shape() -> None:
    """DualAgentModel should return embeddings with expected shapes for both agents."""
    cfg = Config()
    model = DualAgentModel(cfg.embedding_dim)

    xa = torch.randn(4, 3, 128, 128)
    xb = torch.randn(4, 3, 128, 128)

    ha, za, hb, zb = model(xa, xb)

    assert ha.shape[0] == 4
    assert hb.shape[0] == 4
    assert za.shape == (4, cfg.embedding_dim)
    assert zb.shape == (4, cfg.embedding_dim)