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
    # Arrange
    cfg = Config()
    model = ResNet18SSL(cfg.embedding_dim)
    x = torch.randn(4, 3, 128, 128)

    # Act
    h, z = model(x)

    # Assert
    assert h.shape == (4, model.encoder.feat_dim)
    assert z.shape == (4, cfg.embedding_dim)


def test_contrastive_loss_runs_and_is_positive() -> None:
    """Contrastive loss should run without error and return a finite positive scalar."""
    # Arrange
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    temperature = 0.2

    # Act
    loss = contrastive_loss(z1, z2, temperature=temperature)

    # Assert
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_cross_agent_consistency_loss_runs_and_is_bounded() -> None:
    """Consistency loss should run and be within a reasonable range."""
    # Arrange
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)

    # Act
    loss = cross_agent_consistency_loss(z1, z2)

    # Assert
    assert torch.isfinite(loss)
    assert 0.0 <= loss.item() <= 2.0


def test_prototype_memory_update_creates_prototypes() -> None:
    """PrototypeMemory should allocate at least one prototype after an update."""
    # Arrange
    feat_dim = 128
    mem = PrototypeMemory(feat_dim, max_prototypes=10, momentum=0.95)
    z = torch.randn(5, feat_dim)
    novelty_threshold = 0.5

    # Act
    mem.update_or_create(z, novelty_threshold=novelty_threshold)

    # Assert
    assert mem.active_k > 0
    assert mem.prototypes[:mem.active_k].shape[0] == mem.active_k


def test_dual_agent_output_shape() -> None:
    """DualAgentModel should return embeddings with expected shapes for both agents."""
    # Arrange
    cfg = Config()
    model = DualAgentModel(cfg.embedding_dim)
    xa = torch.randn(4, 3, 128, 128)
    xb = torch.randn(4, 3, 128, 128)

    # Act
    ha, za, hb, zb = model(xa, xb)

    # Assert
    assert ha.shape[0] == 4
    assert hb.shape[0] == 4
    assert za.shape == (4, cfg.embedding_dim)
    assert zb.shape == (4, cfg.embedding_dim)