"""Unit тестове (unit tests), дефинирани преди реализацията на основния код.

Покриват:
- структурата на енкодерите
- формата на изходите
- контрастната загуба
- прототипната памет
- двуагентния модел
"""

import torch

from config.config import Config
from src.models import ResNet18SSL, DualAgentModel
from src.losses import contrastive_loss, cross_agent_consistency_loss, PrototypeMemory


def test_resnet18ssl_output_shape():
    cfg = Config()
    model = ResNet18SSL(cfg.embedding_dim)
    x = torch.randn(4, 3, 128, 128)
    h, z = model(x)
    assert h.shape[0] == 4
    assert z.shape == (4, cfg.embedding_dim)


def test_contrastive_loss_runs():
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss = contrastive_loss(z1, z2, 0.2)
    assert loss.item() == loss.item()


def test_cross_agent_consistency_loss_runs():
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss = cross_agent_consistency_loss(z1, z2)
    assert loss.item() == loss.item()


def test_prototype_memory_update():
    mem = PrototypeMemory(128, max_prototypes=10, momentum=0.95)
    z = torch.randn(5, 128)
    mem.update_or_create(z, novelty_threshold=0.5)
    assert mem.active_k > 0


def test_dual_agent_output_shape():
    cfg = Config()
    model = DualAgentModel(cfg.embedding_dim)
    xa = torch.randn(4, 3, 128, 128)
    xb = torch.randn(4, 3, 128, 128)
    ha, za, hb, zb = model(xa, xb)
    assert za.shape == (4, cfg.embedding_dim)
    assert zb.shape == (4, cfg.embedding_dim)