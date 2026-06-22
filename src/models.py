"""Модели (models) за проекта

- ResNet-18 базов енкодер (baseline encoder)
- ResNet-18 с проекционна глава (projection head) за контрастно самообучение
- двуагентен модел (dual-agent model) с два ResNet-18 енкодера
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Encoder(nn.Module):
    """ResNet-18 енкодер без последния линеен слой"""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ProjectionHead(nn.Module):
    """Проекционна глава за контрастно самообучение"""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


class ResNet18SSL(nn.Module):
    """ResNet-18 с проекционна глава за SSL"""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projector = ProjectionHead(self.encoder.feat_dim, 256, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


class DualAgentModel(nn.Module):
    """Двуагентен модел (dual-agent model) с два ResNet-18 енкодера"""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.agent_a = ResNet18SSL(embedding_dim)
        self.agent_b = ResNet18SSL(embedding_dim)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        ha, za = self.agent_a(xa)
        hb, zb = self.agent_b(xb)
        return ha, za, hb, zb