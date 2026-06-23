from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Encoder(nn.Module):
    """ResNet-18 encoder that outputs feature vectors instead of logits."""

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled image features h."""
        return self.backbone(x)


class ProjectionHead(nn.Module):
    """2-layer MLP projection head used for contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features h to normalized embeddings z."""
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)


class ResNet18SSL(nn.Module):
    """
    Self-supervised ResNet-18 backbone.

    Returns:
        h: encoder features before the projection head
        z: contrastive embeddings after the projection head
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = ResNet18Encoder()
        self.projector = ProjectionHead(self.encoder.feat_dim, 256, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


class DualAgentModel(nn.Module):
    """
    Two-agent SSL model.

    Each agent is a ResNet18SSL; we process two views (xa, xb)
    and return both feature and embedding pairs.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.agent_a = ResNet18SSL(embedding_dim)
        self.agent_b = ResNet18SSL(embedding_dim)

    def forward(self, xa: torch.Tensor, xb: torch.Tensor):
        ha, za = self.agent_a(xa)
        hb, zb = self.agent_b(xb)
        return ha, za, hb, zb