"""Функции на загуба (loss functions) и прототипна памет (prototype memory)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """Контрастна функция на загуба (contrastive loss) от тип InfoNCE."""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = cosine_similarity(z, z) / temperature

    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    pos = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    loss = F.cross_entropy(sim, pos)
    return loss


def cross_agent_consistency_loss(z_a: torch.Tensor, z_b: torch.Tensor):
    """Загуба за съгласуваност между агентите (cross-agent consistency loss)."""
    return 1.0 - F.cosine_similarity(z_a, z_b).mean()


class PrototypeMemory(nn.Module):
    """Прототипна памет (prototype memory) за вътрешни категории.
    Съхранява до max_prototypes прототипа и ги обновява с momentum.
    """

    def __init__(self, feat_dim: int, max_prototypes: int = 200, momentum: float = 0.95):
        super().__init__()
        self.max_prototypes = max_prototypes
        self.momentum = momentum
        self.register_buffer("prototypes", torch.zeros(max_prototypes, feat_dim))
        self.register_buffer("counts", torch.zeros(max_prototypes))
        self.active_k = 0

    @torch.no_grad()
    def distances(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        if self.active_k == 0:
            return None
        return torch.cdist(z, self.prototypes[:self.active_k])

    @torch.no_grad()
    def update_or_create(self, z: torch.Tensor, novelty_threshold: float = 0.9):
        z = F.normalize(z, dim=-1)
        for vec in z:
            if self.active_k == 0:
                self.prototypes[0] = vec
                self.counts[0] = 1
                self.active_k = 1
                continue
            dists = torch.norm(self.prototypes[:self.active_k] - vec, dim=1)
            idx = torch.argmin(dists)
            if dists[idx] > novelty_threshold and self.active_k < self.max_prototypes:
                self.prototypes[self.active_k] = vec
                self.counts[self.active_k] = 1
                self.active_k += 1
            else:
                self.prototypes[idx] = (
                    self.momentum * self.prototypes[idx] + (1 - self.momentum) * vec
                )
                self.prototypes[idx] = F.normalize(
                    self.prototypes[idx].unsqueeze(0), dim=-1
                ).squeeze(0)
                self.counts[idx] += 1


def prototype_compactness_loss(z: torch.Tensor, proto_dists: Optional[torch.Tensor]):
    """Загуба за компактност спрямо най-близкия прототип."""
    if proto_dists is None:
        return torch.tensor(0.0, device=z.device)
    return proto_dists.min(dim=1).values.mean()