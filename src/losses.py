from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix between two batches of vectors."""
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.t()


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    NT-Xent contrastive loss for a batch of positive pairs (SimCLR-style).

    Args:
        z1: embeddings for first view, shape [B, D]
        z2: embeddings for second view, shape [B, D]
        temperature: scaling factor for logits

    Returns:
        Scalar loss averaged over all positive pairs.
    """
    batch_size = z1.size(0)

    # stack both views: [2B, D]
    z = torch.cat([z1, z2], dim=0)

    # similarity matrix [2B, 2B]
    sim = cosine_similarity(z, z) / temperature

    # mask out self-similarities on the diagonal
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positive index for each sample: its counterpart view
    pos = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])

    loss = F.cross_entropy(sim, pos)
    return loss


def cross_agent_consistency_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage agreement between two agents' embeddings for the same inputs.

    Returns 1 - mean cosine similarity, so lower is better (more consistent).
    """
    return 1.0 - F.cosine_similarity(z_a, z_b).mean()


class PrototypeMemory(nn.Module):
    """
    Simple prototype memory with momentum updates.

    Stores up to `max_prototypes` normalized vectors and updates them
    online using a momentum rule. New prototypes are created when the
    distance to existing ones exceeds `novelty_threshold`.
    """

    def __init__(
        self,
        feat_dim: int,
        max_prototypes: int = 200,
        momentum: float = 0.95,
    ) -> None:
        super().__init__()
        self.max_prototypes = max_prototypes
        self.momentum = momentum

        self.register_buffer("prototypes", torch.zeros(max_prototypes, feat_dim))
        self.register_buffer("counts", torch.zeros(max_prototypes))
        self.active_k = 0

    @torch.no_grad()
    def distances(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Return distances from batch z to currently active prototypes."""
        if self.active_k == 0:
            return None
        return torch.cdist(z, self.prototypes[:self.active_k])

    @torch.no_grad()
    def update_or_create(self, z: torch.Tensor, novelty_threshold: float = 0.9) -> None:
        """
        Update existing prototypes with momentum or create new ones.

        Args:
            z: batch of feature vectors, shape [B, D]
            novelty_threshold: min distance to consider a new prototype
        """
        z = F.normalize(z, dim=-1)

        for vec in z:
            # if memory is empty, initialize first prototype
            if self.active_k == 0:
                self.prototypes[0] = vec
                self.counts[0] = 1
                self.active_k = 1
                continue

            # find nearest existing prototype
            dists = torch.norm(self.prototypes[:self.active_k] - vec, dim=1)
            idx = torch.argmin(dists)

            # create new prototype if far enough and space available
            if dists[idx] > novelty_threshold and self.active_k < self.max_prototypes:
                self.prototypes[self.active_k] = vec
                self.counts[self.active_k] = 1
                self.active_k += 1
            else:
                # momentum update of existing prototype, then renormalize
                self.prototypes[idx] = (
                    self.momentum * self.prototypes[idx] + (1 - self.momentum) * vec
                )
                self.prototypes[idx] = F.normalize(
                    self.prototypes[idx].unsqueeze(0), dim=-1
                ).squeeze(0)
                self.counts[idx] += 1


def prototype_compactness_loss(
    z: torch.Tensor,
    proto_dists: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Loss that encourages samples to stay close to at least one prototype.

    If there are no prototypes yet, returns zero (no regularization).
    """
    if proto_dists is None:
        return torch.tensor(0.0, device=z.device)
    # for each sample, take distance to nearest prototype and average
    return proto_dists.min(dim=1).values.mean()