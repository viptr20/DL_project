"""Тренировъчни функции (training functions) за трите режима:
- supervised baseline с ResNet-18;
- самообучение (self-supervised) с ResNet-18 + контрастна загуба;
- двуагентен модел с два ResNet-18 енкодера и прототипи.
"""

import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models

from config.config import Config
from src.dataset import Coil100Dataset
from src.models import ResNet18SSL, DualAgentModel
from src.losses import (
    contrastive_loss,
    cross_agent_consistency_loss,
    PrototypeMemory,
    prototype_compactness_loss,
)


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: Config, n_views: int = 2) -> Tuple[DataLoader, DataLoader]:
    dataset = Coil100Dataset(cfg.data_dir, image_size=128, n_views=n_views)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


def train_supervised_baseline(cfg: Config):
    """Обучение на supervised baseline (ResNet-18 + линеен класификатор)."""
    set_seed(cfg.seed)
    train_loader, _ = build_loaders(cfg, n_views=1)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs_supervised):
        model.train()
        for batch in train_loader:
            x = batch["views"][:, 0].to(cfg.device)
            y = batch["label"].to(cfg.device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[supervised] epoch {epoch+1}/{cfg.epochs_supervised} loss={loss.item():.4f}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "resnet18_supervised.pt"))


def train_ssl_single_agent(cfg: Config):
    """Самообучение с ResNet-18 + проекционна глава."""
    set_seed(cfg.seed)
    train_loader, _ = build_loaders(cfg, n_views=2)

    model = ResNet18SSL(cfg.embedding_dim).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)

    for epoch in range(cfg.epochs_ssl):
        model.train()
        for batch in train_loader:
            views = batch["views"].to(cfg.device)
            x1 = views[:, 0]
            x2 = views[:, 1]

            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = contrastive_loss(z1, z2, cfg.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[ssl] epoch {epoch+1}/{cfg.epochs_ssl} loss={loss.item():.4f}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "resnet18_ssl.pt"))


def train_dual_agent(cfg: Config):
    """Обучение на двуагентен модел (dual-agent training)."""
    set_seed(cfg.seed)
    train_loader, _ = build_loaders(cfg, n_views=2)

    model = DualAgentModel(cfg.embedding_dim).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    proto_memory = PrototypeMemory(cfg.embedding_dim, cfg.max_prototypes,
                                   cfg.proto_momentum).to(cfg.device)

    for epoch in range(cfg.epochs_dual):
        model.train()
        for batch in train_loader:
            views = batch["views"].to(cfg.device)  # [B, 2, C, H, W]
            xa = views[:, 0]
            xb = views[:, 1]

            _, za, _, zb = model(xa, xb)

            # контрастна загуба за агентите (contrastive loss)
            loss_ssl_a = contrastive_loss(za, zb, cfg.temperature)
            loss_ssl_b = loss_ssl_a  # симетрично опростяване

            # загуба за съгласуваност между агентите
            loss_cons = cross_agent_consistency_loss(za, zb)

            # прототипна компактност
            with torch.no_grad():
                mean_z = (za + zb) / 2.0
                dists = proto_memory.distances(mean_z)
                proto_memory.update_or_create(mean_z, cfg.novelty_threshold)
            loss_proto = prototype_compactness_loss(mean_z, dists)

            loss = (loss_ssl_a + loss_ssl_b
                    + cfg.agreement_weight * loss_cons
                    + cfg.proto_weight * loss_proto)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f"[dual] epoch {epoch+1}/{cfg.epochs_dual} "
            f"loss={loss.item():.4f} cons={loss_cons.item():.4f} "
            f"proto={loss_proto.item():.4f} prototypes={proto_memory.active_k}"
        )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, "dual_resnet18.pt"))


if __name__ == "__main__":
    cfg = Config()
    print("Тренировъчните функции са дефинирани. Стартирай желаната от main или от отделен скрипт.")