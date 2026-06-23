
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score

from config.config import Config
from src.dataset import Coil100Dataset
from src.models import ResNet18SSL, DualAgentModel


CSV_COLUMNS = [
    "experiment_id",
    "date",
    "model",
    "setup",
    "epochs",
    "batch_size",
    "learning_rate",
    "embedding_dim",
    "augmentation",
    "linear_probe_accuracy",
    "ari",
    "comment",
]


def build_eval_loaders(cfg: Config):
    dataset = Coil100Dataset(cfg.data_dir, image_size=128, n_views=1)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


@torch.no_grad()
def extract_supervised_features(
    cfg: Config, ckpt_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract penultimate-layer features from a supervised ResNet-18 checkpoint.

    Returns:
        xtr, ytr, xva, yva: train/val features and labels.
    """
    model = models.resnet18(weights=None)
    feat_dim = model.fc.in_features
    model.fc = nn.Linear(feat_dim, 100)

    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(cfg.device)
    train_loader, val_loader = build_eval_loaders(cfg)

    xtr, ytr, xva, yva = [], [], [], []

    for batch in train_loader:
        x = batch["views"][:, 0].to(cfg.device)
        y = batch["label"]
        h = feature_extractor(x).flatten(1)
        xtr.append(h.cpu())
        ytr.append(y)

    for batch in val_loader:
        x = batch["views"][:, 0].to(cfg.device)
        y = batch["label"]
        h = feature_extractor(x).flatten(1)
        xva.append(h.cpu())
        yva.append(y)

    return torch.cat(xtr), torch.cat(ytr), torch.cat(xva), torch.cat(yva)


@torch.no_grad()
def extract_ssl_features(
    cfg: Config, ckpt_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract encoder features from a self-supervised ResNet18SSL checkpoint.
    """
    model = ResNet18SSL(cfg.embedding_dim)

    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    train_loader, val_loader = build_eval_loaders(cfg)
    xtr, ytr, xva, yva = [], [], [], []

    for batch in train_loader:
        x = batch["views"][:, 0].to(cfg.device)
        y = batch["label"]
        h, _ = model(x)
        xtr.append(h.cpu())
        ytr.append(y)

    for batch in val_loader:
        x = batch["views"][:, 0].to(cfg.device)
        y = batch["label"]
        h, _ = model(x)
        xva.append(h.cpu())
        yva.append(y)

    return torch.cat(xtr), torch.cat(ytr), torch.cat(xva), torch.cat(yva)


@torch.no_grad()
def extract_dual_features(
    cfg: Config, ckpt_path: str, agent: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features from a DualAgentModel checkpoint.

    Args:
        agent: "a", "b", or "mean" to choose which agent's features to use.

    Returns:
        xtr, ytr, xva, yva: train/val features and labels.
    """
    model = DualAgentModel(cfg.embedding_dim)

    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.to(cfg.device)
    model.eval()

    train_loader, val_loader = build_eval_loaders(cfg)
    xtr, ytr, xva, yva = [], [], [], []

    for batch in train_loader:
        x = batch["views"][:, 0].to(cfg.device)
        y = batch["label"]
        ha, _, hb, _ = model(x, x)

        if agent == "a":
            h = ha
        elif agent == "b":
            h = hb
        else:
            h = (ha + hb) / 2.0

        xtr.append(h.cpu())
        ytr.append(y)

    for batch in val_loader:
        x = batch["views"][:, 0].to(cfg.device)
        y = batch["label"]
        ha, _, hb, _ = model(x, x)

        if agent == "a":
            h = ha
        elif agent == "b":
            h = hb
        else:
            h = (ha + hb) / 2.0

        xva.append(h.cpu())
        yva.append(y)

    return torch.cat(xtr), torch.cat(ytr), torch.cat(xva), torch.cat(yva)


def compute_linear_probe_accuracy(x_train, y_train, x_val, y_val) -> float:
    """
    Train a frozen linear probe on top of features and report accuracy.

    Standard SSL evaluation: train a logistic regression on train features,
    then evaluate on validation labels.
    """
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train.numpy(), y_train.numpy())
    preds = clf.predict(x_val.numpy())
    return float(accuracy_score(y_val.numpy(), preds))


def compute_ari(x_val, y_val, n_clusters: int = 100) -> float:
    """
    Compute Adjusted Rand Index between KMeans clusters and true labels.

    Measures clustering quality of the learned feature space.
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    pred_clusters = km.fit_predict(x_val.numpy())
    return float(adjusted_rand_score(y_val.numpy(), pred_clusters))


def evaluate_supervised_baseline(cfg: Config, ckpt_path: str) -> Dict[str, float]:
    xtr, ytr, xva, yva = extract_supervised_features(cfg, ckpt_path)
    return {
        "linear_probe_accuracy": compute_linear_probe_accuracy(xtr, ytr, xva, yva),
        "ari": compute_ari(xva, yva, n_clusters=100),
    }


def evaluate_ssl_model(cfg: Config, ckpt_path: str) -> Dict[str, float]:
    xtr, ytr, xva, yva = extract_ssl_features(cfg, ckpt_path)
    return {
        "linear_probe_accuracy": compute_linear_probe_accuracy(xtr, ytr, xva, yva),
        "ari": compute_ari(xva, yva, n_clusters=100),
    }


def evaluate_dual_agent_model(
    cfg: Config, ckpt_path: str, agent: str = "mean"
) -> Dict[str, float]:
    xtr, ytr, xva, yva = extract_dual_features(cfg, ckpt_path, agent=agent)
    return {
        "linear_probe_accuracy": compute_linear_probe_accuracy(xtr, ytr, xva, yva),
        "ari": compute_ari(xva, yva, n_clusters=100),
    }


def ensure_csv_exists(csv_path: str):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def get_next_experiment_id(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 1
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return 1
        return max(int(r["experiment_id"]) for r in rows) + 1


def append_row(csv_path: str, row: Dict):
    ensure_csv_exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def log_single_result(
    csv_path: str,
    model_name: str,
    setup: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    embedding_dim: int,
    augmentation: str,
    metrics: Dict[str, float],
    comment: str,
):
    row = {
        "experiment_id": get_next_experiment_id(csv_path),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "setup": setup,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "embedding_dim": embedding_dim,
        "augmentation": augmentation,
        "linear_probe_accuracy": round(metrics["linear_probe_accuracy"], 6),
        "ari": round(metrics["ari"], 6),
        "comment": comment,
    }
    append_row(csv_path, row)
    print(f"Записан експеримент {row['experiment_id']} за {model_name}")


def run_and_log_all_models(
    cfg: Config,
    csv_path: str = "./reports/experiment_report.csv",
    supervised_ckpt: str = "./checkpoints/resnet18_supervised.pt",
    ssl_ckpt: str = "./checkpoints/resnet18_ssl.pt",
    dual_ckpt: str = "./checkpoints/dual_resnet18.pt",
    augmentation: str = "default",
):
    ensure_csv_exists(csv_path)

    if os.path.exists(supervised_ckpt):
        metrics = evaluate_supervised_baseline(cfg, supervised_ckpt)
        log_single_result(
            csv_path=csv_path,
            model_name="ResNet-18",
            setup="supervised",
            epochs=cfg.epochs_supervised,
            batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            embedding_dim=cfg.embedding_dim,
            augmentation=augmentation,
            metrics=metrics,
            comment="Автоматично записан supervised baseline",
        )
    else:
        print(f"Липсва checkpoint: {supervised_ckpt}")

    if os.path.exists(ssl_ckpt):
        metrics = evaluate_ssl_model(cfg, ssl_ckpt)
        log_single_result(
            csv_path=csv_path,
            model_name="ResNet-18 + projection head",
            setup="self-supervised",
            epochs=cfg.epochs_ssl,
            batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            embedding_dim=cfg.embedding_dim,
            augmentation=augmentation,
            metrics=metrics,
            comment="Автоматично записан SSL модел",
        )
    else:
        print(f"Липсва checkpoint: {ssl_ckpt}")

    if os.path.exists(dual_ckpt):
        metrics = evaluate_dual_agent_model(cfg, dual_ckpt, agent="mean")
        log_single_result(
            csv_path=csv_path,
            model_name="Dual-ResNet-18",
            setup="dual-agent",
            epochs=cfg.epochs_dual,
            batch_size=cfg.batch_size,
            learning_rate=cfg.lr,
            embedding_dim=cfg.embedding_dim,
            augmentation=augmentation,
            metrics=metrics,
            comment="Автоматично записан двуагентен модел, средно от двата агента",
        )
    else:
        print(f"Липсва checkpoint: {dual_ckpt}")


if __name__ == "__main__":
    cfg = Config()
    run_and_log_all_models(cfg)