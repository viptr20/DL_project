from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

from config.config import Config
from src.train import (
    train_supervised_baseline,
    train_ssl_single_agent,
    train_dual_agent,
)
from src.eval import (
    evaluate_supervised_baseline,
    evaluate_ssl_model,
    evaluate_dual_agent_model,
    log_single_result,
    ensure_csv_exists,
)

REPORT_PATH = "./reports/experiment_report.csv"
CHECKPOINT_DIR = "./checkpoints"


def make_cfg(base_cfg: Config, **kwargs) -> Config:
    cfg = copy.deepcopy(base_cfg)
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def copy_checkpoint(src: str, dst: str):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    shutil.copy2(src, dst)


def run_supervised_experiment(cfg: Config, comment: str, augmentation: str = "default"):
    train_supervised_baseline(cfg)

    src_ckpt = os.path.join(cfg.ckpt_dir, "resnet18_supervised.pt")
    dst_ckpt = os.path.join(
        cfg.ckpt_dir,
        f"resnet18_supervised_e{cfg.epochs_supervised}_bs{cfg.batch_size}_lr{cfg.lr}_emb{cfg.embedding_dim}.pt",
    )

    copy_checkpoint(src_ckpt, dst_ckpt)
    metrics = evaluate_supervised_baseline(cfg, dst_ckpt)

    log_single_result(
        csv_path=REPORT_PATH,
        model_name="ResNet-18",
        setup="supervised",
        epochs=cfg.epochs_supervised,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        embedding_dim=cfg.embedding_dim,
        augmentation=augmentation,
        metrics=metrics,
        comment=comment,
    )


def run_ssl_experiment(cfg: Config, comment: str, augmentation: str = "default"):
    train_ssl_single_agent(cfg)

    src_ckpt = os.path.join(cfg.ckpt_dir, "resnet18_ssl.pt")
    dst_ckpt = os.path.join(
        cfg.ckpt_dir,
        f"resnet18_ssl_e{cfg.epochs_ssl}_bs{cfg.batch_size}_lr{cfg.lr}_emb{cfg.embedding_dim}.pt",
    )

    copy_checkpoint(src_ckpt, dst_ckpt)
    metrics = evaluate_ssl_model(cfg, dst_ckpt)

    log_single_result(
        csv_path=REPORT_PATH,
        model_name="ResNet-18 + projection head",
        setup="self-supervised",
        epochs=cfg.epochs_ssl,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        embedding_dim=cfg.embedding_dim,
        augmentation=augmentation,
        metrics=metrics,
        comment=comment,
    )


def run_dual_experiment(cfg: Config, comment: str, augmentation: str = "default"):
    train_dual_agent(cfg)

    src_ckpt = os.path.join(cfg.ckpt_dir, "dual_resnet18.pt")
    dst_ckpt = os.path.join(
        cfg.ckpt_dir,
        f"dual_resnet18_e{cfg.epochs_dual}_bs{cfg.batch_size}_lr{cfg.lr}_emb{cfg.embedding_dim}.pt",
    )

    copy_checkpoint(src_ckpt, dst_ckpt)
    metrics = evaluate_dual_agent_model(cfg, dst_ckpt, agent="mean")

    log_single_result(
        csv_path=REPORT_PATH,
        model_name="Dual-ResNet-18",
        setup="dual-agent",
        epochs=cfg.epochs_dual,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        embedding_dim=cfg.embedding_dim,
        augmentation=augmentation,
        metrics=metrics,
        comment=comment,
    )


def run_20_experiments():
    base_cfg = Config()
    ensure_csv_exists(REPORT_PATH)

    experiments = [
        # 4 supervised
        ("supervised", dict(epochs_supervised=10, batch_size=32, lr=1e-3, embedding_dim=128), "supervised e10 bs32 lr1e-3"),
        ("supervised", dict(epochs_supervised=20, batch_size=32, lr=1e-3, embedding_dim=128), "supervised e20 bs32 lr1e-3"),
        ("supervised", dict(epochs_supervised=10, batch_size=64, lr=3e-4, embedding_dim=128), "supervised e10 bs64 lr3e-4"),
        ("supervised", dict(epochs_supervised=20, batch_size=64, lr=3e-4, embedding_dim=128), "supervised e20 bs64 lr3e-4"),

        # 8 ssl
        ("ssl", dict(epochs_ssl=10, batch_size=32, lr=3e-4, embedding_dim=128), "ssl e10 bs32 lr3e-4 emb128"),
        ("ssl", dict(epochs_ssl=20, batch_size=32, lr=3e-4, embedding_dim=128), "ssl e20 bs32 lr3e-4 emb128"),
        ("ssl", dict(epochs_ssl=10, batch_size=64, lr=3e-4, embedding_dim=128), "ssl e10 bs64 lr3e-4 emb128"),
        ("ssl", dict(epochs_ssl=20, batch_size=64, lr=3e-4, embedding_dim=128), "ssl e20 bs64 lr3e-4 emb128"),
        ("ssl", dict(epochs_ssl=10, batch_size=32, lr=1e-4, embedding_dim=256), "ssl e10 bs32 lr1e-4 emb256"),
        ("ssl", dict(epochs_ssl=20, batch_size=32, lr=1e-4, embedding_dim=256), "ssl e20 bs32 lr1e-4 emb256"),
        ("ssl", dict(epochs_ssl=10, batch_size=64, lr=1e-4, embedding_dim=256), "ssl e10 bs64 lr1e-4 emb256"),
        ("ssl", dict(epochs_ssl=20, batch_size=64, lr=1e-4, embedding_dim=256), "ssl e20 bs64 lr1e-4 emb256"),

        # 8 dual-agent
        ("dual", dict(epochs_dual=10, batch_size=32, lr=3e-4, embedding_dim=128), "dual e10 bs32 lr3e-4 emb128"),
        ("dual", dict(epochs_dual=20, batch_size=32, lr=3e-4, embedding_dim=128), "dual e20 bs32 lr3e-4 emb128"),
        ("dual", dict(epochs_dual=10, batch_size=64, lr=3e-4, embedding_dim=128), "dual e10 bs64 lr3e-4 emb128"),
        ("dual", dict(epochs_dual=20, batch_size=64, lr=3e-4, embedding_dim=128), "dual e20 bs64 lr3e-4 emb128"),
        ("dual", dict(epochs_dual=10, batch_size=32, lr=1e-4, embedding_dim=256), "dual e10 bs32 lr1e-4 emb256"),
        ("dual", dict(epochs_dual=20, batch_size=32, lr=1e-4, embedding_dim=256), "dual e20 bs32 lr1e-4 emb256"),
        ("dual", dict(epochs_dual=10, batch_size=64, lr=1e-4, embedding_dim=256), "dual e10 bs64 lr1e-4 emb256"),
        ("dual", dict(epochs_dual=20, batch_size=64, lr=1e-4, embedding_dim=256), "dual e20 bs64 lr1e-4 emb256"),
    ]

    for idx, (kind, params, comment) in enumerate(experiments, start=1):
        print(f"\\n===== Running experiment {idx}/20: {comment} =====")
        cfg = make_cfg(base_cfg, **params)

        if kind == "supervised":
            run_supervised_experiment(cfg, comment)
        elif kind == "ssl":
            run_ssl_experiment(cfg, comment)
        elif kind == "dual":
            run_dual_experiment(cfg, comment)
        else:
            raise ValueError(f"Unknown experiment type: {kind}")


if __name__ == "__main__":
    run_20_experiments()