import torch
from dataclasses import dataclass


@dataclass
class Config:
    data_dir: str = "./data/coil-100"
    ckpt_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    batch_size: int = 64
    num_workers: int = 4
    epochs_supervised: int = 20
    epochs_ssl: int = 30
    epochs_dual: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_dim: int = 128
    temperature: float = 0.2

    max_prototypes: int = 200
    proto_momentum: float = 0.95
    novelty_threshold: float = 0.9

    agreement_weight: float = 0.5
    proto_weight: float = 0.1

    seed: int = 42