import re
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class Coil100Dataset(Dataset):

    def __init__(self, data_dir: str, image_size: int = 128, n_views: int = 2):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.n_views = n_views
        self.samples: List[Tuple[Path, int]] = []

        pattern = re.compile(r"obj(\d+)__?(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
        for p in sorted(self.data_dir.glob("*")):
            if not p.is_file():
                continue
            m = pattern.match(p.name)
            if not m:
                continue
            obj_id = int(m.group(1)) - 1 
            self.samples.append((p, obj_id))

        if not self.samples:
            raise RuntimeError(
                f"Не са намерени COIL-100 файлове в {self.data_dir}. "
                f"Очаква се формат objX__Y.png"
            )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        views = [self.transform(img) for _ in range(self.n_views)]
        return {
            "views": torch.stack(views, dim=0),
            "label": label,
            "path": str(path),
        }