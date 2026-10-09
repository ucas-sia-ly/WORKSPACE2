from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .manifest import Sample


def build_transform(config: dict):
    data = config.get("data", {})
    h, w = data.get("image_size", [322, 322])
    normalize = data.get("normalize", "imagenet")
    operations = [T.Resize((int(h), int(w)), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor()]
    if normalize == "imagenet":
        operations.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    elif normalize != "none":
        raise ValueError("data.normalize must be 'imagenet' or 'none'")
    return T.Compose(operations)


class ImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform):
        self.samples, self.transform = samples, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        with Image.open(self.samples[index].path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


class TripletDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_root: str | Path, transform):
        import csv
        self.root, self.transform = Path(image_root).expanduser().resolve(), transform
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if not {"anchor", "positive", "negative"}.issubset(reader.fieldnames or []):
                raise ValueError("triplets CSV requires anchor,positive,negative columns")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError("Empty triplets CSV")
        for row in self.rows:
            for col in ("anchor", "positive", "negative"):
                if not (self.root / row[col]).is_file():
                    raise FileNotFoundError(f"Triplet image does not exist: {self.root / row[col]}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        out = []
        for col in ("anchor", "positive", "negative"):
            with Image.open(self.root / self.rows[i][col]) as image:
                out.append(self.transform(image.convert("RGB")))
        return tuple(out)
