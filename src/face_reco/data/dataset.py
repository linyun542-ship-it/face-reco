from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def build_transforms(image_size: int = 160, train: bool = True) -> transforms.Compose:
    ops = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(ops)


class CelebAIdentityDataset(Dataset):
    def __init__(self, csv_path: str | Path, image_size: int = 160, train: bool = True):
        self.rows: list[dict[str, str]] = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"No rows in {csv_path}")
        self.transform = build_transforms(image_size, train=train)
        self.identities = sorted({int(r["identity"]) for r in self.rows})
        self.num_classes = max(self.identities) + 1

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = self.transform(img)
        y = int(row["identity"])
        return x, y


class TripletBatchSampler:
    """Yields flat index lists: P identities × K images each."""

    def __init__(
        self,
        dataset: CelebAIdentityDataset,
        identities_per_batch: int = 16,
        images_per_identity: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ):
        self.by_id: dict[int, list[int]] = defaultdict(list)
        for i, row in enumerate(dataset.rows):
            self.by_id[int(row["identity"])].append(i)
        # Need at least images_per_identity samples.
        self.by_id = {
            k: v for k, v in self.by_id.items() if len(v) >= images_per_identity
        }
        self.ids = list(self.by_id.keys())
        if len(self.ids) < identities_per_batch:
            raise ValueError(
                f"Need ≥{identities_per_batch} identities with enough images; "
                f"have {len(self.ids)}"
            )
        self.p = identities_per_batch
        self.k = images_per_identity
        self.rng = random.Random(seed)
        if batches_per_epoch is None:
            batches_per_epoch = max(1, len(dataset) // (self.p * self.k))
        self.batches_per_epoch = batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen = self.rng.sample(self.ids, self.p)
            batch: list[int] = []
            for ident in chosen:
                batch.extend(self.rng.sample(self.by_id[ident], self.k))
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


def make_loader(
    csv_path: str | Path,
    batch_size: int,
    image_size: int = 160,
    train: bool = True,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    ds = CelebAIdentityDataset(csv_path, image_size=image_size, train=train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle and train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def make_triplet_loader(
    csv_path: str | Path,
    image_size: int = 160,
    identities_per_batch: int = 16,
    images_per_identity: int = 4,
    num_workers: int = 2,
    seed: int = 42,
) -> DataLoader:
    ds = CelebAIdentityDataset(csv_path, image_size=image_size, train=True)
    sampler = TripletBatchSampler(
        ds,
        identities_per_batch=identities_per_batch,
        images_per_identity=images_per_identity,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
