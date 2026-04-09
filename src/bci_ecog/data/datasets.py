from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class RawECoGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class SpectrogramECoGDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass
class DataloaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _dataset_for_mode(mode: str, features: np.ndarray, labels: np.ndarray) -> Dataset[tuple[torch.Tensor, torch.Tensor]]:
    if mode == "raw":
        return RawECoGDataset(features, labels)
    if mode == "spectrogram":
        return SpectrogramECoGDataset(features, labels)
    raise ValueError(f"Unsupported input mode: {mode}")


def build_dataloader(
    mode: str,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = _dataset_for_mode(mode, features, labels)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)
