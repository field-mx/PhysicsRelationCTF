"""PyTorch dataset over the frozen SDWPF forecasting origins."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SDWPFForecastDataset(Dataset):
    def __init__(self, cache: str | Path, split: str):
        cache = Path(cache)
        origins = np.load(cache / "window_origins.npz")
        if split not in origins.files:
            raise KeyError(f"unknown split {split}; choices={origins.files}")
        self.origins = origins[split].astype(np.int64)
        self.power_input = np.load(cache / "power_input.npy", mmap_mode="r")
        self.power_target = np.load(cache / "power_target.npy", mmap_mode="r")
        self.valid = np.load(cache / "target_valid.npy", mmap_mode="r")
        self.speed = np.load(cache / "speed_filled.npy", mmap_mode="r")
        self.direction = np.load(cache / "direction_from.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int):
        origin = int(self.origins[index])
        x = np.array(self.power_input[origin - 36 : origin], dtype=np.float32, copy=True)
        y = np.array(self.power_target[origin : origin + 6], dtype=np.float32, copy=True)
        valid = np.array(self.valid[origin : origin + 6], dtype=bool, copy=True)
        y[~valid] = 0.0
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        # Only the final observed SCADA state is used to construct the graph.
        speed = np.array(self.speed[origin - 1], dtype=np.float32, copy=True)
        direction = np.array(self.direction[origin - 1], dtype=np.float32, copy=True)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(valid),
            torch.from_numpy(speed),
            torch.from_numpy(direction),
            origin,
        )
