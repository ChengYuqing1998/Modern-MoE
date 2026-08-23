from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SFTPackedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-length mmap SFT samples with assistant-only ``-100`` labels."""

    def __init__(
        self,
        input_path: str | Path,
        label_path: str | Path,
        sequence_length: int,
    ) -> None:
        self.input_path = Path(input_path)
        self.label_path = Path(label_path)
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.input_path.is_file():
            raise FileNotFoundError(self.input_path)
        if not self.label_path.is_file():
            raise FileNotFoundError(self.label_path)
        self.inputs = np.memmap(self.input_path, mode="r", dtype=np.uint32)
        self.labels = np.memmap(self.label_path, mode="r", dtype=np.int32)
        if len(self.inputs) != len(self.labels):
            raise ValueError("SFT input and label files contain different token counts")
        if len(self.inputs) % self.sequence_length:
            raise ValueError("SFT mmap length is not divisible by sequence_length")

    def __len__(self) -> int:
        return len(self.inputs) // self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(index) * self.sequence_length
        stop = start + self.sequence_length
        inputs = np.asarray(self.inputs[start:stop], dtype=np.int64).copy()
        labels = np.asarray(self.labels[start:stop], dtype=np.int64).copy()
        return torch.from_numpy(inputs), torch.from_numpy(labels)
