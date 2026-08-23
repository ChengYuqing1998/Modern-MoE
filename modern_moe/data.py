from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped, fixed-length causal language-model samples.

    Each sample consumes ``sequence_length + 1`` consecutive token IDs. The
    first ``sequence_length`` IDs are model inputs and the following IDs are
    their next-token targets. No padding is introduced.
    """

    def __init__(
        self,
        bin_path: str | Path,
        sample_index_path: str | Path,
        sequence_length: int,
        dtype: str = "uint32",
    ) -> None:
        self.bin_path = Path(bin_path)
        self.sample_index_path = Path(sample_index_path)
        self.sequence_length = int(sequence_length)

        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not self.bin_path.is_file():
            raise FileNotFoundError(self.bin_path)
        if not self.sample_index_path.is_file():
            raise FileNotFoundError(self.sample_index_path)

        self.tokens = np.memmap(self.bin_path, mode="r", dtype=np.dtype(dtype))
        self.sample_starts = np.load(self.sample_index_path, mmap_mode="r")

        if self.sample_starts.ndim != 1:
            raise ValueError("sample index must be a one-dimensional array")
        if len(self.sample_starts):
            final_end = int(np.max(self.sample_starts)) + self.sequence_length + 1
            if final_end > len(self.tokens):
                raise ValueError(
                    f"Last sample ends at token {final_end}, but the bin file "
                    f"contains only {len(self.tokens)} tokens."
                )

    def __len__(self) -> int:
        return len(self.sample_starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.sample_starts[index])
        stop = start + self.sequence_length + 1

        # Copy out of the read-only memmap before handing the storage to torch.
        sample = np.asarray(self.tokens[start:stop], dtype=np.int64).copy()
        sample = torch.from_numpy(sample)
        return sample[:-1], sample[1:]
