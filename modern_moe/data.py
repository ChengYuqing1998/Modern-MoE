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
        try:
            self.sample_starts = np.load(
                self.sample_index_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        except ValueError as error:
            raise ValueError(
                f"Could not load sample index {self.sample_index_path}; expected "
                "a one-dimensional .npy array with an integer dtype."
            ) from error

        if self.sample_starts.ndim != 1:
            raise ValueError("sample index must be a one-dimensional array")
        if not np.issubdtype(self.sample_starts.dtype, np.integer):
            raise ValueError(
                f"sample index must have an integer dtype, but "
                f"{self.sample_index_path} has dtype {self.sample_starts.dtype}"
            )
        if len(self.sample_starts):
            if np.issubdtype(self.sample_starts.dtype, np.signedinteger):
                minimum_start = int(np.min(self.sample_starts))
                if minimum_start < 0:
                    raise ValueError(
                        "sample index offsets must be non-negative, but "
                        f"{self.sample_index_path} contains {minimum_start}"
                    )

            maximum_start = int(np.max(self.sample_starts))
            sample_size = self.sequence_length + 1
            maximum_valid_start = len(self.tokens) - sample_size
            if maximum_start > maximum_valid_start:
                raise ValueError(
                    f"sample index offset {maximum_start} is out of range: each "
                    f"sample needs {sample_size} tokens, but {self.bin_path} "
                    f"contains {len(self.tokens)}. The largest valid offset is "
                    f"{maximum_valid_start}."
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
