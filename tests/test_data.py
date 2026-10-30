import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# Keep these CPU-only data tests independent of optional model kernel imports.
DATA_MODULE_PATH = Path(__file__).parents[1] / "modern_moe" / "data.py"
DATA_MODULE_SPEC = importlib.util.spec_from_file_location(
    "modern_moe_data_for_tests",
    DATA_MODULE_PATH,
)
assert DATA_MODULE_SPEC is not None
assert DATA_MODULE_SPEC.loader is not None
DATA_MODULE = importlib.util.module_from_spec(DATA_MODULE_SPEC)
DATA_MODULE_SPEC.loader.exec_module(DATA_MODULE)
PackedTokenDataset = DATA_MODULE.PackedTokenDataset


class PackedTokenDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.bin_path = self.root / "tokens.bin"
        np.arange(8, dtype=np.uint32).tofile(self.bin_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_index(self, values: np.ndarray, name: str = "index.npy") -> Path:
        path = self.root / name
        np.save(path, values, allow_pickle=False)
        return path

    def test_integer_offsets_at_boundaries_keep_fixed_length_samples(self) -> None:
        for dtype in (np.int32, np.uint64):
            with self.subTest(dtype=dtype):
                index_path = self.write_index(
                    np.array([0, 4], dtype=dtype),
                    name=f"index-{np.dtype(dtype).name}.npy",
                )
                dataset = PackedTokenDataset(self.bin_path, index_path, 3)

                self.assertEqual(dataset.sample_starts.dtype, np.dtype(dtype))
                self.assertEqual(len(dataset), 2)
                first_x, first_y = dataset[0]
                last_x, last_y = dataset[1]
                self.assertEqual(first_x.dtype, torch.int64)
                self.assertEqual(first_x.tolist(), [0, 1, 2])
                self.assertEqual(first_y.tolist(), [1, 2, 3])
                self.assertEqual(last_x.tolist(), [4, 5, 6])
                self.assertEqual(last_y.tolist(), [5, 6, 7])

    def test_negative_offset_is_rejected_during_construction(self) -> None:
        index_path = self.write_index(np.array([-1], dtype=np.int64))

        with self.assertRaisesRegex(ValueError, r"non-negative.*-1"):
            PackedTokenDataset(self.bin_path, index_path, 3)

    def test_floating_point_indexes_are_rejected(self) -> None:
        cases = (
            np.array([0.5], dtype=np.float64),
            np.array([0.0], dtype=np.float32),
            np.array([], dtype=np.float64),
        )
        for position, values in enumerate(cases):
            with self.subTest(dtype=values.dtype, values=values):
                index_path = self.write_index(values, name=f"float-{position}.npy")
                with self.assertRaisesRegex(ValueError, r"integer dtype.*float"):
                    PackedTokenDataset(self.bin_path, index_path, 3)

    def test_out_of_range_offset_is_rejected(self) -> None:
        index_path = self.write_index(np.array([5], dtype=np.uint64))

        with self.assertRaisesRegex(
            ValueError,
            r"offset 5 is out of range.*largest valid offset is 4",
        ):
            PackedTokenDataset(self.bin_path, index_path, 3)

    def test_empty_integer_index_is_a_zero_length_dataset(self) -> None:
        index_path = self.write_index(np.array([], dtype=np.int16))

        dataset = PackedTokenDataset(self.bin_path, index_path, 3)

        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.sample_starts.dtype, np.dtype(np.int16))
        with self.assertRaises(IndexError):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
