import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tokenize_corpus import tokenize_split


class SyntheticTokenizer:
    eos_token_id = 7

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [list(range(int(text))) for text in texts]}


class TokenizeSplitMetadataTest(unittest.TestCase):
    def run_split(self, token_count, context_length):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.txt"
            input_path.write_text(f"{token_count - 1}\n", encoding="utf-8")

            metadata = tokenize_split(
                input_path=input_path,
                output_dir=root,
                split="validation",
                tokenizer=SyntheticTokenizer(),
                context_length=context_length,
                batch_size=1,
                shuffle_seed=42,
            )
            sample_offsets = np.load(root / "validation.sample_idx.npy")
            return metadata, sample_offsets

    def test_context_four_metadata_is_context_neutral(self):
        metadata, sample_offsets = self.run_split(token_count=10, context_length=4)

        self.assertEqual(metadata["tokens"], 10)
        self.assertEqual(metadata["samples"], 2)
        self.assertEqual(metadata["dropped_tail_tokens"], 1)
        self.assertNotIn("samples_2048", metadata)
        opaque_fields = {
            "input",
            "bin",
            "idx",
            "sample_idx",
            "bin_sha256",
            "idx_sha256",
            "sample_idx_sha256",
        }
        descriptive_metadata = {
            key: value for key, value in metadata.items() if key not in opaque_fields
        }
        self.assertNotIn("2048", json.dumps(descriptive_metadata))
        np.testing.assert_array_equal(sample_offsets, np.array([0, 4], dtype=np.uint64))

    def test_default_context_preserves_legacy_count(self):
        metadata, sample_offsets = self.run_split(token_count=2050, context_length=2048)

        self.assertEqual(metadata["tokens"], 2050)
        self.assertEqual(metadata["samples"], 1)
        self.assertEqual(metadata["samples_2048"], 1)
        self.assertEqual(metadata["dropped_tail_tokens"], 1)
        np.testing.assert_array_equal(sample_offsets, np.array([0], dtype=np.uint64))


if __name__ == "__main__":
    unittest.main()
