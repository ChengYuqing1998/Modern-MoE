"""Build mmap-friendly token binaries from EOD-delimited UTF-8 text."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np
import yaml
from transformers import AutoTokenizer

from modern_moe import ModernMoEConfig


EOD_TEXT = "<|endoftext|>"
TOKEN_DTYPE = np.dtype("<u4")
INDEX_DTYPE = np.dtype("<u8")


def iter_documents(path: Path) -> Iterator[str]:
    parts: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.rstrip("\n") == EOD_TEXT:
                text = "".join(parts).strip()
                if text:
                    yield text
                parts.clear()
            else:
                parts.append(line)
    text = "".join(parts).strip()
    if text:
        yield text


def batched(items: Iterable[str], batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_split(
    input_path: Path,
    output_dir: Path,
    split: str,
    tokenizer,
    context_length: int,
    batch_size: int,
    shuffle_seed: int,
) -> dict:
    bin_path = output_dir / f"{split}.bin"
    idx_path = output_dir / f"{split}.idx"
    sample_path = output_dir / f"{split}.sample_idx.npy"
    offsets = [0]
    token_count = 0
    document_count = 0
    min_token, max_token = None, None

    with bin_path.open("wb") as token_file:
        for texts in batched(iter_documents(input_path), batch_size):
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            for token_ids in encoded:
                token_ids.append(tokenizer.eos_token_id)
                array = np.asarray(token_ids, dtype=TOKEN_DTYPE)
                array.tofile(token_file)
                token_count += array.size
                document_count += 1
                offsets.append(token_count)
                current_min, current_max = int(array.min()), int(array.max())
                min_token = current_min if min_token is None else min(min_token, current_min)
                max_token = current_max if max_token is None else max(max_token, current_max)

    np.asarray(offsets, dtype=INDEX_DTYPE).tofile(idx_path)
    # Each causal sample consumes 2049 tokens: 2048 inputs and their shifted labels.
    sample_count = max(0, (token_count - 1) // context_length)
    sample_offsets = np.arange(sample_count, dtype=np.uint64) * context_length
    if split == "train":
        np.random.default_rng(shuffle_seed).shuffle(sample_offsets)
    np.save(sample_path, sample_offsets, allow_pickle=False)

    return {
        "input": str(input_path),
        "bin": str(bin_path),
        "idx": str(idx_path),
        "sample_idx": str(sample_path),
        "documents": document_count,
        "tokens": token_count,
        "samples_2048": sample_count,
        "dropped_tail_tokens": max(0, token_count - (sample_count * context_length + 1)),
        "min_token_id": min_token,
        "max_token_id": max_token,
        "bin_bytes": bin_path.stat().st_size,
        "idx_bytes": idx_path.stat().st_size,
        "bin_sha256": sha256(bin_path),
        "idx_sha256": sha256(idx_path),
        "sample_idx_sha256": sha256(sample_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/pilot_235mb/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/pilot_235mb/tokenized_qwen3_ctx2048"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/modern_moe_pilot.yaml"))
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation"),
        default=("train", "validation"),
        help="Tokenize only the requested splits.",
    )
    args = parser.parse_args()

    if args.context_length < 1:
        raise ValueError("context-length must be positive")
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw["attention_pattern"] = tuple(raw["attention_pattern"])
    config = ModernMoEConfig(**raw)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    if len(tokenizer) > config.vocab_size:
        raise ValueError("tokenizer IDs exceed model embedding rows")
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 1,
        "layout": "flat little-endian uint32 token IDs + little-endian uint64 document offsets",
        "tokenizer": "Qwen/Qwen3-30B-A3B-Base",
        "tokenizer_path": config.tokenizer_path,
        "tokenizer_ids": len(tokenizer),
        "model_vocab_size": config.vocab_size,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "context_length": args.context_length,
        "tokens_per_training_sample": args.context_length + 1,
        "packing": "continuous, no padding; documents separated by one EOS",
        "shuffle_seed": args.shuffle_seed,
        "splits": {},
    }
    for split in args.splits:
        print(f"tokenizing {split} ...", flush=True)
        metadata["splits"][split] = tokenize_split(
            args.input_dir / f"{split}.txt",
            args.output_dir,
            split,
            tokenizer,
            args.context_length,
            args.batch_size,
            args.shuffle_seed,
        )
        print(json.dumps(metadata["splits"][split], indent=2), flush=True)
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
