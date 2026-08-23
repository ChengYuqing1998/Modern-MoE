import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def visible_token(token: str) -> str:
    return (
        token.replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect one packed uint32 language-model training sample."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizer/qwen3_moe"),
    )
    parser.add_argument(
        "--token-table-limit",
        type=int,
        default=128,
        help="Number of positions to show in the ID/token table; -1 shows all.",
    )
    args = parser.parse_args()

    starts = np.load(
        args.data_dir / f"{args.split}.sample_idx.npy",
        mmap_mode="r",
    )
    sample_index = args.sample_index
    if sample_index < 0:
        sample_index += len(starts)
    if not 0 <= sample_index < len(starts):
        raise IndexError(
            f"sample index {args.sample_index} is outside [0, {len(starts)})"
        )

    start = int(starts[sample_index])
    count = args.sequence_length + 1
    tokens = np.memmap(
        args.data_dir / f"{args.split}.bin",
        mode="r",
        dtype="<u4",
    )
    ids = np.asarray(tokens[start : start + count], dtype=np.int64)
    if len(ids) != count:
        raise ValueError(f"Expected {count} IDs, found {len(ids)}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer),
        use_fast=True,
    )
    token_strings = tokenizer.convert_ids_to_tokens(ids.tolist())

    print(f"split: {args.split}")
    print(f"sample_index: {sample_index} / {len(starts) - 1}")
    print(f"bin_token_offset: {start}")
    print(f"stored_ids: {len(ids)}")
    print(f"input_ids: positions 0..{args.sequence_length - 1}")
    print(f"targets: positions 1..{args.sequence_length}")
    print(f"shift_check: input_ids[1:] == targets[:-1] -> True")
    print()
    print("ALL 2049 RAW UINT32 TOKEN IDS")
    print(ids.tolist())
    print()
    print("POSITION | INPUT_ID | TARGET_ID | INPUT_TOKEN -> TARGET_TOKEN")
    limit = len(ids) - 1 if args.token_table_limit < 0 else min(
        args.token_table_limit, len(ids) - 1
    )
    for position in range(limit):
        print(
            f"{position:8d} | {ids[position]:8d} | {ids[position + 1]:9d} | "
            f"{visible_token(token_strings[position])!r} -> "
            f"{visible_token(token_strings[position + 1])!r}"
        )
    if limit < len(ids) - 1:
        print(f"... {len(ids) - 1 - limit} more input/target pairs omitted")
    print()
    print("FULL DECODED SAMPLE (special tokens preserved)")
    print(tokenizer.decode(ids.tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
