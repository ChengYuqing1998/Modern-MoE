#!/usr/bin/env python3
"""Tokenize the three-phase train data with visible progress.

Only train.txt is tokenized. The existing Strict Mix V4 validation binaries are
hard-linked into every output directory so all runs can reuse the exact same
fixed 60-sample monitor at 2048 context length.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from scripts.tokenize_corpus import batched


ROOT = Path("data/pretrain")
TOKENIZER_PATH = Path("tokenizer/qwen3_moe")
CONTEXT_LENGTH = 2048
# Keep each tokenizer input comfortably below Transformers' hard maximum
# sequence length.  Some of the merged textbook/source files do not contain
# EOD separators, so without this bound the whole file can become one giant
# "document" (tens of millions of tokens).
# 32k characters leaves a large safety margin even for code/symbol-heavy
# text whose tokenizer expansion can exceed two tokens per character.
MAX_DOCUMENT_CHARS = 32_000
EOS_ID = 151643
TOKEN_DTYPE = np.dtype("<u4")
INDEX_DTYPE = np.dtype("<u8")
VALIDATION_SOURCE = Path("data/strict_mix_1gb/tokenized_qwen3_ctx2048_v4")
VALIDATION_MONITOR = Path(
    "configs/validation_monitor_incremental_hq_300mb_v1.json"
)
PHASES = {
    "phase-1": ROOT / "phase-1",
    "phase-2": ROOT / "phase-2",
    "phase-3-open-source": ROOT / "phase-3/open-source",
    "phase-3-with-legacy-1gb": ROOT / "phase-3/with-legacy-1gb",
}


def iter_documents_lenient(path: Path, decode_stats: dict):
    """Read EOD-delimited text, replacing bad UTF-8 and chunking long docs.

    A few source files are effectively single documents because they have no
    ``<|endoftext|>`` markers.  Yielding bounded chunks prevents the tokenizer
    from constructing a 50M-token sequence and keeps memory/progress stable.
    """
    parts = []
    chars = 0

    def flush(force: bool = False):
        nonlocal parts, chars
        while parts and (force or chars >= MAX_DOCUMENT_CHARS):
            text = "".join(parts).strip()
            if not text:
                parts.clear()
                chars = 0
                return
            if len(text) <= MAX_DOCUMENT_CHARS:
                parts.clear()
                chars = 0
                decode_stats["replacement_chars"] += text.count("\ufffd")
                yield text
                return
            # Split a very long line/document without materialising another
            # multi-megabyte string.  Preserve all content; only boundaries
            # (and the normal EOS inserted by the writer) change.
            cut = MAX_DOCUMENT_CHARS
            chunk = text[:cut].strip()
            remainder = text[cut:]
            if chunk:
                decode_stats["replacement_chars"] += chunk.count("\ufffd")
                yield chunk
            parts = [remainder]
            chars = len(remainder)
            if not force and chars < MAX_DOCUMENT_CHARS:
                return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.rstrip("\n") == "<|endoftext|>":
                yield from flush(force=True)
                parts.clear()
                chars = 0
            else:
                parts.append(line)
                chars += len(line)
                yield from flush()
    yield from flush(force=True)


def link_validation(output_dir: Path) -> None:
    for name in ("validation.bin", "validation.idx", "validation.sample_idx.npy"):
        source = VALIDATION_SOURCE / name
        target = output_dir / name
        if target.exists():
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)


def tokenize_phase(
    phase_name: str,
    phase_dir: Path,
    tokenizer,
    batch_size: int,
    progress_every: int,
    shuffle_seed: int,
    force: bool,
) -> None:
    input_path = phase_dir / "raw/train.txt"
    output_dir = phase_dir / "tokenized_qwen3_ctx2048"
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not force:
        print(f"[{phase_name}] already complete: {metadata_path}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    partial = [output_dir / "train.bin", output_dir / "train.idx"]
    if any(path.exists() for path in partial) and not force:
        raise SystemExit(
            f"[{phase_name}] partial output exists; rerun with --force to overwrite"
        )

    print(
        f"[{phase_name}] START input={input_path} "
        f"bytes={input_path.stat().st_size:,} batch_size={batch_size} "
        f"context={CONTEXT_LENGTH}",
        flush=True,
    )
    started = time.perf_counter()
    last_report = started
    documents = 0
    token_count = 0
    decode_stats = {"replacement_chars": 0}
    offsets = [0]
    bin_path = output_dir / "train.bin"
    idx_path = output_dir / "train.idx"
    sample_path = output_dir / "train.sample_idx.npy"

    with bin_path.open("wb") as token_file:
        for texts in batched(iter_documents_lenient(input_path, decode_stats), batch_size):
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            for token_ids in encoded:
                token_ids.append(EOS_ID)
                array = np.asarray(token_ids, dtype=TOKEN_DTYPE)
                array.tofile(token_file)
                token_count += int(array.size)
                documents += 1
                offsets.append(token_count)
            if documents % progress_every < len(texts):
                now = time.perf_counter()
                elapsed = now - started
                interval = now - last_report
                print(
                    f"[{phase_name}] documents={documents:,} "
                    f"tokens={token_count:,} bin={token_file.tell():,} bytes "
                    f"elapsed={elapsed / 60:.1f}m "
                    f"speed={token_count / max(elapsed, 1):,.0f} tok/s "
                    f"last_interval={interval:.1f}s",
                    flush=True,
                )
                last_report = now

    print(f"[{phase_name}] writing train.idx ...", flush=True)
    np.asarray(offsets, dtype=INDEX_DTYPE).tofile(idx_path)
    sample_count = max(0, (token_count - 1) // CONTEXT_LENGTH)
    print(
        f"[{phase_name}] building and shuffling {sample_count:,} sample offsets ...",
        flush=True,
    )
    sample_offsets = np.arange(sample_count, dtype=np.uint64) * CONTEXT_LENGTH
    np.random.default_rng(shuffle_seed).shuffle(sample_offsets)
    np.save(sample_path, sample_offsets, allow_pickle=False)
    link_validation(output_dir)

    elapsed = time.perf_counter() - started
    metadata = {
        "format_version": 1,
        "phase": phase_name,
        "input": str(input_path),
        "tokenizer_path": str(TOKENIZER_PATH),
        "eos_token_id": EOS_ID,
        "context_length": CONTEXT_LENGTH,
        "max_document_chars": MAX_DOCUMENT_CHARS,
        "tokens_per_training_sample": CONTEXT_LENGTH + 1,
        "packing": "continuous, no padding; documents separated by one EOS",
        "shuffle_seed": shuffle_seed,
        "train": {
            "documents": documents,
            "tokens": token_count,
            "samples": sample_count,
            "bin": str(bin_path),
            "bin_bytes": bin_path.stat().st_size,
            "idx": str(idx_path),
            "sample_idx": str(sample_path),
        },
        "validation": {
            "reused_from": str(VALIDATION_SOURCE),
            "fixed_monitor_config": str(VALIDATION_MONITOR),
            "fixed_monitor_samples": 60,
            "context_length": 2048,
        },
        "elapsed_seconds": elapsed,
        "utf8_replacement_chars": decode_stats["replacement_chars"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{phase_name}] COMPLETE documents={documents:,} tokens={token_count:,} "
        f"samples={sample_count:,} elapsed={elapsed / 60:.1f}m "
        f"output={output_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("all", *PHASES.keys()),
        default="all",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--shuffle-seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    print(f"loading tokenizer from {TOKENIZER_PATH} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=True)
    if tokenizer.eos_token_id != EOS_ID:
        raise ValueError(
            f"Unexpected EOS ID {tokenizer.eos_token_id}; expected {EOS_ID}"
        )
    selected = PHASES.items() if args.phase == "all" else [(args.phase, PHASES[args.phase])]
    for phase_name, phase_dir in selected:
        tokenize_phase(
            phase_name,
            phase_dir,
            tokenizer,
            args.batch_size,
            args.progress_every,
            args.shuffle_seed,
            args.force,
        )


if __name__ == "__main__":
    main()
