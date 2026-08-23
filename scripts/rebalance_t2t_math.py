"""Replace most of the final MiniMind block with unused FineMath records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO, Iterator


EOD_LINE = b"<|endoftext|>\n"
SEPARATOR = b"\n<|endoftext|>\n"


def segment_documents(
    handle: BinaryIO, start: int, length: int
) -> Iterator[bytes]:
    handle.seek(start)
    end = start + length
    document: list[bytes] = []
    while handle.tell() < end:
        line = handle.readline()
        if not line:
            break
        if handle.tell() > end:
            raise ValueError("source segment does not end on a line boundary")
        document.append(line)
        if line == EOD_LINE:
            yield b"".join(document)
            document.clear()
    if document or handle.tell() != end:
        raise ValueError("source segment does not end on an EOD boundary")


def copy_range(
    source: BinaryIO, destination: BinaryIO, start: int, length: int
) -> None:
    source.seek(start)
    remaining = length
    while remaining:
        chunk = source.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise EOFError("unexpected end of source training file")
        destination.write(chunk)
        remaining -= len(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", type=Path, default=Path("data/strict_mix_1gb/raw/train.txt")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/strict_mix_1gb/raw/manifest.json"),
    )
    parser.add_argument(
        "--finemath-cache",
        type=Path,
        default=Path("data/strict_mix_1gb/source_cache/finemath_4plus.jsonl"),
    )
    parser.add_argument("--t2t-target-bytes", type=int, default=30_000_000)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/strict_mix_1gb/raw/rebalance_t2t_math.json"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source_order = list(manifest["sources"])
    t2t_name = "minimind_t2t_strict"
    t2t_index = source_order.index(t2t_name)
    prefix_names = source_order[:t2t_index]
    # Local dictionary/novel additions appear after the original T2T block in
    # train.txt but are represented after it in the manifest.
    original_prefix_names = [
        name
        for name in prefix_names
        if name not in {"dictionary_corpus_local", "moe_bl_novels_local"}
    ]
    t2t_start = sum(
        manifest["sources"][name]["train"]["bytes"]
        for name in original_prefix_names
    )
    old_t2t = manifest["sources"][t2t_name]["train"]
    old_t2t_bytes = old_t2t["bytes"]
    old_t2t_documents = old_t2t["documents"]
    suffix_start = t2t_start + old_t2t_bytes
    source_size = args.train.stat().st_size
    if source_size != manifest["train_bytes"]:
        raise ValueError("train size does not match manifest")
    if suffix_start > source_size:
        raise ValueError("computed T2T segment exceeds train size")

    temporary = args.train.with_suffix(".rebalance.tmp")
    kept_t2t_bytes = kept_t2t_documents = 0
    fine_bytes = fine_documents = fine_duplicates = 0
    seen_finemath: set[bytes] = set()
    skip_finemath_documents = (
        manifest["sources"]["finemath_4plus"]["train"]["documents"]
        + manifest["sources"]["finemath_4plus"]["validation"]["documents"]
    )

    try:
        with args.train.open("rb") as source, temporary.open("wb") as output:
            copy_range(source, output, 0, t2t_start)
            for document in segment_documents(
                source, t2t_start, old_t2t_bytes
            ):
                if kept_t2t_bytes + len(document) > args.t2t_target_bytes:
                    break
                output.write(document)
                kept_t2t_bytes += len(document)
                kept_t2t_documents += 1

            math_budget = old_t2t_bytes - kept_t2t_bytes
            with args.finemath_cache.open("r", encoding="utf-8") as cache:
                for index, line in enumerate(cache):
                    if index < skip_finemath_documents:
                        continue
                    record = json.loads(line)
                    raw = record["text"].encode("utf-8") + SEPARATOR
                    digest = hashlib.blake2b(raw, digest_size=16).digest()
                    if digest in seen_finemath:
                        fine_duplicates += 1
                        continue
                    seen_finemath.add(digest)
                    if len(raw) > math_budget - fine_bytes:
                        continue
                    output.write(raw)
                    fine_bytes += len(raw)
                    fine_documents += 1
                    if fine_bytes == math_budget:
                        break

            copy_range(
                source,
                output,
                suffix_start,
                source_size - suffix_start,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.train)
    finally:
        if temporary.exists():
            temporary.unlink()

    report = {
        "format_version": 1,
        "train": str(args.train),
        "old_train_bytes": source_size,
        "new_train_bytes": args.train.stat().st_size,
        "old_t2t_bytes": old_t2t_bytes,
        "old_t2t_documents": old_t2t_documents,
        "new_t2t_bytes": kept_t2t_bytes,
        "new_t2t_documents": kept_t2t_documents,
        "removed_t2t_bytes": old_t2t_bytes - kept_t2t_bytes,
        "removed_t2t_documents": old_t2t_documents - kept_t2t_documents,
        "added_finemath_bytes": fine_bytes,
        "added_finemath_documents": fine_documents,
        "finemath_duplicate_records_rejected": fine_duplicates,
        "unfilled_bytes": math_budget - fine_bytes,
        "train_sha256": sha256(args.train),
    }
    manifest["sources"][t2t_name]["train"].update(
        {
            "bytes": kept_t2t_bytes,
            "documents": kept_t2t_documents,
        }
    )
    finemath_train = manifest["sources"]["finemath_4plus"]["train"]
    finemath_train["bytes"] += fine_bytes
    finemath_train["documents"] += fine_documents
    manifest["train_bytes"] = args.train.stat().st_size
    manifest["total_text_bytes"] = (
        manifest["train_bytes"] + manifest["validation_bytes"]
    )
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
