#!/usr/bin/env python3
"""Collect selected repository text files into TXT plus document-level JSONL."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--glob", action="append", required=True, dest="globs")
    parser.add_argument("--text-output", required=True, type=Path)
    parser.add_argument("--jsonl-output", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--min-chars", type=int, default=200)
    args = parser.parse_args()

    paths = sorted({path for pattern in args.globs for path in args.root.glob(pattern)})
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    kept = failed = 0
    with (
        args.text_output.open("w", encoding="utf-8") as text_out,
        args.jsonl_output.open("w", encoding="utf-8") as jsonl_out,
    ):
        for path in paths:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
                text = "".join(
                    char
                    for char in text
                    if char != "\ufffd"
                    and (
                        not unicodedata.category(char).startswith("C")
                        or char in "\n\t"
                    )
                ).strip()
            except (UnicodeDecodeError, OSError):
                failed += 1
                continue
            if len(text) < args.min_chars:
                continue
            relative = str(path.relative_to(args.root))
            text_out.write(f"# {relative}\n\n{text}\n\n")
            jsonl_out.write(json.dumps({
                "title": relative,
                "text": text,
                "source": args.source,
                "source_path": relative,
                "language": args.language,
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(json.dumps({"files_found": len(paths), "kept": kept, "failed": failed}))


if __name__ == "__main__":
    main()
