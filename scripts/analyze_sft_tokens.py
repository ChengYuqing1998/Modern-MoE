"""Count Qwen3 ChatML and assistant-supervised tokens in SFT JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def rendered_chatml(tokenizer, messages: list[dict[str, str]]) -> tuple[str, str]:
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    supervised: list[str] = []
    # Parse the already-rendered template so Qwen3's normalization of thinking
    # blocks is reflected exactly in both totals and supervised-token counts.
    for segment in full.split(IM_START)[1:]:
        role, separator, body = segment.partition("\n")
        if separator and role == "assistant" and IM_END in body:
            content = body.rsplit(IM_END, 1)[0]
            supervised.append(f"{content}{IM_END}\n")
    return full, "".join(supervised)


def percentile(values: list[int], q: float) -> int:
    return int(np.percentile(np.asarray(values, dtype=np.int64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/qwen3_moe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), use_fast=True)
    totals = Counter()
    by_category: dict[str, Counter] = {}
    lengths: list[int] = []
    assistant_lengths: list[int] = []
    batch_full: list[str] = []
    batch_assistant: list[str] = []
    batch_categories: list[str] = []

    def consume() -> None:
        if not batch_full:
            return
        full_ids = tokenizer(batch_full, add_special_tokens=False).input_ids
        assistant_ids = tokenizer(batch_assistant, add_special_tokens=False).input_ids
        for ids, assistant, category in zip(full_ids, assistant_ids, batch_categories):
            n = len(ids)
            a = len(assistant)
            lengths.append(n)
            assistant_lengths.append(a)
            totals["examples"] += 1
            totals["tokens"] += n
            totals["assistant_tokens"] += a
            totals["over_context_examples"] += n > args.context_length
            totals["over_context_tokens"] += max(0, n - args.context_length)
            counter = by_category.setdefault(category, Counter())
            counter["examples"] += 1
            counter["tokens"] += n
            counter["assistant_tokens"] += a
        batch_full.clear()
        batch_assistant.clear()
        batch_categories.clear()

    with args.input.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            full, assistant = rendered_chatml(tokenizer, row["messages"])
            batch_full.append(full)
            batch_assistant.append(assistant)
            batch_categories.append(row["category"])
            if len(batch_full) >= args.batch_size:
                consume()
    consume()

    stats = {
        "input": str(args.input),
        "tokenizer": str(args.tokenizer),
        "tokenizer_ids": len(tokenizer),
        "context_length": args.context_length,
        **dict(totals),
        "assistant_fraction": totals["assistant_tokens"] / totals["tokens"],
        "mean_tokens_per_example": totals["tokens"] / totals["examples"],
        "mean_assistant_tokens_per_example": totals["assistant_tokens"] / totals["examples"],
        "length_percentiles": {
            str(q): percentile(lengths, q) for q in (50, 75, 90, 95, 99, 100)
        },
        "assistant_length_percentiles": {
            str(q): percentile(assistant_lengths, q) for q in (50, 75, 90, 95, 99, 100)
        },
        "over_context_fraction": totals["over_context_examples"] / totals["examples"],
        "category_stats": {
            category: {
                **dict(counter),
                "assistant_fraction": counter["assistant_tokens"] / counter["tokens"],
            }
            for category, counter in by_category.items()
        },
        "supervision_policy": "Mask system/user/assistant-header; supervise assistant content and <|im_end|>.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
