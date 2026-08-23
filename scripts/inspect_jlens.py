from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_moe.jlens import JacobianLens, capture_residuals
from scripts.generate import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a prompt with J-Lens.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--position", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def visible_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id])
    return repr(text.replace("\n", "\\n"))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("J-Lens inspection for this KDA model requires CUDA")
    model, config = load_model(args.checkpoint)
    lens = JacobianLens.load(args.lens)
    if lens.hidden_size != config.hidden_size:
        raise ValueError("Lens hidden size does not match checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    input_ids = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to("cuda")
    with torch.no_grad():
        activations, model_logits = capture_residuals(model, input_ids)
        lens_logits = lens.read(model, activations, position=args.position)
    position = args.position % input_ids.size(1)
    print(
        f"token_position={position} "
        f"token={visible_token(tokenizer, input_ids[0, position].item())} "
        f"lens_samples={lens.samples}"
    )
    print("layer | J-Lens top tokens")
    print("-" * 88)
    for layer, logits in sorted(lens_logits.items()):
        ids = logits[0].topk(args.top_k).indices.tolist()
        words = "  ".join(visible_token(tokenizer, token_id) for token_id in ids)
        print(f"{layer:>5} | {words}")
    final_ids = model_logits[0, position].topk(args.top_k).indices.tolist()
    final_words = "  ".join(
        visible_token(tokenizer, token_id) for token_id in final_ids
    )
    print("-" * 88)
    print(f"model | {final_words}")


if __name__ == "__main__":
    main()
