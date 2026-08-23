from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_moe.jlens import JacobianLens, capture_residuals_and_routes
from modern_moe.jlens_visualization import write_jlens_html
from scripts.generate import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an interactive J-Lens HTML.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--position-chunk", type=int, default=8)
    return parser.parse_args()


def token_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id])
    return text.replace("\n", "↵").replace("\t", "⇥") or "∅"


def layer_cells(
    logits: torch.Tensor,
    tokenizer,
    top_k: int,
) -> list[dict]:
    values, indices = logits.float().topk(top_k, dim=-1)
    margins = values[:, 0] - values[:, 1] if top_k > 1 else values[:, 0]
    return [
        {
            "top": [
                token_text(tokenizer, token_id)
                for token_id in indices[position].tolist()
            ],
            "ids": indices[position].tolist(),
            "margin": float(margins[position]),
        }
        for position in range(logits.size(0))
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("J-Lens visualization for this KDA model requires CUDA")
    if args.top_k < 2:
        raise ValueError("top-k must be at least 2")
    model, config = load_model(args.checkpoint)
    lens = JacobianLens.load(args.lens)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    input_ids = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_tokens,
    ).input_ids.to("cuda")
    with torch.no_grad():
        activations, model_logits, routes = capture_residuals_and_routes(
            model,
            input_ids,
        )
    weight = model.lm_head.weight.float()
    jlens_rows, logit_rows = [], []
    layers = sorted(lens.matrices)
    for layer in layers:
        j_cells, l_cells = [], []
        matrix = lens.matrices[layer].to("cuda")
        layer_states = activations[layer][0]
        states = layer_states.float()
        for start in range(0, states.size(0), args.position_chunk):
            hidden = states[start : start + args.position_chunk]
            j_logits = (hidden @ matrix.T) @ weight.T
            native_hidden = layer_states[start : start + args.position_chunk]
            logit_logits = model.lm_head(model.norm(native_hidden)).float()
            j_cells.extend(layer_cells(j_logits, tokenizer, args.top_k))
            l_cells.extend(layer_cells(logit_logits, tokenizer, args.top_k))
        jlens_rows.append(j_cells)
        logit_rows.append(l_cells)
        print(f"visualized_layer={layer}", flush=True)

    final_ids = model_logits[0].argmax(dim=-1).tolist()
    crystallization = []
    for position, final_id in enumerate(final_ids):
        found = -1
        for layer_index in range(max(0, len(layers) - 1)):
            here = final_id in jlens_rows[layer_index][position]["ids"]
            following = final_id in jlens_rows[layer_index + 1][position]["ids"]
            if here and following:
                found = layers[layer_index]
                break
        crystallization.append(found)
    payload = {
        "title": "Modern-MoE · Jacobian Lens",
        "prompt": args.prompt,
        "samples": lens.samples,
        "layers": layers,
        "tokens": [
            token_text(tokenizer, token_id) for token_id in input_ids[0].tolist()
        ],
        "jlens": jlens_rows,
        "logit": logit_rows,
        "routes": [
            routes[layer][0].detach().cpu().tolist() for layer in layers
        ],
        "crystallization": crystallization,
    }
    write_jlens_html(payload, args.output)
    print(f"saved={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
