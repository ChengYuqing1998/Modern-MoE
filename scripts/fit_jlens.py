from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_moe.jlens import JacobianLens, jacobian_for_tokens
from scripts.generate import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a Jacobian Lens.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=1,
        help="Jacobian rows per backward pass; increase only with spare VRAM.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated zero-based layers, or 'all'.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing output lens.",
    )
    return parser.parse_args()


def iter_documents(path: Path):
    document = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                document.append(line)
            elif document:
                yield "".join(document).strip()
                document.clear()
    if document:
        yield "".join(document).strip()


def atomic_save(lens: JacobianLens, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    lens.save(temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("J-Lens fitting for this KDA model requires CUDA")
    if args.num_prompts < 1:
        raise ValueError("num-prompts must be positive")
    model, config = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    layers = (
        list(range(config.num_hidden_layers))
        if args.layers == "all"
        else sorted({int(value) for value in args.layers.split(",")})
    )
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(
                f"{args.output} already exists; pass --resume to continue"
            )
        previous = JacobianLens.load(args.output)
        if sorted(previous.matrices) != layers:
            raise ValueError("Existing lens layers do not match --layers")
        if previous.hidden_size != config.hidden_size:
            raise ValueError("Existing lens does not match checkpoint")
        sums = {
            layer: matrix.double() * previous.samples
            for layer, matrix in previous.matrices.items()
        }
        count = previous.samples
    else:
        sums = {
            layer: torch.zeros(
                config.hidden_size,
                config.hidden_size,
                dtype=torch.float64,
            )
            for layer in layers
        }
        count = 0
    if count >= args.num_prompts:
        print(
            f"already_complete fitted_prompts={count} output={args.output}",
            flush=True,
        )
        return
    accepted_documents = 0
    for text in iter_documents(args.prompts_file):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=args.sequence_length,
        ).input_ids
        if encoded.size(1) <= args.skip_first + 1:
            continue
        accepted_documents += 1
        if accepted_documents <= count:
            continue
        encoded = encoded.to("cuda")
        matrices = jacobian_for_tokens(
            model,
            encoded,
            layers,
            dim_batch=args.dim_batch,
            skip_first=args.skip_first,
        )
        for layer, matrix in matrices.items():
            sums[layer].add_(matrix.detach().cpu().double())
        count += 1
        lens = JacobianLens(
            matrices={
                layer: (matrix / count).float()
                for layer, matrix in sums.items()
            },
            hidden_size=config.hidden_size,
            samples=count,
            skip_first=args.skip_first,
        )
        atomic_save(lens, args.output)
        print(
            f"fitted_prompts={count}/{args.num_prompts} "
            f"saved={args.output}",
            flush=True,
        )
        if count >= args.num_prompts:
            break
    if count == 0:
        raise RuntimeError("No sufficiently long prompts were found")


if __name__ == "__main__":
    main()
