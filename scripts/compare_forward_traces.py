"""Compare Native and MCore forward traces for the Modern-MoE training gate."""

from __future__ import annotations

import argparse
import json

import torch


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rms": float(diff.square().mean().sqrt()),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", required=True)
    parser.add_argument("--mcore", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    native = torch.load(args.native, map_location="cpu", weights_only=False)
    mcore = torch.load(args.mcore, map_location="cpu", weights_only=False)
    intermediate = mcore["intermediate"]
    result = {"nodes": {}}
    pairs = [("module.embedding", "embedding")]
    pairs += [(f"module.decoder.layers.{i}", f"layer{i}") for i in range(16)]
    pairs += [("module.decoder.final_layernorm", "final_norm")]
    for mcore_name, native_name in pairs:
        # MCore uses [sequence, batch, hidden], Native uses [batch, sequence, hidden].
        result["nodes"][native_name] = stats(
            native[native_name], intermediate[mcore_name].transpose(0, 1)
        )

    native_logits = native["final_logits"].float().reshape(-1)
    mcore_logits = mcore["mcore_logits"][:, -1, :].float().reshape(-1)
    result["logits"] = stats(native_logits, mcore_logits)
    for k in (1, 5, 10, 50):
        lhs = set(torch.topk(native_logits, k).indices.tolist())
        rhs = set(torch.topk(mcore_logits, k).indices.tolist())
        result["logits"][f"top{k}_overlap"] = len(lhs & rhs)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
