#!/usr/bin/env python3
"""Compare Native and SGLang forward traces and report the first divergence."""

import argparse
import math
from collections import OrderedDict

import torch


ORDER = [
    "embedding",
    "layer0.input_norm",
    "layer0.attention.q_proj",
    "layer0.attention.k_proj",
    "layer0.attention.v_proj",
    "layer0.attention",
    "layer0.post_attention_norm",
    "layer0.moe",
    "layer0",
    "final_norm",
    "final_logits",
]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def canonical(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"trace value is not a tensor: {type(x)!r}")
    return x.detach().float().reshape(-1)


def cosine(a, b):
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return torch.dot(a, b).item() / denom.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("native")
    ap.add_argument("sglang")
    ap.add_argument("--cos-threshold", type=float, default=0.999)
    args = ap.parse_args()

    native = load(args.native)
    sglang = load(args.sglang)
    print(f"native input_ids: {native.get('input_ids', '<missing>')}")
    print(f"sglang input_ids: {sglang.get('input_ids', '<missing>')}")

    if "input_ids" in native and "input_ids" in sglang:
        same = torch.equal(native["input_ids"].cpu(), sglang["input_ids"].cpu())
        print(f"input_ids_equal: {same}")

    keys = list(OrderedDict.fromkeys(ORDER + list(native.keys()) + list(sglang.keys())))
    first_bad = None
    for key in keys:
        if key not in native or key not in sglang:
            continue
        if key in {"input_ids", "top_ids", "prompt", "rendered_prompt"}:
            continue
        try:
            a = canonical(native[key])
            b = canonical(sglang[key])
        except (TypeError, AttributeError):
            continue
        if a.numel() != b.numel():
            print(f"{key:32s} SHAPE_NUMEL_MISMATCH native={a.numel()} sglang={b.numel()}")
            if first_bad is None:
                first_bad = key
            continue
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        rel = max_abs / max(a.abs().max().item(), 1e-12)
        cs = cosine(a, b)
        bad = (cs < args.cos_threshold) or (not math.isfinite(max_abs))
        marker = " ❌ FIRST/DIFF" if bad and first_bad is None else (" ❌" if bad else " ✅")
        if bad and first_bad is None:
            first_bad = key
        print(
            f"{key:32s} native={tuple(native[key].shape) if hasattr(native[key], 'shape') else '?'} "
            f"sglang={tuple(sglang[key].shape) if hasattr(sglang[key], 'shape') else '?'} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rel_max={rel:.6e} cosine={cs:.9f}{marker}"
        )

    print(f"first_significant_divergence: {first_bad or '<none>'}")


if __name__ == "__main__":
    main()
