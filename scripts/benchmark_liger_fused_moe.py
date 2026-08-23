"""Strict training A/B: packed ScatterMoE versus Liger fused MoE."""

from __future__ import annotations

import argparse
import os
import statistics
import time

# Liger's full autotune can retain a working set per candidate and OOM.  The
# pinned configurations are also the reproducible production setting.
os.environ.setdefault("LIGER_FUSED_MOE_AUTOTUNE", "0")

import torch
import torch.nn.functional as F
from modern_moe.liger_moe import liger_fused_moe
from scattermoe.mlp import GLUMLP


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).item()


def errors(name: str, left: torch.Tensor, right: torch.Tensor) -> None:
    delta = (left.float() - right.float()).abs()
    print(
        f"{name}: max_abs={delta.max().item():.8g} "
        f"mean_abs={delta.mean().item():.8g} cosine={cosine(left, right):.9f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x0 = torch.randn(args.tokens, args.hidden, device=device, dtype=dtype)
    gate0 = torch.randn(
        args.experts, 2 * args.intermediate, args.hidden, device=device, dtype=dtype
    ) / args.hidden**0.5
    down0 = torch.randn(
        args.experts, args.hidden, args.intermediate, device=device, dtype=dtype
    ) / args.intermediate**0.5
    router_logits = torch.randn(args.tokens, args.experts, device=device)
    selected_logits, indices64 = router_logits.topk(args.top_k, dim=-1)
    weights0 = selected_logits.softmax(dim=-1).to(dtype)
    probe = torch.randn(args.tokens, args.hidden, device=device, dtype=torch.float32)

    scatter_module = GLUMLP(
        args.hidden, args.intermediate, args.experts, args.top_k, bias=False
    ).to(device=device, dtype=dtype)
    gate, up = gate0.chunk(2, dim=1)
    with torch.no_grad():
        scatter_module.experts.weight.copy_(torch.cat((up, gate), dim=1))
        scatter_module.output_experts.weight.copy_(down0)
    liger_gate_up = gate0.detach().clone().requires_grad_(True)
    liger_down = down0.detach().clone().requires_grad_(True)
    indices32 = indices64.to(torch.int32)

    def run_scatter(capture: bool = False):
        scatter_module.zero_grad(set_to_none=True)
        x = x0.detach().clone().requires_grad_(True)
        weights = weights0.detach().clone().requires_grad_(True)
        output = scatter_module(x, weights, indices64)
        loss = (output.float() * probe).sum() / args.tokens
        loss.backward()
        packed_grad = scatter_module.experts.weight.grad
        up_grad, gate_grad = packed_grad.chunk(2, dim=1)
        canonical_grad = torch.cat((gate_grad, up_grad), dim=1)
        if capture:
            return (
                output.detach(), x.grad.detach(), weights.grad.detach(),
                canonical_grad.detach(), scatter_module.output_experts.weight.grad.detach(),
            )

    def run_liger(capture: bool = False):
        liger_gate_up.grad = None
        liger_down.grad = None
        x = x0.detach().clone().requires_grad_(True)
        weights = weights0.detach().clone().requires_grad_(True)
        output = liger_fused_moe(
            x, liger_gate_up, liger_down, indices32, weights
        )
        loss = (output.float() * probe).sum() / args.tokens
        loss.backward()
        if capture:
            return (
                output.detach(), x.grad.detach(), weights.grad.detach(),
                liger_gate_up.grad.detach(), liger_down.grad.detach(),
            )

    print("compiling and checking correctness...", flush=True)
    scatter_values = run_scatter(capture=True)
    liger_values = run_liger(capture=True)
    for name, left, right in zip(
        ("output", "grad_x", "grad_router_weights", "grad_gate_up", "grad_down"),
        scatter_values,
        liger_values,
        strict=True,
    ):
        errors(name, left, right)

    def benchmark(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        return (
            statistics.mean(samples), statistics.median(samples), min(samples),
            torch.cuda.max_memory_allocated() / 1024**3,
            torch.cuda.max_memory_reserved() / 1024**3,
        )

    for name, fn in (("scattermoe", run_scatter), ("liger", run_liger)):
        mean, median, minimum, allocated, reserved = benchmark(fn)
        print(
            f"{name}: mean={mean:.3f}ms median={median:.3f}ms min={minimum:.3f}ms "
            f"peak_alloc={allocated:.3f}GiB peak_reserved={reserved:.3f}GiB"
        )


if __name__ == "__main__":
    main()
