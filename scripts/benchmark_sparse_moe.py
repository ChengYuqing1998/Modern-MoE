"""Microbenchmark the SparseMoE training and decode hot paths on CUDA."""

from __future__ import annotations

import argparse
import statistics

import torch
import yaml

from modern_moe.config import ModernMoEConfig
from modern_moe.layers import SparseMoE


def elapsed_ms(fn, *, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), statistics.mean(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nanogptmoe_v2_500m.yaml")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    with open(args.config, encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values["attention_pattern"] = tuple(values["attention_pattern"])
    config = ModernMoEConfig(**values)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    module = SparseMoE(config).to(device=device, dtype=dtype)

    module.eval()
    decode_x = torch.randn(1, 1, config.hidden_size, device=device, dtype=dtype)
    with torch.inference_mode():
        decode = lambda: module(decode_x, compute_router_losses=False)[0]
        median, mean = elapsed_ms(decode, warmup=args.warmup, iterations=args.iterations)
    print(f"decode tokens=1: median={median:.4f} ms mean={mean:.4f} ms")

    module.train()
    if args.compile:
        module = torch.compile(
            module,
            mode="max-autotune-no-cudagraphs",
            fullgraph=False,
        )
    train_x = torch.randn(
        args.tokens, config.hidden_size, device=device, dtype=dtype
    )
    forward = lambda: module(train_x, compute_router_losses=True)[0]
    median, mean = elapsed_ms(forward, warmup=args.warmup, iterations=args.iterations)
    print(f"train forward tokens={args.tokens}: median={median:.4f} ms mean={mean:.4f} ms")

    def forward_backward() -> None:
        module.zero_grad(set_to_none=True)
        x = train_x.detach().requires_grad_(True)
        output, aux_loss, z_loss = module(x, compute_router_losses=True)
        (output.square().mean() + 0.01 * aux_loss + 0.001 * z_loss).backward()

    median, mean = elapsed_ms(
        forward_backward,
        warmup=max(3, args.warmup // 4),
        iterations=max(10, args.iterations // 5),
    )
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"train forward+backward tokens={args.tokens}: "
        f"median={median:.4f} ms mean={mean:.4f} ms peak={peak_mib:.1f} MiB"
    )
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
        ) as profile:
            forward_backward()
        torch.cuda.synchronize()
        print(profile.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=30
        ))


if __name__ == "__main__":
    main()
