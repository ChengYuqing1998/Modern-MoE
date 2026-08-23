"""Strict CUDA Graph A/B for Apple's compiled CE and Triton CCE kernels."""

from __future__ import annotations

import argparse
import gc
import statistics
from pathlib import Path

import torch
import yaml

from modern_moe.config import ModernMoEConfig
from modern_moe.model import ModernMoEForCausalLM
from scripts.train import CUDAGraphedMicrobatch


def run(impl: str, config_path: Path, warmup: int, iterations: int):
    torch.manual_seed(1337)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["attention_pattern"] = tuple(values["attention_pattern"])
    config = ModernMoEConfig(**values)
    model = ModernMoEForCausalLM(config).to("cuda", dtype=torch.bfloat16).train()
    model.training_linear_ce_impl = impl
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    graph = CUDAGraphedMicrobatch(
        model=model,
        optimizer=optimizer,
        batch_size=2,
        sequence_length=2048,
        accumulation_steps=12,
        dtype=torch.bfloat16,
        router_aux_coef=config.router_aux_loss_coef,
        router_z_coef=config.router_z_loss_coef,
        mtp_loss_coef=config.mtp_loss_coef,
        warmup_steps=3,
    )
    generator = torch.Generator(device="cuda").manual_seed(7)
    tokens = torch.randint(config.vocab_size, (2, 2048), device="cuda", generator=generator)
    for _ in range(warmup):
        graph.replay(tokens, tokens)
    torch.cuda.synchronize()
    graph.metric_sums.zero_()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _, losses = graph.replay(tokens, tokens)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    loss_values = tuple(float(value.detach()) for value in losses)
    result = {
        "impl": impl,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
        "losses": loss_values,
        "peak_alloc": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved": torch.cuda.max_memory_reserved() / 1024**3,
    }
    del graph, optimizer, model, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/nanogptmoe_v2_500m_liger.yaml"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    results = [run(impl, args.config, args.warmup, args.iterations) for impl in ("torch_compile", "cce")]
    for result in results:
        print(
            f"impl={result['impl']} mean={result['mean']:.3f}ms "
            f"median={result['median']:.3f}ms min={result['minimum']:.3f}ms "
            f"max={result['maximum']:.3f}ms losses={result['losses']} "
            f"peak_alloc={result['peak_alloc']:.3f}GiB "
            f"peak_reserved={result['peak_reserved']:.3f}GiB"
        )
    baseline, candidate = results
    print(
        f"cce_minus_torch_compile_median={candidate['median'] - baseline['median']:.3f}ms "
        f"speedup={baseline['median'] / candidate['median']:.4f}x "
        f"lm_loss_abs_diff={abs(candidate['losses'][1] - baseline['losses'][1]):.8f}"
    )


if __name__ == "__main__":
    main()
