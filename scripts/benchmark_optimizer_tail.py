"""Split clip_grad_norm_ and AdamW timing after graphed accumulation."""

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


def run(kind: str, config_path: Path, warmup: int, iterations: int):
    torch.manual_seed(1337)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    values["attention_pattern"] = tuple(values["attention_pattern"])
    config = ModernMoEConfig(**values)
    model = ModernMoEForCausalLM(config).to("cuda", dtype=torch.bfloat16).train()
    model.training_linear_ce_impl = "cce"
    kwargs = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    if kind in ("fused", "fused_graph"):
        optimizer = torch.optim.AdamW(
            model.parameters(), fused=True, capturable=kind == "fused_graph", **kwargs
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), foreach=True, **kwargs)
    graph = CUDAGraphedMicrobatch(
        model, optimizer, 2, 2048, 12, torch.bfloat16,
        config.router_aux_loss_coef, config.router_z_loss_coef,
        config.mtp_loss_coef, warmup_steps=3,
    )
    generator = torch.Generator(device="cuda").manual_seed(7)
    tokens = torch.randint(config.vocab_size, (2, 2048), device="cuda", generator=generator)

    tail_graph = None

    def eager_tail():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        optimizer.step()

    def update(measure: bool):
        optimizer.zero_grad(set_to_none=False)
        for _ in range(12):
            graph.replay(tokens, tokens)
        accumulation_end = torch.cuda.Event(enable_timing=True)
        clip_end = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        accumulation_end.record()
        if tail_graph is None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            clip_end.record()
            optimizer.step()
        else:
            # The graph contains both operations, so only their combined tail
            # is meaningful; record the boundary in the combined slot.
            clip_end.record()
            tail_graph.replay()
        step_end.record()
        step_end.synchronize()
        if measure:
            return accumulation_end.elapsed_time(clip_end), clip_end.elapsed_time(step_end)

    for _ in range(warmup):
        update(False)
    if kind == "fused_graph":
        optimizer.zero_grad(set_to_none=False)
        for _ in range(12):
            graph.replay(tokens, tokens)
        torch.cuda.synchronize()
        tail_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(tail_graph):
            eager_tail()
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = [update(True) for _ in range(iterations)]
    clips = [sample[0] for sample in samples]
    steps = [sample[1] for sample in samples]
    result = dict(
        kind=kind,
        clip=statistics.median(clips),
        step=statistics.median(steps),
        total=statistics.median([a + b for a, b in samples]),
        clip_mean=statistics.mean(clips),
        step_mean=statistics.mean(steps),
        peak_alloc=torch.cuda.max_memory_allocated() / 1024**3,
        peak_reserved=torch.cuda.max_memory_reserved() / 1024**3,
    )
    del graph, optimizer, model, tokens
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/nanogptmoe_v2_500m_liger.yaml"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    for kind in ("fused", "foreach", "fused_graph"):
        result = run(kind, args.config, args.warmup, args.iterations)
        print(
            f"adamw={kind} clip_median={result['clip']:.3f}ms "
            f"step_median={result['step']:.3f}ms tail_median={result['total']:.3f}ms "
            f"clip_mean={result['clip_mean']:.3f}ms step_mean={result['step_mean']:.3f}ms "
            f"peak_alloc={result['peak_alloc']:.3f}GiB "
            f"peak_reserved={result['peak_reserved']:.3f}GiB"
        )


if __name__ == "__main__":
    main()
