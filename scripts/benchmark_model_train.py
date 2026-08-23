"""Benchmark a full ModernMoE training forward/backward without data I/O."""

from __future__ import annotations

import argparse
import statistics

import torch
import yaml

from modern_moe.config import ModernMoEConfig
from modern_moe.model import ModernMoEForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nanogptmoe_v2_500m.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=12)
    parser.add_argument(
        "--loss-logits-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    with open(args.config, encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values["attention_pattern"] = tuple(values["attention_pattern"])
    config = ModernMoEConfig(**values)
    model = ModernMoEForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, fused=True
    )
    tokens = torch.randint(
        0,
        config.vocab_size,
        (args.batch_size, args.sequence_length),
        device="cuda",
    )

    def step() -> None:
        model.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(tokens, mtp_targets=tokens)
                logits = output.logits
                if args.loss_logits_dtype == "fp32":
                    logits = logits.float()
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens.reshape(-1),
                )
                loss = (
                    loss
                    + config.router_aux_loss_coef * output.router_aux_loss
                    + config.router_z_loss_coef * output.router_z_loss
                ) / args.gradient_accumulation
            loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    print(
        f"full optimizer_step batch={args.batch_size} "
        f"sequence={args.sequence_length}: median={statistics.median(samples):.3f} ms "
        f"mean={statistics.mean(samples):.3f} ms "
        f"peak={torch.cuda.max_memory_allocated() / 1024**3:.3f} GiB"
    )
    print(
        f"allocated={torch.cuda.memory_allocated() / 1024**3:.3f} GiB "
        f"reserved={torch.cuda.memory_reserved() / 1024**3:.3f} GiB"
    )


if __name__ == "__main__":
    main()
