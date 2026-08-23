"""Compare padded and grouped-mm SparseMoE outputs and gradients on CUDA."""

from __future__ import annotations

import argparse

import torch
import yaml

from modern_moe.config import ModernMoEConfig
from modern_moe.layers import SparseMoE


def difference(left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
    delta = (left.float() - right.float()).abs()
    scale = torch.maximum(left.float().abs(), right.float().abs()).clamp_min(1e-6)
    return delta.max().item(), (delta / scale).max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nanogptmoe_v2_500m.yaml")
    parser.add_argument("--tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values["attention_pattern"] = tuple(values["attention_pattern"])
    config = ModernMoEConfig(**values)
    torch.manual_seed(20260813)
    padded = SparseMoE(config).to(device="cuda", dtype=torch.bfloat16).train()
    grouped = SparseMoE(config).to(device="cuda", dtype=torch.bfloat16).train()
    grouped.load_state_dict(padded.state_dict())
    padded.use_grouped_mm = False
    grouped.use_grouped_mm = True

    source = torch.randn(
        args.tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    padded_input = source.detach().clone().requires_grad_(True)
    grouped_input = source.detach().clone().requires_grad_(True)
    padded_values = padded(padded_input, compute_router_losses=True)
    grouped_values = grouped(grouped_input, compute_router_losses=True)
    padded_loss = sum(value.float().square().mean() for value in padded_values)
    grouped_loss = sum(value.float().square().mean() for value in grouped_values)
    padded_loss.backward()
    grouped_loss.backward()

    print("output max_abs/max_rel:", difference(padded_values[0], grouped_values[0]))
    print("input_grad max_abs/max_rel:", difference(padded_input.grad, grouped_input.grad))
    worst_name = ""
    worst_abs = worst_rel = 0.0
    grouped_parameters = dict(grouped.named_parameters())
    for name, parameter in padded.named_parameters():
        absolute, relative = difference(parameter.grad, grouped_parameters[name].grad)
        if absolute > worst_abs:
            worst_name, worst_abs, worst_rel = name, absolute, relative
    print(
        f"worst_parameter={worst_name} max_abs={worst_abs:.8g} "
        f"max_rel={worst_rel:.8g}"
    )


if __name__ == "__main__":
    main()
