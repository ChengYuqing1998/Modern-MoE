"""Parameter accounting for nanoK3 sparse activation."""

from __future__ import annotations

import torch.nn as nn

from .model import NanoK3ForCausalLM


def module_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def active_parameters(model: NanoK3ForCausalLM) -> int:
    """Parameters participating in one token's forward path.

    Shared embeddings/head and all attention/AttnRes parameters count as active.
    In an MoE layer, the router, latent projections, shared path, and exactly
    top-k routed experts count as active.
    """

    count = 0
    seen: set[int] = set()

    def add(module: nn.Module) -> None:
        nonlocal count
        for parameter in module.parameters():
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                count += parameter.numel()

    for module in (
        model.embed_tokens,
        model.norm,
        model.output_attn_res_norm,
        model.output_attn_res_proj,
        model.lm_head,
    ):
        add(module)
    for layer in model.layers:
        add(layer.self_attn)
        add(layer.input_norm)
        add(layer.post_attention_norm)
        add(layer.attention_res_norm)
        add(layer.mlp_res_norm)
        add(layer.attention_res_proj)
        add(layer.mlp_res_proj)
        if not layer.is_moe:
            add(layer.feed_forward)
            continue
        moe = layer.feed_forward
        add(moe.router)
        add(moe.down_project)
        add(moe.routed_norm)
        add(moe.up_project)
        add(moe.shared_experts)
        for index in range(model.config.num_experts_per_token):
            add(moe.experts[index])
    return count


def parameter_report(model: NanoK3ForCausalLM) -> dict[str, int | float]:
    total = model.num_parameters()
    active = active_parameters(model)
    return {
        "total_parameters": total,
        "active_parameters_per_token": active,
        "active_fraction": active / total,
    }
