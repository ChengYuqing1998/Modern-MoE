"""Training-only ScatterMoE dispatch with checkpoint-compatible parameters."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_SHARED_ROUTING_CACHE: dict[tuple[torch.device, torch.dtype, int, int], tuple] = {}


def _shared_routing_metadata(flat_x, shared_count, index_dtype):
    key = (flat_x.device, index_dtype, flat_x.size(0), shared_count)
    cached = _SHARED_ROUTING_CACHE.get(key)
    if cached is not None:
        return cached
    token_count = flat_x.size(0)
    sorted_experts = torch.arange(
        shared_count, device=flat_x.device, dtype=index_dtype
    ).repeat_interleave(token_count)
    token_offsets = torch.arange(
        token_count, device=flat_x.device, dtype=index_dtype
    )
    sorted_scattered = torch.cat(
        [token_offsets * shared_count + index for index in range(shared_count)]
    )
    offsets = torch.arange(
        1, shared_count + 1, device=flat_x.device, dtype=index_dtype
    ) * token_count
    cached = (sorted_experts, sorted_scattered, offsets)
    _SHARED_ROUTING_CACHE[key] = cached
    return cached


def _scatter_glu(module, x, expert_p, expert_idxs, experts, routing_metadata=None):
    from scattermoe.mlp import flatten_sort_count
    from scattermoe.parallel_experts import parallel_linear

    num_experts = len(experts)
    top_k = expert_idxs.size(1)
    if routing_metadata is None:
        sorted_experts, sorted_scattered, offsets = flatten_sort_count(
            expert_idxs, num_experts=num_experts
        )
    else:
        sorted_experts, sorted_scattered, offsets = routing_metadata
    # Keep the original independent Parameters canonical. These differentiable
    # stacks let autograd return gradients to the same tensors and therefore
    # preserve both checkpoint keys and AdamW state layout.
    gate_up = torch.stack(
        [
            torch.cat((expert.up_proj.weight, expert.gate_proj.weight), dim=0)
            for expert in experts
        ]
    )
    down = torch.stack([expert.down_proj.weight for expert in experts])
    hidden = parallel_linear(
        x,
        gate_up.permute(0, 2, 1),
        top_k,
        sorted_experts,
        sorted_scattered,
        offsets,
        grouped_out=True,
    )
    up, gate = hidden.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    return parallel_linear(
        hidden,
        down.permute(0, 2, 1),
        1,
        sorted_experts,
        sorted_scattered,
        offsets,
        grouped_in=True,
        gates=expert_p,
    )


def scattermoe_training_forward(
    module,
    flat_x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    routed = _scatter_glu(
        module,
        flat_x,
        weights.to(flat_x.dtype),
        indices,
        module.experts,
    )
    shared_count = len(module.shared_experts)
    if not shared_count:
        return routed
    shared_indices = torch.arange(
        shared_count, device=flat_x.device, dtype=indices.dtype
    ).expand(flat_x.size(0), -1)
    shared_weights = flat_x.new_ones(flat_x.size(0), shared_count)
    shared = _scatter_glu(
        module,
        flat_x,
        shared_weights,
        shared_indices,
        module.shared_experts,
        routing_metadata=_shared_routing_metadata(
            flat_x, shared_count, indices.dtype
        ),
    )
    return routed + shared
