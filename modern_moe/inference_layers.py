"""Inference-only kernels for Modern-MoE.

These functions operate on non-persistent expert-major weight caches owned by
``SparseMoE``.  They are never used by the training forward path.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def build_expert_cache(module) -> None:
    if module._inference_gate_up is not None:
        return
    if hasattr(module, "routed"):
        if getattr(module, "parameter_layout", "packed_scattermoe") == "packed_liger":
            routed_gate, routed_up = module.routed.experts.weight.chunk(2, dim=1)
        else:
            routed_up, routed_gate = module.routed.experts.weight.chunk(2, dim=1)
        shared_up, shared_gate = module.shared.experts.weight.chunk(2, dim=1)
        module._inference_gate_up = torch.cat(
            (
                torch.cat((routed_gate, routed_up), dim=1),
                torch.cat((shared_gate, shared_up), dim=1),
            ),
            dim=0,
        ).detach()
        module._inference_down = torch.cat(
            (
                module.routed.output_experts.weight,
                module.shared.output_experts.weight,
            ),
            dim=0,
        ).detach()
    else:
        all_experts = list(module.experts) + list(module.shared_experts)
        module._inference_gate_up = torch.stack(
            [
                torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0)
                for expert in all_experts
            ],
            dim=0,
        ).detach()
        module._inference_down = torch.stack(
            [expert.down_proj.weight for expert in all_experts], dim=0
        ).detach()


def _shared_count(module) -> int:
    return (
        module.num_shared_experts
        if hasattr(module, "num_shared_experts")
        else len(module.shared_experts)
    )


def batched_swiglu(
    inputs: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    gate_up = torch.bmm(inputs, gate_up_weight.transpose(1, 2))
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    return torch.bmm(hidden, down_weight.transpose(1, 2))


def selected_expert_forward(
    module,
    flat_x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Decode a tiny token batch using only Top-k and shared experts."""
    build_expert_cache(module)
    token_count = flat_x.size(0)
    shared_count = _shared_count(module)
    if shared_count:
        shared_indices = torch.arange(
            module.num_experts,
            module.num_experts + shared_count,
            device=indices.device,
        ).expand(token_count, -1)
        selected = torch.cat((indices, shared_indices), dim=1)
        coefficients = torch.cat(
            (
                weights.to(flat_x.dtype),
                torch.ones(
                    token_count,
                    shared_count,
                    dtype=flat_x.dtype,
                    device=flat_x.device,
                ),
            ),
            dim=1,
        )
    else:
        selected = indices
        coefficients = weights.to(flat_x.dtype)

    return preselected_expert_forward(module, flat_x, selected, coefficients)


def preselected_expert_forward(
    module,
    flat_x: torch.Tensor,
    selected: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Execute already-routed experts without rebuilding shared metadata."""
    build_expert_cache(module)
    token_count = flat_x.size(0)
    if (
        token_count == 1
        and flat_x.is_cuda
        and os.getenv("MODERN_MOE_USE_VLLM_FUSED_EXPERTS", "0") == "1"
    ):
        from .vllm_fused_experts import fused_selected_experts

        return fused_selected_experts(
            flat_x,
            module._inference_gate_up,
            module._inference_down,
            selected.reshape(-1),
            coefficients.reshape(-1),
        )
    expert_count = selected.size(1)
    gate_up_weight = module._inference_gate_up[selected].flatten(0, 1)
    down_weight = module._inference_down[selected].flatten(0, 1)
    inputs = (
        flat_x[:, None, :]
        .expand(-1, expert_count, -1)
        .reshape(token_count * expert_count, 1, flat_x.size(1))
    )
    outputs = batched_swiglu(inputs, gate_up_weight, down_weight)
    outputs = outputs.view(token_count, expert_count, flat_x.size(1))
    return (outputs * coefficients[:, :, None]).sum(dim=1)


def padded_prefill_forward(
    module,
    flat_x: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Prefill with padded expert-major BMMs instead of Python expert loops."""
    build_expert_cache(module)
    token_count, hidden_size = flat_x.shape
    assignment_tokens = torch.arange(
        token_count, device=flat_x.device
    ).repeat_interleave(module.top_k)
    assignment_experts = indices.reshape(-1)
    order = torch.argsort(assignment_experts, stable=True)
    sorted_experts = assignment_experts.index_select(0, order)
    sorted_tokens = assignment_tokens.index_select(0, order)
    sorted_weights = weights.reshape(-1).index_select(0, order)
    counts = torch.bincount(sorted_experts, minlength=module.num_experts)
    max_count = int(counts.max().item())

    grouped = flat_x.index_select(0, sorted_tokens)
    padded = flat_x.new_zeros(module.num_experts, max_count, hidden_size)
    valid = torch.zeros(
        module.num_experts, max_count, dtype=torch.bool, device=flat_x.device
    )
    cursor = 0
    for expert_idx in range(module.num_experts):
        count = int(counts[expert_idx].item())
        if count:
            padded[expert_idx, :count] = grouped[cursor : cursor + count]
            valid[expert_idx, :count] = True
            cursor += count
    expert_outputs = batched_swiglu(
        padded,
        module._inference_gate_up[: module.num_experts],
        module._inference_down[: module.num_experts],
    )
    valid_outputs = expert_outputs[valid]
    routed = torch.zeros_like(flat_x)
    routed.index_add_(
        0,
        sorted_tokens,
        (valid_outputs * sorted_weights[:, None].to(valid_outputs.dtype)).to(
            routed.dtype
        ),
    )

    shared_count = _shared_count(module)
    if not shared_count:
        return routed
    shared_inputs = flat_x.unsqueeze(0).expand(shared_count, -1, -1)
    shared = batched_swiglu(
        shared_inputs,
        module._inference_gate_up[module.num_experts :],
        module._inference_down[module.num_experts :],
    ).sum(dim=0)
    return routed + shared.to(routed.dtype)
