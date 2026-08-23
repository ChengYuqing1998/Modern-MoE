"""Liger fused MoE integration with synchronization-free tile allocation."""

from __future__ import annotations

import os
from contextvars import ContextVar

os.environ.setdefault("LIGER_FUSED_MOE_AUTOTUNE", "1")

import torch
import triton
import liger_kernel.ops.fused_moe as _liger


_PATCHED = False
_ROUTING_TILE_COUNTS = ContextVar("modern_moe_routing_tile_counts", default=None)


def _compute_routing_metadata_no_sync(
    topk_indices: torch.Tensor,
    expert_count: int,
    block_m_token: int = _liger.BLOCK_M_TOKEN,
):
    """Equivalent Liger metadata with a tight upper-bound tile allocation."""
    token_count, top_k = topk_indices.shape
    assignment_count = token_count * top_k
    device = topk_indices.device
    expert_pow2 = triton.next_power_of_2(expert_count)
    top_k_pow2 = triton.next_power_of_2(top_k)
    tokens_per_block = max(1, 1024 // top_k_pow2)
    token_tiles = triton.cdiv(token_count, tokens_per_block)

    cached = _ROUTING_TILE_COUNTS.get()
    if (
        cached is not None
        and cached[0].shape == (expert_count, token_tiles)
        and cached[0].device == device
    ):
        tile_expert_counts, expert_token_count = cached
    else:
        tile_expert_counts = torch.empty(
            expert_count, token_tiles, dtype=torch.int32, device=device
        )
        _liger._moe_router_histogram_kernel[(token_tiles,)](
            topk_indices,
            tile_expert_counts,
            token_count,
            expert_count,
            token_tiles,
            TOKENS_PER_TILE=tokens_per_block,
            K_POW2=top_k_pow2,
            K=top_k,
            E_POW2=expert_pow2,
        )
        expert_token_count = tile_expert_counts.sum(dim=1, dtype=torch.int32)
    expert_start_idx = torch.empty(
        expert_count + 1, dtype=torch.int32, device=device
    )
    expert_tile_offset = torch.empty_like(expert_start_idx)
    _liger._moe_router_prefix_sum_kernel[(expert_count + 2,)](
        expert_token_count,
        expert_start_idx,
        expert_tile_offset,
        E=expert_count,
        partial_sum_ptr=tile_expert_counts,
        n_tiles=token_tiles,
        TK=assignment_count,
        BLOCK_M=128,
        BLOCK_N=expert_pow2,
        BLOCK_M_TOKEN=block_m_token,
    )

    # sum(ceil(count[e] / block)) <= ceil(total / block) + E - 1.
    max_m_tiles = triton.cdiv(assignment_count, block_m_token) + expert_count - 1
    tile_row_start = torch.full(
        (max_m_tiles,), assignment_count, dtype=torch.int32, device=device
    )
    tile_expert = torch.full(
        (max_m_tiles,), expert_count - 1, dtype=torch.int32, device=device
    )
    scatter_idx = torch.empty(assignment_count, dtype=torch.int32, device=device)
    reverse_scatter_idx = torch.empty_like(scatter_idx)
    gather_idx = torch.empty_like(scatter_idx)
    if assignment_count:
        _liger._moe_router_scatter_kernel[(token_tiles,)](
            scatter_idx,
            reverse_scatter_idx,
            gather_idx,
            tile_row_start,
            tile_expert,
            topk_indices,
            token_count,
            tile_expert_counts,
            token_tiles,
            expert_start_idx[:expert_count],
            expert_tile_offset[:expert_count],
            K_POW2=top_k_pow2,
            K=top_k,
            TOKENS_PER_BLOCK=tokens_per_block,
            BLOCK_M_TOKEN=block_m_token,
        )
    return (
        expert_token_count,
        expert_start_idx,
        gather_idx,
        scatter_idx,
        reverse_scatter_idx,
        tile_row_start,
        tile_expert,
    )


def liger_fused_moe(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    tile_expert_counts: torch.Tensor | None = None,
    expert_token_count: torch.Tensor | None = None,
) -> torch.Tensor:
    global _PATCHED
    if not _PATCHED:
        _liger.compute_routing_metadata = _compute_routing_metadata_no_sync
        _PATCHED = True
    cached_metadata = (
        (tile_expert_counts, expert_token_count)
        if tile_expert_counts is not None and expert_token_count is not None
        else None
    )
    token = _ROUTING_TILE_COUNTS.set(cached_metadata)
    try:
        return _liger.LigerFusedMoEFunction.apply(
            x, gate_up_weight, down_weight, indices.to(torch.int32), weights
        )
    finally:
        _ROUTING_TILE_COUNTS.reset(token)


def dense_shared_swiglu(
    x: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Always-on shared experts with no routing, sorting, or scattering."""
    from liger_kernel.ops import LigerSiLUMulFunction

    expert_count = up_gate_weight.size(0)
    inputs = x.unsqueeze(0).expand(expert_count, -1, -1)
    up_gate = torch.bmm(inputs, up_gate_weight.transpose(1, 2))
    up, gate = up_gate.chunk(2, dim=-1)
    hidden = LigerSiLUMulFunction.apply(gate, up)
    return torch.bmm(hidden, down_weight.transpose(1, 2)).sum(dim=0).to(x.dtype)
