"""Parameterized inference-only fused MoE routing.

The router projection remains a regular Linear. This module fuses FP32
softmax-equivalent Top-k selection, selected-weight renormalization, and
shared-expert metadata construction for tiny decode batches.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_inference_router_kernel(
    logits_ptr,
    selected_ptr,
    coefficients_ptr,
    stride_logits_t,
    stride_selected_t,
    stride_coeff_t,
    T: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    SHARED: tl.constexpr,
    E_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    ranks = tl.arange(0, E_BLOCK)
    valid = ranks < E
    logits = tl.load(
        logits_ptr + row * stride_logits_t + ranks,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    # K is a compile-time parameter. Each model shape gets its own cached
    # specialization without hard-coding K=3 in the source.
    remaining = logits
    selected_max = tl.max(logits, axis=0)
    selected_sum = 0.0
    for slot in tl.static_range(0, K):
        expert = tl.argmax(remaining, axis=0)
        value = tl.max(remaining, axis=0)
        unnormalized = tl.exp(value - selected_max)
        selected_sum += unnormalized
        tl.store(selected_ptr + row * stride_selected_t + slot, expert)
        tl.store(
            coefficients_ptr + row * stride_coeff_t + slot, unnormalized
        )
        remaining = tl.where(ranks == expert, -float("inf"), remaining)

    # The same program reads back its raw selected weights after all stores.
    tl.debug_barrier()
    for slot in tl.static_range(0, K):
        raw = tl.load(coefficients_ptr + row * stride_coeff_t + slot)
        tl.store(
            coefficients_ptr + row * stride_coeff_t + slot,
            raw / selected_sum,
        )

    shared_slots = ranks - K
    is_shared = (shared_slots >= 0) & (shared_slots < SHARED)
    tl.store(
        selected_ptr + row * stride_selected_t + ranks,
        E + shared_slots,
        mask=is_shared,
    )
    tl.store(
        coefficients_ptr + row * stride_coeff_t + ranks,
        1.0,
        mask=is_shared,
    )


def fused_inference_route(
    logits: torch.Tensor,
    top_k: int,
    shared_count: int,
    weight_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return routed+shared expert IDs and coefficients for decode."""
    if not fused_inference_route_supported(logits, top_k, shared_count):
        raise ValueError("unsupported fused inference router shape")
    token_count, expert_count = logits.shape
    width = top_k + shared_count
    selected = torch.empty(
        token_count, width, device=logits.device, dtype=torch.int32
    )
    coefficients = torch.empty(
        token_count, width, device=logits.device, dtype=weight_dtype
    )
    expert_block = triton.next_power_of_2(expert_count)
    _fused_inference_router_kernel[(token_count,)](
        logits,
        selected,
        coefficients,
        logits.stride(0),
        selected.stride(0),
        coefficients.stride(0),
        T=token_count,
        E=expert_count,
        K=top_k,
        SHARED=shared_count,
        E_BLOCK=expert_block,
        num_warps=4 if expert_block >= 32 else 1,
    )
    return selected, coefficients


def fused_inference_route_supported(
    logits: torch.Tensor, top_k: int, shared_count: int
) -> bool:
    return (
        logits.is_cuda
        and logits.ndim == 2
        and logits.dtype == torch.float32
        and 1 <= logits.size(0) <= 4
        and 1 <= top_k <= min(8, logits.size(1))
        and 0 <= shared_count <= 8
        and top_k + shared_count <= triton.next_power_of_2(logits.size(1))
        and logits.size(1) <= 128
    )
