"""Minimal BF16 decode-only adaptation of vLLM's Triton fused MoE kernel.

The expert GEMM mapping is adapted from vLLM (Apache-2.0), commit
63a9a5010a6d1539c52957646ef9d6bbcf7a4deb.  This deliberately omits vLLM's
quantization, expert-parallel, bias, and general-batch routing machinery.  It
is specialized for one-token decode with unique selected expert IDs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_expert_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    expert_ids_ptr,
    weights_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ae: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ce: tl.constexpr,
    APPLY_WEIGHT: tl.constexpr,
    SELECTED_A: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """vLLM naive-assignment GEMM specialized to M=1 per selected expert."""
    expert_slot = tl.program_id(0)
    block_n = tl.program_id(1)
    expert = tl.load(expert_ids_ptr + expert_slot).to(tl.int64)
    offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((16, BLOCK_N), dtype=tl.float32)

    # tl.dot requires an M tile. Only row zero contains the decode token; the
    # other rows are masked zeros, matching vLLM's naive block assignment.
    row = tl.arange(0, 16)
    a_base = expert_slot * stride_ae if SELECTED_A else 0
    a_ptrs = a_ptr + a_base + offs_k[None, :]
    b_ptrs = (
        b_ptr
        + expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    for k_start in range(0, K, BLOCK_K):
        a = tl.load(
            a_ptrs,
            mask=(row[:, None] == 0) & (offs_k[None, :] + k_start < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] + k_start < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * stride_bk

    if APPLY_WEIGHT:
        accumulator *= tl.load(weights_ptr + expert_slot)
    tl.store(
        c_ptr + expert_slot * stride_ce + row[:, None] * stride_ce + offs_n[None, :],
        accumulator,
        mask=(row[:, None] == 0) & (offs_n[None, :] < N),
    )


@triton.jit
def _swiglu_kernel(gate_up_ptr, hidden_ptr, I: tl.constexpr, BLOCK: tl.constexpr):
    expert_slot = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = expert_slot * (2 * I)
    gate = tl.load(gate_up_ptr + base + offs, mask=offs < I, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + base + I + offs, mask=offs < I, other=0.0).to(tl.float32)
    silu = gate * tl.sigmoid(gate)
    tl.store(hidden_ptr + expert_slot * I + offs, silu * up, mask=offs < I)


@triton.jit
def _sum_experts_kernel(outputs_ptr, result_ptr, E: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for expert_slot in range(E):
        total += tl.load(outputs_ptr + expert_slot * H + offs, mask=offs < H, other=0.0)
    tl.store(result_ptr + offs, total, mask=offs < H)


def fused_selected_experts(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    expert_ids: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    """Run one decode token through selected routed and shared experts."""
    if x.shape[0] != 1:
        raise ValueError("vLLM fused decode path currently requires exactly one token")
    slots = expert_ids.numel()
    hidden_size = x.shape[1]
    intermediate_size = down_weight.shape[2]
    gate_up = torch.empty((slots, 2 * intermediate_size), device=x.device, dtype=x.dtype)
    hidden = torch.empty((slots, intermediate_size), device=x.device, dtype=x.dtype)
    outputs = torch.empty((slots, hidden_size), device=x.device, dtype=x.dtype)
    result = torch.empty_like(x)

    _selected_expert_gemm[(slots, triton.cdiv(2 * intermediate_size, 64))](
        x, gate_up_weight, gate_up, expert_ids, coefficients,
        N=2 * intermediate_size, K=hidden_size, stride_ae=0,
        stride_be=gate_up_weight.stride(0), stride_bk=gate_up_weight.stride(2),
        stride_bn=gate_up_weight.stride(1), stride_ce=gate_up.stride(0),
        APPLY_WEIGHT=False, SELECTED_A=False,
        BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4,
    )
    _swiglu_kernel[(slots,)](
        gate_up, hidden, I=intermediate_size,
        BLOCK=triton.next_power_of_2(intermediate_size), num_warps=4,
    )
    _selected_expert_gemm[(slots, triton.cdiv(hidden_size, 64))](
        hidden, down_weight, outputs, expert_ids, coefficients,
        N=hidden_size, K=intermediate_size, stride_ae=hidden.stride(0),
        stride_be=down_weight.stride(0), stride_bk=down_weight.stride(2),
        stride_bn=down_weight.stride(1), stride_ce=outputs.stride(0),
        APPLY_WEIGHT=True, SELECTED_A=True,
        BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=4,
    )
    _sum_experts_kernel[(1,)](
        outputs, result, E=slots, H=hidden_size,
        BLOCK=triton.next_power_of_2(hidden_size), num_warps=4,
    )
    return result
