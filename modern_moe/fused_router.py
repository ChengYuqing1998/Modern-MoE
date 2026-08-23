"""Training-only fused router for the fixed small-expert MoE case."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _router_forward_kernel(
    logits_ptr,
    probabilities_ptr,
    weights_ptr,
    indices_ptr,
    logsumexp_ptr,
    tile_counts_ptr,
    T,
    stride_logits_t,
    stride_prob_t,
    stride_weight_t,
    stride_index_t,
    n_tiles,
    E: tl.constexpr,
    E_BLOCK: tl.constexpr,
    K: tl.constexpr,
    TOKENS_PER_TILE: tl.constexpr,
):
    tile = tl.program_id(0)
    rows = tile * TOKENS_PER_TILE + tl.arange(0, TOKENS_PER_TILE)
    experts = tl.arange(0, E_BLOCK)
    valid_rows = rows < T
    valid_experts = experts < E
    logits = tl.load(
        logits_ptr + rows[:, None] * stride_logits_t + experts[None, :],
        mask=valid_rows[:, None] & valid_experts[None, :],
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(logits, axis=1)
    exponentials = tl.exp(logits - row_max[:, None])
    denominator = tl.sum(exponentials, axis=1)
    probabilities = exponentials / denominator[:, None]
    tl.store(
        probabilities_ptr + rows[:, None] * stride_prob_t + experts[None, :],
        probabilities,
        mask=valid_rows[:, None] & valid_experts[None, :],
    )
    tl.store(logsumexp_ptr + rows, row_max + tl.log(denominator), mask=valid_rows)

    # E is only 12 in the target model.  Iterative argmax avoids launching a
    # separate generic top-k kernel and gives deterministic distinct experts.
    idx0 = tl.argmax(logits, axis=1)
    val0 = tl.max(logits, axis=1)
    masked1 = tl.where(experts[None, :] == idx0[:, None], -float("inf"), logits)
    idx1 = tl.argmax(masked1, axis=1)
    val1 = tl.max(masked1, axis=1)
    masked2 = tl.where(
        (experts[None, :] == idx0[:, None]) | (experts[None, :] == idx1[:, None]),
        -float("inf"),
        logits,
    )
    idx2 = tl.argmax(masked2, axis=1)
    val2 = tl.max(masked2, axis=1)

    # The production configuration is Top-3.  Keeping K constexpr makes an
    # accidental incompatible configuration fail at the Python boundary.
    selected_max = tl.maximum(val0, tl.maximum(val1, val2))
    exp0 = tl.exp(val0 - selected_max)
    exp1 = tl.exp(val1 - selected_max)
    exp2 = tl.exp(val2 - selected_max)
    selected_sum = exp0 + exp1 + exp2
    weight0 = exp0 / selected_sum
    weight1 = exp1 / selected_sum
    weight2 = exp2 / selected_sum

    tl.store(indices_ptr + rows * stride_index_t, idx0, mask=valid_rows)
    tl.store(indices_ptr + rows * stride_index_t + 1, idx1, mask=valid_rows)
    tl.store(indices_ptr + rows * stride_index_t + 2, idx2, mask=valid_rows)
    tl.store(weights_ptr + rows * stride_weight_t, weight0, mask=valid_rows)
    tl.store(weights_ptr + rows * stride_weight_t + 1, weight1, mask=valid_rows)
    tl.store(weights_ptr + rows * stride_weight_t + 2, weight2, mask=valid_rows)

    count = (
        tl.sum((idx0[:, None] == experts[None, :]) & valid_rows[:, None], axis=0)
        + tl.sum((idx1[:, None] == experts[None, :]) & valid_rows[:, None], axis=0)
        + tl.sum((idx2[:, None] == experts[None, :]) & valid_rows[:, None], axis=0)
    )
    tl.store(
        tile_counts_ptr + experts * n_tiles + tile,
        count,
        mask=valid_experts,
    )


@triton.jit
def _router_backward_kernel(
    probabilities_ptr,
    weights_ptr,
    indices_ptr,
    grad_probabilities_ptr,
    grad_weights_ptr,
    grad_logsumexp_ptr,
    grad_logits_ptr,
    T,
    stride_prob_t,
    stride_weight_t,
    stride_index_t,
    stride_grad_prob_t,
    stride_grad_weight_t,
    stride_grad_logits_t,
    E: tl.constexpr,
    E_BLOCK: tl.constexpr,
    K: tl.constexpr,
    HAS_GRAD_PROBABILITIES: tl.constexpr,
    HAS_GRAD_WEIGHTS: tl.constexpr,
    HAS_GRAD_LOGSUMEXP: tl.constexpr,
):
    row = tl.program_id(0)
    experts = tl.arange(0, E_BLOCK)
    valid = experts < E
    probabilities = tl.load(
        probabilities_ptr + row * stride_prob_t + experts,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    grad = tl.zeros([E_BLOCK], tl.float32)

    if HAS_GRAD_PROBABILITIES:
        grad_probabilities = tl.load(
            grad_probabilities_ptr + row * stride_grad_prob_t + experts,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(grad_probabilities * probabilities, axis=0)
        grad += probabilities * (grad_probabilities - dot)

    if HAS_GRAD_WEIGHTS:
        slots = tl.arange(0, 4)
        slot_mask = slots < K
        indices = tl.load(
            indices_ptr + row * stride_index_t + slots, mask=slot_mask, other=-1
        )
        weights = tl.load(
            weights_ptr + row * stride_weight_t + slots, mask=slot_mask, other=0.0
        ).to(tl.float32)
        grad_weights = tl.load(
            grad_weights_ptr + row * stride_grad_weight_t + slots,
            mask=slot_mask,
            other=0.0,
        ).to(tl.float32)
        selected_dot = tl.sum(grad_weights * weights, axis=0)
        selected_grad = weights * (grad_weights - selected_dot)
        grad += tl.sum(
            tl.where(experts[:, None] == indices[None, :], selected_grad[None, :], 0.0),
            axis=1,
        )

    if HAS_GRAD_LOGSUMEXP:
        grad_lse = tl.load(grad_logsumexp_ptr + row).to(tl.float32)
        grad += grad_lse * probabilities

    tl.store(grad_logits_ptr + row * stride_grad_logits_t + experts, grad, mask=valid)


class _FusedRouterFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, top_k: int, weight_dtype: torch.dtype):
        if logits.ndim != 2 or logits.dtype != torch.float32:
            raise ValueError("fused router expects a 2-D float32 logits tensor")
        if top_k != 3:
            raise ValueError("fused router currently supports top_k=3 only")
        token_count, expert_count = logits.shape
        if expert_count > 16:
            raise ValueError("fused router currently supports at most 16 experts")
        # Match Liger's K=3 histogram tiling: 1024 / next_power_of_2(K).
        tokens_per_tile = 256
        tile_count = triton.cdiv(token_count, tokens_per_tile)
        probabilities = torch.empty_like(logits)
        weights = torch.empty(
            token_count, top_k, device=logits.device, dtype=weight_dtype
        )
        indices = torch.empty(
            token_count, top_k, device=logits.device, dtype=torch.int32
        )
        logsumexp = torch.empty(token_count, device=logits.device, dtype=torch.float32)
        tile_counts = torch.empty(
            expert_count, tile_count, device=logits.device, dtype=torch.int32
        )
        _router_forward_kernel[(tile_count,)](
            logits,
            probabilities,
            weights,
            indices,
            logsumexp,
            tile_counts,
            token_count,
            logits.stride(0),
            probabilities.stride(0),
            weights.stride(0),
            indices.stride(0),
            tile_count,
            E=expert_count,
            E_BLOCK=triton.next_power_of_2(expert_count),
            K=top_k,
            TOKENS_PER_TILE=tokens_per_tile,
            num_warps=4,
        )
        ctx.save_for_backward(probabilities, weights, indices)
        ctx.expert_count = expert_count
        ctx.mark_non_differentiable(indices, tile_counts)
        ctx.set_materialize_grads(False)
        return probabilities, weights, indices, logsumexp, tile_counts

    @staticmethod
    def backward(
        ctx,
        grad_probabilities,
        grad_weights,
        _grad_indices,
        grad_logsumexp,
        _grad_tile_counts,
    ):
        probabilities, weights, indices = ctx.saved_tensors
        token_count = probabilities.size(0)
        grad_logits = torch.empty_like(probabilities)
        # Triton still requires valid pointer arguments for constexpr-disabled
        # branches, so reuse an existing tensor when an output has no gradient.
        gp = grad_probabilities if grad_probabilities is not None else probabilities
        gw = grad_weights if grad_weights is not None else weights
        gl = grad_logsumexp if grad_logsumexp is not None else probabilities[:, 0]
        _router_backward_kernel[(token_count,)](
            probabilities,
            weights,
            indices,
            gp,
            gw,
            gl,
            grad_logits,
            token_count,
            probabilities.stride(0),
            weights.stride(0),
            indices.stride(0),
            gp.stride(0),
            gw.stride(0),
            grad_logits.stride(0),
            E=ctx.expert_count,
            E_BLOCK=triton.next_power_of_2(ctx.expert_count),
            K=weights.size(1),
            HAS_GRAD_PROBABILITIES=grad_probabilities is not None,
            HAS_GRAD_WEIGHTS=grad_weights is not None,
            HAS_GRAD_LOGSUMEXP=grad_logsumexp is not None,
            num_warps=1,
        )
        return grad_logits, None, None


def fused_router(
    logits: torch.Tensor,
    top_k: int,
    weight_dtype: torch.dtype,
):
    """Return probabilities, Top-k, LSE, histogram, and expert totals."""
    outputs = _FusedRouterFunction.apply(logits, top_k, weight_dtype)
    probabilities, weights, indices, logsumexp, tile_counts = outputs
    # Liger's next metadata stage mutates tile_counts into per-tile prefix
    # sums. Materialize the tiny (E,) totals first so both Liger and aux loss
    # can reuse them without cloning the (E, n_tiles) histogram.
    expert_counts = tile_counts.sum(dim=1, dtype=torch.int32)
    return probabilities, weights, indices, logsumexp, tile_counts, expert_counts
