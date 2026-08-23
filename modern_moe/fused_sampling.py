"""GPU-resident sampling state and a Top-k-first fused filtering path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_logits_kernel(
    logits_ptr,
    seen_ptr,
    output_ptr,
    vocab_size: tl.constexpr,
    penalty: tl.constexpr,
    temperature: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < vocab_size
    values = tl.load(logits_ptr + offsets, mask=mask).to(tl.float32)
    if penalty != 1.0:
        seen = tl.load(seen_ptr + offsets, mask=mask, other=0)
        penalized = tl.where(values < 0, values * penalty, values / penalty)
        values = tl.where(seen != 0, penalized, values)
    values /= temperature
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def _ban_repeated_ngram_kernel(
    history_ptr,
    logits_ptr,
    history_length,
    ngram_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    candidate_count = history_length - ngram_size + 1
    valid = position < candidate_count
    matches = valid
    for prefix_offset in range(ngram_size - 1):
        previous = tl.load(history_ptr + position + prefix_offset, mask=valid, other=-1)
        suffix = tl.load(
            history_ptr + history_length - ngram_size + 1 + prefix_offset
        )
        matches &= previous == suffix
    banned = tl.load(
        history_ptr + position + ngram_size - 1,
        mask=matches,
        other=0,
    )
    # Concurrent stores are benign: every matching position writes -inf.
    tl.store(logits_ptr + banned, -float("inf"), mask=matches)


@triton.jit
def _top_p_sample_kernel(
    values_ptr,
    indices_ptr,
    uniform_ptr,
    token_ptr,
    candidate_count: tl.constexpr,
    requested_top_k: tl.constexpr,
    top_p: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Top-k threshold, nucleus filtering, normalization and sampling."""
    offsets = tl.arange(0, BLOCK)
    valid = offsets < candidate_count
    values = tl.load(values_ptr + offsets, mask=valid, other=-float("inf"))
    threshold = tl.load(values_ptr + requested_top_k - 1)
    valid &= values >= threshold

    maximum = tl.max(tl.where(valid, values, -float("inf")), axis=0)
    weights = tl.where(valid, tl.exp(values - maximum), 0.0)
    first_total = tl.sum(weights, axis=0)
    first_cumulative = tl.cumsum(weights, axis=0) / first_total
    # Reference semantics shift `cumulative > top_p` one position right, so
    # the first token that crosses the nucleus threshold remains eligible.
    previous_cumulative = first_cumulative - weights / first_total
    if top_p > 0.0 and top_p < 1.0:
        keep = valid & ((offsets == 0) | (previous_cumulative <= top_p))
    else:
        keep = valid

    kept_weights = tl.where(keep, weights, 0.0)
    kept_total = tl.sum(kept_weights, axis=0)
    target = tl.load(uniform_ptr) * kept_total
    cumulative = tl.cumsum(kept_weights, axis=0)
    selected_offset = tl.min(
        tl.where(keep & (cumulative >= target), offsets, BLOCK), axis=0
    )
    # torch.rand may theoretically return exactly zero; select the first kept
    # candidate in that case rather than an earlier zero-weight lane.
    selected_offset = tl.where(selected_offset == BLOCK, 0, selected_offset)
    token = tl.load(indices_ptr + selected_offset)
    tl.store(token_ptr, token)


class FusedSampler:
    """Sampling helper specialized for batch=1 and a positive Top-k."""

    def __init__(
        self,
        prompt: torch.Tensor,
        config,
        max_length: int,
        vocab_size: int,
        backend: str = "triton",
    ) -> None:
        if prompt.ndim != 2 or prompt.size(0) != 1 or not prompt.is_cuda:
            raise ValueError("FusedSampler requires one CUDA prompt")
        if config.temperature <= 0 or config.top_k <= 0:
            raise ValueError("FusedSampler requires temperature > 0 and top_k > 0")
        self.config = config
        if backend not in {"triton", "flashinfer"}:
            raise ValueError(f"unknown fused sampling backend: {backend}")
        self.backend = backend
        self.length = int(prompt.size(1))
        self.history = torch.empty(
            max_length, device=prompt.device, dtype=prompt.dtype
        )
        self.history[: self.length].copy_(prompt[0])
        self.seen = torch.zeros(
            vocab_size, device=prompt.device, dtype=torch.uint8
        )
        self.seen.scatter_(0, prompt[0], 1)
        self.uniform = torch.empty((), device=prompt.device, dtype=torch.float32)
        self.token = torch.empty((1, 1), device=prompt.device, dtype=prompt.dtype)

    def candidate_probabilities(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_size = logits.size(-1)
        prepared = torch.empty((vocab_size,), device=logits.device, dtype=torch.float32)
        _prepare_logits_kernel[(triton.cdiv(vocab_size, 256),)](
            logits.reshape(-1),
            self.seen,
            prepared,
            vocab_size=vocab_size,
            penalty=float(self.config.repetition_penalty),
            temperature=float(self.config.temperature),
            BLOCK=256,
            num_warps=4,
        )
        ngram_size = int(self.config.no_repeat_ngram_size)
        if ngram_size >= 2 and self.length >= ngram_size - 1:
            count = self.length - ngram_size + 1
            if count > 0:
                _ban_repeated_ngram_kernel[(triton.cdiv(count, 256),)](
                    self.history,
                    prepared,
                    history_length=self.length,
                    ngram_size=ngram_size,
                    BLOCK=256,
                    num_warps=4,
                )
        if self.backend == "flashinfer":
            import flashinfer

            token = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                prepared[None],
                int(self.config.top_k),
                float(self.config.top_p),
                filter_apply_order="top_k_first",
                deterministic=True,
            ).view(1, 1)
            self.history[self.length : self.length + 1].copy_(token.reshape(-1))
            self.seen.scatter_(
                0, self.history[self.length : self.length + 1], 1
            )
            self.length += 1
            return token
        requested_top_k = min(int(self.config.top_k), vocab_size)
        # BF16 logits frequently tie at the kth value. The reference path
        # masks values *below* that threshold and therefore retains every tie.
        # Keep a bounded surplus so top-p sees the same tied candidates while
        # still avoiding a full-vocabulary sort.
        candidate_count = min(max(256, requested_top_k * 4), vocab_size)
        values, indices = torch.topk(prepared, candidate_count, sorted=True)
        threshold = values[requested_top_k - 1]
        active = values >= threshold
        active_values = values.masked_fill(~active, float("-inf"))
        if 0 < self.config.top_p < 1:
            cumulative = active_values.softmax(dim=-1).cumsum(dim=-1)
            remove = cumulative > self.config.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            active_values.masked_fill_(remove, float("-inf"))
        return indices, active_values.softmax(dim=-1)

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        # Keep preprocessing shared with the validation helper, but avoid its
        # unfused candidate probability construction in the production path.
        vocab_size = logits.size(-1)
        prepared = torch.empty((vocab_size,), device=logits.device, dtype=torch.float32)
        _prepare_logits_kernel[(triton.cdiv(vocab_size, 256),)](
            logits.reshape(-1), self.seen, prepared,
            vocab_size=vocab_size,
            penalty=float(self.config.repetition_penalty),
            temperature=float(self.config.temperature),
            BLOCK=256, num_warps=4,
        )
        ngram_size = int(self.config.no_repeat_ngram_size)
        if ngram_size >= 2 and self.length >= ngram_size - 1:
            count = self.length - ngram_size + 1
            if count > 0:
                _ban_repeated_ngram_kernel[(triton.cdiv(count, 256),)](
                    self.history, prepared, history_length=self.length,
                    ngram_size=ngram_size, BLOCK=256, num_warps=4,
                )
        requested_top_k = min(int(self.config.top_k), vocab_size)
        candidate_count = min(max(256, requested_top_k * 4), vocab_size)
        values, indices = torch.topk(prepared, candidate_count, sorted=True)
        self.uniform.uniform_()
        _top_p_sample_kernel[(1,)](
            values, indices, self.uniform, self.token,
            candidate_count=candidate_count,
            requested_top_k=requested_top_k,
            top_p=float(self.config.top_p),
            BLOCK=triton.next_power_of_2(candidate_count),
            num_warps=8,
        )
        token = self.token
        self.history[self.length : self.length + 1].copy_(token.reshape(-1))
        self.seen.scatter_(0, token.reshape(-1), 1)
        self.length += 1
        return token
