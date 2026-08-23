import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModernMoEConfig
from .inference_layers import padded_prefill_forward, selected_expert_forward

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except ImportError:
    chunk_kda = None
    fused_recurrent_kda = None

@dataclass
class FullAttentionCache:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0


@dataclass
class KDAState:
    recurrent_state: Optional[torch.Tensor] = None


@dataclass
class PackedSeqParams:
    """Optional variable-length (padding-free) packing metadata.

    When provided to ``ModernMoEForCausalLM.forward``, the model runs in the
    flat, padding-free layout: ``input_ids`` is ``[N]`` (all sequences
    concatenated, no padding) and ``cu_seqlens`` marks the boundaries so
    FlashAttention's varlen kernel keeps every sequence's causal attention
    confined to itself.  Position for RoPE is likewise reset per sequence.

    This is entirely additive: ``None`` (or not passed) keeps the original
    dense/fixed-length behaviour, so SFT / TOP-D callers are unaffected.
    """

    cu_seqlens: torch.Tensor     # [num_seqs + 1] cumulative boundaries on N
    max_seqlen: int              # longest sequence length in the pack
    # Optional per-token positions for RoPE (reset per sequence).  If None it is
    # derived from cu_seqlens, but providing it avoids a scatter/gather.
    positions: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_training_cos", None, persistent=False)
        self.register_buffer("_training_sin", None, persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        angles = torch.outer(positions.float(), self.inv_freq.float())
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def training_values(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = self._training_cos
        if (
            cached is None
            or cached.size(0) != sequence_length
            or cached.device != device
            or cached.dtype != dtype
        ):
            positions = torch.arange(sequence_length, device=device)
            cos, sin = self(positions, dtype)
            self._training_cos = cos.detach()
            self._training_sin = sin.detach()
        return self._training_cos, self._training_sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.cat((cos, cos), dim=-1)[None, None, :, :]
    sin = torch.cat((sin, sin), dim=-1)[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


def _apply_rope_flat(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE for the flat ``[N, heads, dim]`` (varlen) layout.

    ``cos``/``sin`` have one row per token (length N); position is the leading
    axis here, so we align with ``[N, 1, 2*dim]`` against ``[N, heads, dim]``.
    """
    cos = torch.cat((cos, cos), dim=-1)[:, None, :]
    sin = torch.cat((sin, sin), dim=-1)[:, None, :]
    return x * cos + _rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=1)


class FullCausalAttention(nn.Module):
    """Causal GQA with selectable eager, PyTorch SDPA, or FlashAttention backend."""

    def __init__(self, config: ModernMoEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.backend = config.full_attention_backend
        self.dropout = config.attention_dropout
        self.use_fused_rope = config.fused_rope
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> torch.Tensor:
        if packed_seq_params is not None:
            return self._forward_varlen(x, packed_seq_params, attention_mask)
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_fused_rope and self.training and x.is_cuda:
            from liger_kernel.ops import LigerRopeFunction

            cos, sin = self.rope.training_values(seq_len, x.device, q.dtype)
            q, k = LigerRopeFunction.apply(
                q, k, cos.unsqueeze(0), sin.unsqueeze(0), None, 1
            )
        else:
            positions = torch.arange(seq_len, device=x.device)
            cos, sin = self.rope(positions, q.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if attention_mask is not None and attention_mask.shape != (batch, seq_len):
            raise ValueError("attention_mask must have shape [batch, sequence]")

        dropout_p = self.dropout if self.training else 0.0
        if self.backend == "eager":
            output = self._eager_attention(
                q, k, v, attention_mask, seq_len, dropout_p
            )
        elif self.backend == "sdpa":
            output = self._sdpa_attention(
                q, k, v, attention_mask, seq_len, dropout_p
            )
        elif self.backend == "flash_attn":
            output = self._flash_attention(
                q, k, v, attention_mask, dropout_p
            )
        else:
            raise RuntimeError(f"Unknown full-attention backend: {self.backend}")

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)

    def _forward_varlen(
        self,
        x: torch.Tensor,
        packed: PackedSeqParams,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Padding-free concatenated forward via FlashAttention varlen.

        ``x`` is flat ``[N, hidden]`` (all sequences concatenated).  Q/K/V are
        ``[N, heads, dim]``; RoPE uses per-token positions reset at each
        sequence boundary; attention is run with ``flash_attn_varlen_func`` and
        ``cu_seqlens`` so no sequence sees another.  Returns flat ``[N, hidden]``.
        """
        if attention_mask is not None:
            raise ValueError("packed_seq_params (varlen) does not accept attention_mask")
        N, _ = x.shape
        cu = packed.cu_seqlens
        max_seqlen = packed.max_seqlen

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_kv_heads, self.head_dim)

        if packed.positions is not None:
            positions = packed.positions
        else:
            positions = self._derive_positions(cu, N, x.device)
        cos, sin = self.rope(positions, q.dtype)
        q, k = _apply_rope_flat(q, cos, sin), _apply_rope_flat(k, cos, sin)

        from flash_attn_interface import flash_attn_varlen_func

        cu = cu.to(torch.int32)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu,
            max_seqlen,
            max_seqlen,
            causal=True,
            softmax_scale=None,
        )  # [N, heads, dim]
        return self.o_proj(out.view(N, -1))

    @staticmethod
    def _derive_positions(cu: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        """Per-token RoPE position, reset to 0 at each sequence boundary."""
        seq_lens = cu[1:] - cu[:-1]
        positions = torch.cat(
            [torch.arange(int(l), device=device) for l in seq_lens.tolist()]
        )
        return positions.contiguous()

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: Optional[FullAttentionCache],
        max_cache_length: int,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, FullAttentionCache]:
        """Incremental attention with a preallocated native-GQA KV cache."""
        batch, seq_len, _ = x.shape
        if cache is None:
            key = torch.empty(
                batch,
                max_cache_length,
                self.num_kv_heads,
                self.head_dim,
                dtype=x.dtype,
                device=x.device,
            )
            value = torch.empty_like(key)
            cache = FullAttentionCache(key=key, value=value)
        if cache.key.size(0) != batch:
            raise ValueError("KV cache batch size does not match input")
        if cache_position is None and cache.length + seq_len > cache.key.size(1):
            raise ValueError(
                f"KV cache capacity {cache.key.size(1)} exceeded by "
                f"{cache.length + seq_len} tokens"
            )

        shape_q = (batch, seq_len, self.num_heads, self.head_dim)
        shape_kv = (batch, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_proj(x).view(shape_q)
        k = self.k_proj(x).view(shape_kv)
        v = self.v_proj(x).view(shape_kv)
        if cache_position is None:
            positions = torch.arange(
                cache.length,
                cache.length + seq_len,
                device=x.device,
            )
            attention_cache_length: int | torch.Tensor = cache.length
        else:
            if seq_len != 1 or cache_position.shape != (batch,):
                raise ValueError(
                    "cache_position must have shape [batch] for single-token decode"
                )
            positions = cache_position.to(torch.long)
            attention_cache_length = cache_position
        cos, sin = self.rope(positions, q.dtype)
        q_heads = q.transpose(1, 2)
        k_heads = k.transpose(1, 2)
        q = apply_rope(q_heads, cos, sin).transpose(1, 2)
        k = apply_rope(k_heads, cos, sin).transpose(1, 2)

        if self.backend == "flash_attn":
            try:
                from flash_attn_interface import flash_attn_with_kvcache
            except ImportError as error:
                raise RuntimeError(
                    "FlashAttention 3 KV-cache inference requires "
                    "flash_attn_interface.flash_attn_with_kvcache"
                ) from error
            output = flash_attn_with_kvcache(
                q,
                cache.key,
                cache.value,
                k=k,
                v=v,
                cache_seqlens=attention_cache_length,
                causal=True,
            )
        else:
            start = cache.length
            stop = start + seq_len
            cache.key[:, start:stop].copy_(k)
            cache.value[:, start:stop].copy_(v)
            all_k = cache.key[:, :stop].transpose(1, 2)
            all_v = cache.value[:, :stop].transpose(1, 2)
            q_heads = q.transpose(1, 2)
            all_k = repeat_kv(all_k, self.kv_groups)
            all_v = repeat_kv(all_v, self.kv_groups)
            query_positions = torch.arange(start, stop, device=x.device)
            key_positions = torch.arange(stop, device=x.device)
            causal = key_positions[None, :] <= query_positions[:, None]
            output = F.scaled_dot_product_attention(
                q_heads,
                all_k,
                all_v,
                attn_mask=causal[None, None],
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        if cache_position is None:
            cache.length += seq_len
        output = output.contiguous().view(batch, seq_len, -1)
        return self.o_proj(output), cache

    def _eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        dropout_p: float,
    ) -> torch.Tensor:
        """Project-owned reference implementation with explicit score matrix."""
        k, v = repeat_kv(k, self.kv_groups), repeat_kv(v, self.kv_groups)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=q.device
        ).tril()
        allowed = causal[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        weights = F.dropout(weights, p=dropout_p, training=self.training)
        return torch.matmul(weights, v)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        dropout_p: float,
    ) -> torch.Tensor:
        """PyTorch dispatcher; it selects an available fused CUDA kernel."""
        if attention_mask is None:
            mask = None
            is_causal = True
        else:
            causal = torch.ones(
                seq_len, seq_len, dtype=torch.bool, device=q.device
            ).tril()
            mask = causal[None, None, :, :] & attention_mask[
                :, None, None, :
            ].bool()
            is_causal = False
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            enable_gqa=self.kv_groups > 1,
        )

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout_p: float,
    ) -> torch.Tensor:
        """Dao-AILab FlashAttention, preserving native GQA head counts."""
        if attention_mask is not None:
            raise ValueError(
                "flash_attn backend currently requires packed samples without "
                "an attention_mask"
            )
        try:
            # FlashAttention 3 wheels expose this top-level interface.
            from flash_attn_interface import flash_attn_func
            flash_version = 3
        except ImportError:
            try:
                # FlashAttention 2 uses the flash_attn package namespace.
                from flash_attn import flash_attn_func
                flash_version = 2
            except ImportError as error:
                raise RuntimeError(
                    "full_attention_backend='flash_attn' was selected, but "
                    "neither FlashAttention 3 (`flash_attn_interface`) nor "
                    "FlashAttention 2 (`flash_attn`) is importable."
                ) from error

        # flash_attn_func consumes [batch, sequence, heads, head_dim].
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if flash_version == 3:
            if dropout_p:
                raise ValueError(
                    "The installed FlashAttention 3 interface does not expose "
                    "attention dropout; set attention_dropout to 0."
                )
            output = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=None,
                causal=True,
            )
        else:
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                softmax_scale=None,
                causal=True,
            )
        return output.transpose(1, 2)


class KimiDeltaAttention(nn.Module):
    """KDA linear attention using FLA's parallel chunk training kernel.

    The projections deliberately retain the original model parameterization;
    only the sequential Python recurrence is replaced. ``g`` is the log-space
    channel-wise decay expected by the official KDA operator.
    """

    def __init__(self, config: ModernMoEConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        projection_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.beta_proj = nn.Linear(config.hidden_size, self.num_heads, bias=True)
        self.decay_proj = nn.Linear(config.hidden_size, projection_size, bias=True)
        self.output_gate = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.head_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if chunk_kda is None:
            raise RuntimeError(
                "KDA training requires the official FLA chunk kernel. "
                "Install the project requirements so `fla-core` is available."
            )
        if not x.is_cuda:
            raise RuntimeError("The FLA KDA chunk kernel requires a CUDA tensor.")

        batch, seq_len, _ = x.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape)
        k = self.k_proj(x).view(shape)
        v = self.v_proj(x).view(shape)
        beta = self.beta_proj(x).sigmoid().float()
        log_decay = -F.softplus(self.decay_proj(x).view(shape).float())

        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            valid = attention_mask[:, :, None, None].bool()
            q = q.masked_fill(~valid, 0)
            k = k.masked_fill(~valid, 0)
            v = v.masked_fill(~valid, 0)
            log_decay = log_decay.masked_fill(~valid, 0)
            beta = beta.masked_fill(~attention_mask[:, :, None].bool(), 0)

        output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=log_decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        output = output.to(x.dtype)
        output = self.head_norm(output)
        gate = F.silu(self.output_gate(x).view(shape))
        output = (output * gate).reshape(batch, seq_len, -1)
        return self.o_proj(output)

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: Optional[KDAState],
    ) -> tuple[torch.Tensor, KDAState]:
        """Prefill/decode using FLA's persistent KDA recurrent state."""
        if chunk_kda is None or fused_recurrent_kda is None:
            raise RuntimeError("FLA KDA inference kernels are unavailable")
        if not x.is_cuda:
            raise RuntimeError("FLA KDA inference requires a CUDA tensor")
        cache = cache or KDAState()
        batch, seq_len, _ = x.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape)
        k = self.k_proj(x).view(shape)
        v = self.v_proj(x).view(shape)
        beta = self.beta_proj(x).sigmoid().float()
        log_decay = -F.softplus(self.decay_proj(x).view(shape).float())
        operator = chunk_kda if seq_len > 1 else fused_recurrent_kda
        output, final_state = operator(
            q=q,
            k=k,
            v=v,
            g=log_decay,
            beta=beta,
            initial_state=cache.recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        cache.recurrent_state = final_state
        output = self.head_norm(output.to(x.dtype))
        gate = F.silu(self.output_gate(x).view(shape))
        output = (output * gate).reshape(batch, seq_len, -1)
        return self.o_proj(output), cache


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DenseSwiGLU(nn.Module):
    """Dense SwiGLU FFN with the same output contract as SparseMoE."""

    def __init__(
        self,
        config: ModernMoEConfig,
        intermediate_size: int | None = None,
    ):
        super().__init__()
        self.ffn = SwiGLUExpert(
            config.hidden_size,
            config.intermediate_size if intermediate_size is None else intermediate_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        compute_router_losses: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del compute_router_losses
        output = self.ffn(x)
        zero = output.new_zeros(())
        return output, zero, zero


class SparseMoE(nn.Module):
    """Top-k routed experts plus always-on shared experts."""

    def __init__(self, config: ModernMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_experts)
        )
        self.shared_experts = nn.ModuleList(
            SwiGLUExpert(config.hidden_size, config.intermediate_size)
            for _ in range(config.num_shared_experts)
        )
        self.use_inference_fast_path = (
            os.getenv("MODERN_MOE_USE_INFERENCE_FAST_PATH", "1") == "1"
        )
        self.use_scattermoe_training = False
        self.register_buffer("_inference_gate_up", None, persistent=False)
        self.register_buffer("_inference_down", None, persistent=False)

    def clear_inference_cache(self) -> None:
        """Discard detached expert-weight snapshots used by eval fast paths."""
        self._inference_gate_up = None
        self._inference_down = None

    def train(self, mode: bool = True):
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # The eval cache contains detached copies of expert parameters.  It is
        # non-persistent and therefore absent from state_dict, so invalidate it
        # explicitly before new checkpoint weights are copied into the module.
        self.clear_inference_cache()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        compute_router_losses: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        flat_x = x.reshape(-1, x.size(-1))
        logits = self.router(flat_x).float()
        if (
            self.use_inference_fast_path
            and not self.training
            and flat_x.size(0) <= 4
            and os.getenv("MODERN_MOE_USE_FUSED_INFERENCE_ROUTER", "0") == "1"
        ):
            from .inference_layers import preselected_expert_forward
            from .inference_router import (
                fused_inference_route,
                fused_inference_route_supported,
            )

            shared_count = len(self.shared_experts)
            if fused_inference_route_supported(logits, self.top_k, shared_count):
                selected, coefficients = fused_inference_route(
                    logits, self.top_k, shared_count, flat_x.dtype
                )
                output = preselected_expert_forward(
                    self, flat_x, selected, coefficients
                )
                zero = logits.new_zeros(())
                return output.view(original_shape), zero, zero
        probabilities = F.softmax(logits, dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        if self.use_inference_fast_path and not self.training:
            if flat_x.size(0) <= 4:
                output = selected_expert_forward(self, flat_x, indices, weights)
            else:
                output = padded_prefill_forward(self, flat_x, indices, weights)
            zero = logits.new_zeros(())
            return output.view(original_shape), zero, zero

        if self.training and self.use_scattermoe_training:
            from .scattermoe_layers import scattermoe_training_forward

            output = scattermoe_training_forward(self, flat_x, indices, weights)
            if compute_router_losses:
                assignment = (
                    F.one_hot(indices, self.num_experts).float().sum(dim=1)
                    / self.top_k
                )
                load = assignment.mean(dim=0)
                importance = probabilities.mean(dim=0)
                aux_loss = self.num_experts * torch.sum(load * importance)
                z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
            else:
                aux_loss = logits.new_zeros(())
                z_loss = logits.new_zeros(())
            return output.view(original_shape), aux_loss, z_loss

        routed = torch.zeros_like(flat_x)

        for expert_idx, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(indices == expert_idx)
            if token_idx.numel():
                expert_out = expert(flat_x.index_select(0, token_idx))
                routed.index_add_(
                    0,
                    token_idx,
                    expert_out
                    * weights[token_idx, slot_idx, None].to(expert_out.dtype),
                )

        shared = sum(
            (expert(flat_x) for expert in self.shared_experts),
            torch.zeros_like(flat_x),
        )
        if compute_router_losses:
            assignment = (
                F.one_hot(indices, self.num_experts).float().sum(dim=1)
                / self.top_k
            )
            load = assignment.mean(dim=0)
            importance = probabilities.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(load * importance)
            z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
        else:
            aux_loss = logits.new_zeros(())
            z_loss = logits.new_zeros(())
        return (routed + shared).view(original_shape), aux_loss, z_loss
