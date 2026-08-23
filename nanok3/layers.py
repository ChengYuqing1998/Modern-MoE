"""K3 architecture layers for nanoK3.

The equations and parameterization follow Moonshot AI's Kimi K3 technical
report and Hugging Face reference code. Copyright (c) 2026 Moonshot AI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import NanoK3Config

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except ImportError:  # Keep config/parameter inspection available without FLA.
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_kda = None
    fused_recurrent_kda = None

try:
    from flash_attn_interface import (
        flash_attn_func as flash_attn_3_func,
        flash_attn_with_kvcache as flash_attn_3_with_kvcache,
    )
except ImportError:
    flash_attn_3_func = None
    flash_attn_3_with_kvcache = None


@dataclass
class MLACache:
    key: torch.Tensor
    value: torch.Tensor
    length: int = 0


@dataclass
class KDAState:
    recurrent_state: Optional[torch.Tensor] = None
    q_conv_state: Optional[torch.Tensor] = None
    k_conv_state: Optional[torch.Tensor] = None
    v_conv_state: Optional[torch.Tensor] = None


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype) * self.weight


class SiTUAndMul(nn.Module):
    """K3's bounded Sigmoid Tanh Unit GLU activation."""

    def __init__(self, beta: float = 4.0, linear_beta: float = 25.0):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        dtype = gate.dtype
        gate = gate.float()
        up = up.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (gate * up).to(dtype)


class SiTUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, config: NanoK3Config):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = SiTUAndMul(config.situ_beta, config.situ_linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.activation(self.gate_proj(x), self.up_proj(x))
        )


def _causal_mask(
    batch: int,
    query_length: int,
    key_length: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if attention_mask is None and query_length == key_length:
        return None
    offset = key_length - query_length
    q = torch.arange(query_length, device=device)[:, None] + offset
    k = torch.arange(key_length, device=device)[None, :]
    allowed = k <= q
    mask = torch.zeros(
        (batch, 1, query_length, key_length), device=device, dtype=dtype
    )
    mask.masked_fill_(~allowed[None, None], torch.finfo(dtype).min)
    if attention_mask is not None:
        if attention_mask.shape != (batch, key_length):
            raise ValueError("attention_mask must have shape [batch, key_length]")
        mask.masked_fill_(
            ~attention_mask[:, None, None, :].bool(),
            torch.finfo(dtype).min,
        )
    return mask


class GatedMLA(nn.Module):
    """NoPE Multi-head Latent Attention with K3's full-rank output gate."""

    def __init__(self, config: NanoK3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.scaling = self.q_head_dim**-0.5
        self.q_a_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=False
        )
        self.q_a_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.q_head_dim,
            bias=False,
        )
        self.kv_a_proj = nn.Linear(
            config.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_norm = RMSNorm(config.kv_lora_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
        )
        projection_size = self.num_heads * config.v_head_dim
        self.g_proj = (
            nn.Linear(config.hidden_size, projection_size, bias=False)
            if config.mla_output_gate
            else None
        )
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_position_channel = torch.split(
            q,
            [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim],
            dim=-1,
        )

        compressed = self.kv_a_proj(x)
        kv_latent, shared_position_channel = torch.split(
            compressed,
            [self.config.kv_lora_rank, self.config.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_norm(kv_latent))
        kv = kv.view(
            batch,
            seq_len,
            self.num_heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim,
        ).transpose(1, 2)
        k_nope, value = torch.split(
            kv,
            [self.config.qk_nope_head_dim, self.config.v_head_dim],
            dim=-1,
        )
        # K3 intentionally applies NoPE. The compatibility-named "rope" channel
        # is shared across heads but is never rotated.
        k_position_channel = shared_position_channel[:, None].expand(
            -1, self.num_heads, -1, -1
        )
        query = torch.cat((q_nope, q_position_channel), dim=-1)
        key = torch.cat((k_nope, k_position_channel), dim=-1)
        mask = _causal_mask(
            batch, seq_len, seq_len, x.device, query.dtype, attention_mask
        )

        if self.config.mla_backend == "flash_attention_3":
            if flash_attn_3_func is None:
                raise RuntimeError(
                    "mla_backend=flash_attention_3 requires FlashAttention 3"
                )
            if attention_mask is not None and not attention_mask.bool().all():
                raise NotImplementedError(
                    "FlashAttention 3 MLA currently expects packed, unpadded samples"
                )
            # FA3 consumes [batch, sequence, heads, dimension]. As in the
            # official K3 FA2 path, pad V when its head dimension differs from
            # Q/K, then discard the padding after attention.
            query_fa = query.transpose(1, 2)
            key_fa = key.transpose(1, 2)
            value_fa = value.transpose(1, 2)
            value_padding = self.q_head_dim - self.config.v_head_dim
            if value_padding < 0:
                raise ValueError("Flash MLA requires v_head_dim <= q_head_dim")
            if value_padding:
                value_fa = F.pad(value_fa, (0, value_padding))
            output = flash_attn_3_func(
                query_fa,
                key_fa,
                value_fa,
                softmax_scale=self.scaling,
                causal=True,
            )
            if value_padding:
                output = output[..., : self.config.v_head_dim]
            output = output.transpose(1, 2)
        elif self.config.mla_backend == "sdpa":
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=mask is None,
                scale=self.scaling,
            )
        else:
            scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
            if mask is not None:
                scores = scores + mask
            else:
                causal = torch.ones(
                    seq_len, seq_len, device=x.device, dtype=torch.bool
                ).tril()
                scores = scores.masked_fill(
                    ~causal, torch.finfo(scores.dtype).min
                )
            probabilities = scores.float().softmax(dim=-1).to(query.dtype)
            output = torch.matmul(probabilities, value)

        # Official K3 keeps the attention accumulation output in FP32 during
        # training. The projection returns to the module parameter dtype.
        if self.training:
            output = output.float()
        output = output.transpose(1, 2).reshape(
            batch, seq_len, self.num_heads * self.config.v_head_dim
        )
        if self.g_proj is not None:
            output = output * self.g_proj(x).float().sigmoid()
        return self.o_proj(output.to(self.o_proj.weight.dtype)).to(x.dtype)

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: Optional[MLACache],
        max_cache_length: int,
    ) -> tuple[torch.Tensor, MLACache]:
        """FlashAttention 3 prefill/decode with a preallocated MLA KV cache."""
        if flash_attn_3_with_kvcache is None:
            raise RuntimeError(
                "nanoK3 MLA cache requires flash_attn_with_kvcache from FA3"
            )
        batch, seq_len, _ = x.shape
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(x)))
        q = q.view(batch, seq_len, self.num_heads, self.q_head_dim)
        q_nope, q_shared = torch.split(
            q,
            [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim],
            dim=-1,
        )
        compressed = self.kv_a_proj(x)
        kv_latent, k_shared = torch.split(
            compressed,
            [self.config.kv_lora_rank, self.config.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_norm(kv_latent))
        kv = kv.view(
            batch,
            seq_len,
            self.num_heads,
            self.config.qk_nope_head_dim + self.config.v_head_dim,
        )
        k_nope, value = torch.split(
            kv,
            [self.config.qk_nope_head_dim, self.config.v_head_dim],
            dim=-1,
        )
        key = torch.cat(
            (k_nope, k_shared[:, :, None].expand(-1, -1, self.num_heads, -1)),
            dim=-1,
        )
        query = torch.cat((q_nope, q_shared), dim=-1)
        value_padding = self.q_head_dim - self.config.v_head_dim
        if value_padding:
            value = F.pad(value, (0, value_padding))

        if cache is None:
            cache = MLACache(
                key=torch.empty(
                    batch,
                    max_cache_length,
                    self.num_heads,
                    self.q_head_dim,
                    device=x.device,
                    dtype=x.dtype,
                ),
                value=torch.empty(
                    batch,
                    max_cache_length,
                    self.num_heads,
                    self.q_head_dim,
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
        if cache.length + seq_len > cache.key.size(1):
            raise ValueError("MLA KV cache capacity exceeded")
        output = flash_attn_3_with_kvcache(
            query,
            cache.key,
            cache.value,
            k=key,
            v=value,
            cache_seqlens=cache.length,
            softmax_scale=self.scaling,
            causal=True,
        )
        cache.length += seq_len
        if value_padding:
            output = output[..., : self.config.v_head_dim]
        output = output.reshape(
            batch, seq_len, self.num_heads * self.config.v_head_dim
        )
        if self.g_proj is not None:
            output = output.float() * self.g_proj(x).float().sigmoid()
        return self.o_proj(output.to(self.o_proj.weight.dtype)).to(x.dtype), cache


class KimiDeltaAttention(nn.Module):
    """K3 KDA using the official FLA chunk-kernel/autograd interface."""

    def __init__(self, config: NanoK3Config):
        super().__init__()
        self.config = config
        projection_size = config.kda_num_heads * config.kda_head_dim
        self.q_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        if ShortConvolution is None:
            self.q_conv1d = self.k_conv1d = self.v_conv1d = None
        else:
            kwargs = dict(
                hidden_size=projection_size,
                kernel_size=config.kda_short_conv_kernel_size,
                activation="silu",
            )
            self.q_conv1d = ShortConvolution(**kwargs)
            self.k_conv1d = ShortConvolution(**kwargs)
            self.v_conv1d = ShortConvolution(**kwargs)
        # K3 report §2.1.1 initializes the log-scale A_h to zero. The released
        # inference class uses a placeholder initializer because checkpoints
        # immediately overwrite it.
        self.A_log = nn.Parameter(torch.zeros(config.kda_num_heads))
        self.f_a_proj = nn.Linear(
            config.hidden_size, config.kda_head_dim, bias=False
        )
        self.f_b_proj = nn.Linear(
            config.kda_head_dim, projection_size, bias=False
        )
        self.dt_bias = nn.Parameter(torch.zeros(projection_size))
        self.beta_proj = nn.Linear(
            config.hidden_size, config.kda_num_heads, bias=False
        )
        self.g_proj = nn.Linear(config.hidden_size, projection_size, bias=False)
        self.o_norm = (
            FusedRMSNormGated(
                config.kda_head_dim,
                eps=config.rms_norm_eps,
                activation="sigmoid",
            )
            if FusedRMSNormGated is not None
            else None
        )
        self.o_proj = nn.Linear(projection_size, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if chunk_kda is None or self.q_conv1d is None or self.o_norm is None:
            raise RuntimeError("nanoK3 KDA requires fla-core")
        if not x.is_cuda:
            raise RuntimeError("FLA chunk_kda training requires a CUDA tensor")
        batch, seq_len, _ = x.shape
        if attention_mask is not None:
            if attention_mask.shape != (batch, seq_len):
                raise ValueError("attention_mask must have shape [batch, sequence]")
            if not attention_mask.bool().all():
                raise NotImplementedError(
                    "KDA padding requires official-style unpadding and cu_seqlens; "
                    "nanoK3 currently expects packed, unpadded training samples"
                )
        h, d = self.config.kda_num_heads, self.config.kda_head_dim
        # FLA returns ``(output, final_state)`` even when no convolution state
        # is requested. The official K3 implementation unpacks both values.
        q, _ = self.q_conv1d(
            x=self.q_proj(x),
            output_final_state=False,
        )
        k, _ = self.k_conv1d(
            x=self.k_proj(x),
            output_final_state=False,
        )
        v, _ = self.v_conv1d(
            x=self.v_proj(x),
            output_final_state=False,
        )
        q = q.view(batch, seq_len, h, d)
        k = k.view(batch, seq_len, h, d)
        v = v.view(batch, seq_len, h, d)
        decay_logits = self.f_b_proj(self.f_a_proj(x)).view(batch, seq_len, h, d)
        beta = self.beta_proj(x).float()
        output, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=decay_logits,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=self.config.kda_gate_lower_bound,
            state_v_first=True,
        )
        gate = self.g_proj(x).view(batch, seq_len, h, d)
        output = self.o_norm(output, gate)
        return self.o_proj(output.reshape(batch, seq_len, h * d))

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: Optional[KDAState],
    ) -> tuple[torch.Tensor, KDAState]:
        """KDA prefill/decode with persistent recurrent and ShortConv states."""
        if chunk_kda is None or fused_recurrent_kda is None:
            raise RuntimeError("nanoK3 KDA inference kernels are unavailable")
        if not x.is_cuda:
            raise RuntimeError("nanoK3 KDA cached inference requires CUDA")
        cache = cache or KDAState()
        batch, seq_len, _ = x.shape
        h, d = self.config.kda_num_heads, self.config.kda_head_dim
        q, cache.q_conv_state = self.q_conv1d(
            x=self.q_proj(x),
            cache=cache.q_conv_state,
            output_final_state=True,
        )
        k, cache.k_conv_state = self.k_conv1d(
            x=self.k_proj(x),
            cache=cache.k_conv_state,
            output_final_state=True,
        )
        v, cache.v_conv_state = self.v_conv1d(
            x=self.v_proj(x),
            cache=cache.v_conv_state,
            output_final_state=True,
        )
        q = q.view(batch, seq_len, h, d)
        k = k.view(batch, seq_len, h, d)
        v = v.view(batch, seq_len, h, d)
        decay_logits = self.f_b_proj(self.f_a_proj(x)).view(batch, seq_len, h, d)
        beta = self.beta_proj(x).float()
        common = dict(
            q=q,
            k=k,
            v=v,
            g=decay_logits,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            initial_state=cache.recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            lower_bound=self.config.kda_gate_lower_bound,
            state_v_first=True,
        )
        if seq_len == 1:
            output, cache.recurrent_state = fused_recurrent_kda(**common)
        else:
            output, cache.recurrent_state = chunk_kda(
                **common,
                safe_gate=True,
            )
        gate = self.g_proj(x).view(batch, seq_len, h, d)
        output = self.o_norm(output, gate)
        return self.o_proj(output.reshape(batch, seq_len, h * d)), cache


@dataclass
class RouterState:
    indices: torch.Tensor
    weights: torch.Tensor
    loads: torch.Tensor


class QuantileBalancedRouter(nn.Module):
    """K3 sigmoid Top-k router with auxiliary-loss-free Quantile Balancing."""

    def __init__(self, config: NanoK3Config):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size)
        )
        self.register_buffer(
            "expert_bias", torch.zeros(config.num_experts), persistent=True
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.no_grad()
    def _quantile_balance(
        self,
        scores: torch.Tensor,
        biased_scores: torch.Tensor,
    ) -> None:
        k = self.config.num_experts_per_token
        cutoff = biased_scores.topk(k + 1, dim=-1).values[:, -1]
        margins = scores - cutoff[:, None]
        quantile_level = 1.0 - k / self.config.num_experts
        proposal = -torch.quantile(
            margins.float(), quantile_level, dim=0
        )
        proposal -= proposal.mean()
        if self.config.quantile_ema:
            self.expert_bias.mul_(self.config.quantile_ema).add_(
                proposal * (1.0 - self.config.quantile_ema)
            )
        else:
            self.expert_bias.copy_(proposal)

    def forward(self, x: torch.Tensor) -> RouterState:
        flat = x.reshape(-1, x.shape[-1])
        scores = F.linear(flat.float(), self.weight.float()).sigmoid()
        biased_scores = scores + self.expert_bias.float()
        indices = biased_scores.topk(
            self.config.num_experts_per_token, dim=-1, sorted=False
        ).indices
        weights = scores.gather(1, indices)
        if self.config.moe_renormalize:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.config.routed_scaling_factor
        loads = F.one_hot(
            indices, self.config.num_experts
        ).sum(dim=(0, 1))
        if (
            self.training
            and self.config.quantile_balancing
            and scores.shape[0] > 1
            and self.config.num_experts_per_token < self.config.num_experts
        ):
            self._quantile_balance(scores.detach(), biased_scores.detach())
        return RouterState(indices=indices, weights=weights, loads=loads)


class StableLatentMoE(nn.Module):
    """Normalized LatentMoE with routed and full-width shared experts."""

    def __init__(self, config: NanoK3Config):
        super().__init__()
        self.config = config
        self.down_project = nn.Linear(
            config.hidden_size, config.latent_moe_dim, bias=False
        )
        self.experts = nn.ModuleList(
            SiTUMLP(config.latent_moe_dim, config.moe_intermediate_size, config)
            for _ in range(config.num_experts)
        )
        self.router = QuantileBalancedRouter(config)
        self.routed_norm = RMSNorm(config.latent_moe_dim, config.rms_norm_eps)
        self.up_project = nn.Linear(
            config.latent_moe_dim, config.hidden_size, bias=False
        )
        # The official reference represents N shared experts as one GLU whose
        # intermediate dimension is N times the per-expert width.
        self.shared_experts = SiTUMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.num_shared_experts,
            config,
        )
        self.last_router_loads: Optional[torch.Tensor] = None
        self._stacked_expert_weights: Optional[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

    def _single_token_routed(
        self,
        latent: torch.Tensor,
        router: RouterState,
    ) -> torch.Tensor:
        """Run only selected experts using three batched GEMMs and no CPU sync."""
        if self._stacked_expert_weights is None:
            self._stacked_expert_weights = (
                torch.stack([expert.gate_proj.weight for expert in self.experts]),
                torch.stack([expert.up_proj.weight for expert in self.experts]),
                torch.stack([expert.down_proj.weight for expert in self.experts]),
            )
        gate_all, up_all, down_all = self._stacked_expert_weights
        expert_ids = router.indices[0]
        gate_weight = gate_all.index_select(0, expert_ids)
        up_weight = up_all.index_select(0, expert_ids)
        down_weight = down_all.index_select(0, expert_ids)
        selected_input = latent.expand(expert_ids.numel(), -1).unsqueeze(-1)
        gate = torch.bmm(gate_weight, selected_input).squeeze(-1)
        up = torch.bmm(up_weight, selected_input).squeeze(-1)
        activated = self.experts[0].activation(gate, up).unsqueeze(-1)
        outputs = torch.bmm(down_weight, activated).squeeze(-1)
        return (
            outputs
            * router.weights[0, :, None].to(outputs.dtype)
        ).sum(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        original_shape = x.shape
        router = self.router(x)
        latent = self.down_project(x).reshape(-1, self.config.latent_moe_dim)
        if not self.training and latent.shape[0] == 1:
            routed = self._single_token_routed(latent, router)
        else:
            routed = torch.zeros_like(latent)
            for expert_index, expert in enumerate(self.experts):
                token_index, slot_index = torch.where(
                    router.indices == expert_index
                )
                if token_index.numel() == 0:
                    continue
                expert_output = expert(latent.index_select(0, token_index))
                contribution = expert_output * router.weights[
                    token_index, slot_index, None
                ].to(expert_output.dtype)
                routed = routed.index_add(0, token_index, contribution)
        routed = routed.view(*original_shape[:-1], self.config.latent_moe_dim)
        routed = self.up_project(self.routed_norm(routed))
        self.last_router_loads = router.loads.detach()
        return self.shared_experts(identity) + routed


def apply_attention_residual(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: RMSNorm,
    use_checkpoint: bool,
) -> torch.Tensor:
    """Eq. 9/10 Block AttnRes over prior block states and current prefix."""

    def operation(prefix: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        values = torch.cat((blocks, prefix.unsqueeze(1)), dim=1)
        values_float = values.float()
        normalized = values_float * torch.rsqrt(
            values_float.square().mean(-1, keepdim=True) + norm.eps
        )
        score_weight = norm.weight.float() * projection.weight.squeeze(0).float()
        scores = (normalized * score_weight).sum(dim=-1)
        probabilities = scores.softmax(dim=-1).unsqueeze(1)
        return torch.matmul(probabilities, values_float).squeeze(1).to(values.dtype)

    if use_checkpoint and torch.is_grad_enabled() and prefix_sum.requires_grad:
        return checkpoint(operation, prefix_sum, block_residual, use_reentrant=False)
    return operation(prefix_sum, block_residual)
