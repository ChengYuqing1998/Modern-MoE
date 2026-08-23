"""Configuration for the text-only nanoK3 language model.

Architecture adapted from Moonshot AI's Kimi K3 reference implementation and
technical report. Copyright (c) 2026 Moonshot AI; used under the Kimi K3
License. See ``nanok3/KIMI_K3_LICENSE``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class NanoK3Config:
    model_type: str = "nanoK3"
    vocab_size: int = 151_936
    tokenizer_path: str = "tokenizer/qwen3_moe"
    hidden_size: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # K3 is NoPE. KDA provides position sensitivity.
    max_position_embeddings: int = 8_192
    attention_pattern: tuple[str, ...] = ("kda", "kda", "kda", "mla")
    final_layer_is_mla: bool = True
    attention_dropout: float = 0.0

    # Gated MLA.
    q_lora_rank: int = 128
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64  # Name retained for K3 compatibility; no RoPE is applied.
    v_head_dim: int = 128
    mla_output_gate: bool = True
    mla_backend: str = "sdpa"

    # KDA.
    kda_head_dim: int = 128
    kda_num_heads: int = 4
    kda_short_conv_kernel_size: int = 4
    kda_gate_lower_bound: float = -5.0
    kda_full_rank_gate: bool = True

    # Stable LatentMoE.
    first_k_dense_replace: int = 1
    dense_intermediate_size: int = 2_048
    latent_moe_dim: int = 384
    moe_intermediate_size: int = 256
    num_experts: int = 36
    num_experts_per_token: int = 2
    num_shared_experts: int = 1
    routed_scaling_factor: float = 1.0
    moe_renormalize: bool = True
    quantile_balancing: bool = True
    quantile_ema: float = 0.0

    # SiTU-GLU.
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0

    # Block Attention Residuals.
    attn_res_block_size: int = 4
    attn_res_checkpoint: bool = True

    def __post_init__(self) -> None:
        positive = (
            "vocab_size", "hidden_size", "num_hidden_layers",
            "num_attention_heads", "q_lora_rank", "kv_lora_rank",
            "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim",
            "kda_head_dim", "kda_num_heads", "dense_intermediate_size",
            "latent_moe_dim", "moe_intermediate_size", "num_experts",
            "num_experts_per_token", "attn_res_block_size",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.num_experts_per_token <= self.num_experts:
            raise ValueError("num_experts_per_token must be in [1, num_experts]")
        if not self.attention_pattern:
            raise ValueError("attention_pattern cannot be empty")
        if any(kind not in {"kda", "mla"} for kind in self.attention_pattern):
            raise ValueError("attention_pattern entries must be 'kda' or 'mla'")
        if self.kda_num_heads * self.kda_head_dim <= 0:
            raise ValueError("invalid KDA projection size")
        if self.mla_backend not in {"eager", "sdpa", "flash_attention_3"}:
            raise ValueError(
                "mla_backend must be eager, sdpa, or flash_attention_3"
            )
        if not 0.0 <= self.quantile_ema < 1.0:
            raise ValueError("quantile_ema must be in [0, 1)")

    def attention_type(self, layer_idx: int) -> str:
        if self.final_layer_is_mla and layer_idx == self.num_hidden_layers - 1:
            return "mla"
        return self.attention_pattern[layer_idx % len(self.attention_pattern)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
