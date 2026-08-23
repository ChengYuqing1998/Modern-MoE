from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


@dataclass
class ModernMoEConfig:
    """Configuration for the decoder-only hybrid MoE language model."""

    architecture_name: str = "Modern-MoE"
    vocab_size: int = 64_000
    tokenizer_path: str = "tokenizer/qwen3_moe"
    hidden_size: int = 1_024
    num_hidden_layers: int = 20
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    intermediate_size: int = 1_536
    use_moe: bool = True
    first_k_dense_replace: int = 0
    dense_intermediate_size: int = 0
    num_experts: int = 8
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_parameter_layout: str = "legacy"
    max_position_embeddings: int = 8_192
    rope_theta: float = 500_000.0
    attention_pattern: Tuple[str, ...] = ("linear", "linear", "linear", "full")
    full_attention_backend: str = "eager"
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    fused_add_rms_norm: bool = False
    fused_router: bool = False
    fused_rope: bool = False
    router_aux_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3
    num_mtp_layers: int = 0
    mtp_loss_coef: float = 0.1
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.use_moe and not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if self.moe_parameter_layout not in {
            "legacy", "packed_scattermoe", "packed_liger"
        }:
            raise ValueError(
                "moe_parameter_layout must be 'legacy', 'packed_scattermoe', "
                "or 'packed_liger'"
            )
        if not 0 <= self.first_k_dense_replace <= self.num_hidden_layers:
            raise ValueError(
                "first_k_dense_replace must be in [0, num_hidden_layers]"
            )
        if self.first_k_dense_replace and self.dense_intermediate_size < 1:
            raise ValueError(
                "dense_intermediate_size must be positive when "
                "first_k_dense_replace is non-zero"
            )
        if not self.attention_pattern:
            raise ValueError("attention_pattern cannot be empty")
        if any(kind not in {"full", "linear"} for kind in self.attention_pattern):
            raise ValueError("attention_pattern entries must be 'full' or 'linear'")
        if self.full_attention_backend not in {"eager", "sdpa", "flash_attn"}:
            raise ValueError(
                "full_attention_backend must be 'eager', 'sdpa', or 'flash_attn'"
            )
        if self.num_mtp_layers < 0:
            raise ValueError("num_mtp_layers must be non-negative")
        if self.mtp_loss_coef < 0:
            raise ValueError("mtp_loss_coef must be non-negative")

    def attention_type(self, layer_idx: int) -> str:
        return self.attention_pattern[layer_idx % len(self.attention_pattern)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
