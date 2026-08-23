"""Trainable text-only nanoK3 causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NanoK3Config
from .layers import (
    GatedMLA,
    KDAState,
    KimiDeltaAttention,
    MLACache,
    RMSNorm,
    SiTUMLP,
    StableLatentMoE,
    apply_attention_residual,
)

LayerCache = KDAState | MLACache


@dataclass
class NanoK3Output:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    router_loads: Optional[tuple[torch.Tensor, ...]] = None


@dataclass
class NanoK3InferenceOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    cache: list[LayerCache]


class NanoK3DecoderLayer(nn.Module):
    def __init__(self, config: NanoK3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_type = config.attention_type(layer_idx)
        self.self_attn = (
            KimiDeltaAttention(config)
            if self.attention_type == "kda"
            else GatedMLA(config)
        )
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        if layer_idx < config.first_k_dense_replace:
            self.feed_forward = SiTUMLP(
                config.hidden_size, config.dense_intermediate_size, config
            )
            self.is_moe = False
        else:
            self.feed_forward = StableLatentMoE(config)
            self.is_moe = True

        self.attention_res_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp_res_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention_res_proj = nn.Linear(
            config.hidden_size, 1, bias=False
        )
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states

        if block_residual.shape[1] > 0:
            hidden_states = apply_attention_residual(
                prefix_sum.reshape(-1, hidden_size),
                block_residual,
                self.attention_res_proj,
                self.attention_res_norm,
                self.config.attn_res_checkpoint and self.training,
            ).view(batch, seq_len, hidden_size)

        # The embedding and every completed block become depth-attention values.
        if self.layer_idx % self.config.attn_res_block_size == 0:
            block_residual = torch.cat(
                (
                    block_residual,
                    prefix_sum.reshape(-1, hidden_size).unsqueeze(1),
                ),
                dim=1,
            )
            prefix_sum = None

        attention_output = self.self_attn(
            self.input_norm(hidden_states), attention_mask
        )
        prefix_sum = (
            attention_output
            if prefix_sum is None
            else prefix_sum + attention_output
        )

        hidden_states = apply_attention_residual(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.config.attn_res_checkpoint and self.training,
        ).view(batch, seq_len, hidden_size)
        feed_forward_output = self.feed_forward(
            self.post_attention_norm(hidden_states)
        )
        prefix_sum = prefix_sum + feed_forward_output
        return prefix_sum, block_residual

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        cache: Optional[LayerCache],
        max_cache_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, LayerCache]:
        """AttnRes layer with cached sequence mixing and unchanged depth math."""
        batch, seq_len, hidden_size = hidden_states.shape
        prefix_sum = hidden_states
        if block_residual.shape[1] > 0:
            hidden_states = apply_attention_residual(
                prefix_sum.reshape(-1, hidden_size),
                block_residual,
                self.attention_res_proj,
                self.attention_res_norm,
                False,
            ).view(batch, seq_len, hidden_size)
        if self.layer_idx % self.config.attn_res_block_size == 0:
            block_residual = torch.cat(
                (
                    block_residual,
                    prefix_sum.reshape(-1, hidden_size).unsqueeze(1),
                ),
                dim=1,
            )
            prefix_sum = None
        normalized = self.input_norm(hidden_states)
        if self.attention_type == "kda":
            attention_output, cache = self.self_attn.forward_cached(
                normalized, cache
            )
        else:
            attention_output, cache = self.self_attn.forward_cached(
                normalized, cache, max_cache_length
            )
        prefix_sum = (
            attention_output
            if prefix_sum is None
            else prefix_sum + attention_output
        )
        hidden_states = apply_attention_residual(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            False,
        ).view(batch, seq_len, hidden_size)
        prefix_sum = prefix_sum + self.feed_forward(
            self.post_attention_norm(hidden_states)
        )
        return prefix_sum, block_residual, cache


class NanoK3ForCausalLM(nn.Module):
    def __init__(self, config: NanoK3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            NanoK3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        )
        self.output_attn_res_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.output_attn_res_proj = nn.Linear(
            config.hidden_size, 1, bias=False
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.inference_mode()
    def forward_inference(
        self,
        input_ids: torch.Tensor,
        cache: Optional[list[LayerCache]] = None,
        max_cache_length: Optional[int] = None,
    ) -> NanoK3InferenceOutput:
        """Prefill or decode one/more tokens with KDA and MLA layer caches."""
        if self.training:
            raise RuntimeError("Call model.eval() before forward_inference()")
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must have shape [batch, sequence]")
        max_cache_length = (
            max_cache_length or self.config.max_position_embeddings
        )
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("cache must contain one entry per decoder layer")

        hidden_states = self.embed_tokens(input_ids)
        batch, seq_len, hidden_size = hidden_states.shape
        block_residual = hidden_states.new_zeros(
            batch * seq_len, 0, hidden_size
        )
        updated: list[LayerCache] = []
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states, block_residual, layer_cache = layer.forward_cached(
                hidden_states,
                block_residual,
                layer_cache,
                max_cache_length,
            )
            updated.append(layer_cache)
        hidden_states = apply_attention_residual(
            hidden_states.reshape(-1, hidden_size),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            False,
        ).view(batch, seq_len, hidden_size)
        normalized = self.norm(hidden_states)
        return NanoK3InferenceOutput(
            logits=self.lm_head(normalized),
            hidden_states=hidden_states,
            cache=updated,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> NanoK3Output:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        hidden_states = self.embed_tokens(input_ids)
        batch, seq_len, hidden_size = hidden_states.shape
        block_residual = hidden_states.new_zeros(
            batch * seq_len, 0, hidden_size
        )
        router_loads = []
        for layer in self.layers:
            hidden_states, block_residual = layer(
                hidden_states, block_residual, attention_mask
            )
            if layer.is_moe:
                loads = layer.feed_forward.last_router_loads
                if loads is not None:
                    router_loads.append(loads)

        hidden_states = apply_attention_residual(
            hidden_states.reshape(-1, hidden_size),
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            self.config.attn_res_checkpoint and self.training,
        ).view(batch, seq_len, hidden_size)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().float().view(
                    -1, self.config.vocab_size
                ),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return NanoK3Output(
            logits=logits,
            loss=loss,
            router_loads=tuple(router_loads) if router_loads else None,
        )

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
