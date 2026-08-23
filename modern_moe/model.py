from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModernMoEConfig
from .layers import (
    DenseSwiGLU,
    FullAttentionCache,
    FullCausalAttention,
    KDAState,
    KimiDeltaAttention,
    PackedSeqParams,
    RMSNorm,
    SparseMoE,
)
from .packed_moe import PackedSparseMoE

try:
    from liger_kernel.ops import LigerFusedAddRMSNormFunction
except ImportError:
    LigerFusedAddRMSNormFunction = None

LayerCache = Union[FullAttentionCache, KDAState]


@dataclass
class InferenceOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    cache: list[LayerCache]


@dataclass
class CausalLMOutput:
    logits: Optional[torch.Tensor]
    loss_hidden_states: Optional[torch.Tensor] = None
    classifier_weight: Optional[torch.Tensor] = None
    linear_ce_impl: str = "pytorch"
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    mtp_logits: Optional[tuple[torch.Tensor, ...]] = None
    router_aux_loss: Optional[torch.Tensor] = None
    router_z_loss: Optional[torch.Tensor] = None


class DecoderLayer(nn.Module):
    """Classic residual paths with strict Pre-RMSNorm."""

    def __init__(self, config: ModernMoEConfig, layer_idx: int):
        super().__init__()
        attention_type = config.attention_type(layer_idx)
        self.attention_type = attention_type
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = (
            FullCausalAttention(config)
            if attention_type == "full"
            else KimiDeltaAttention(config)
        )
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.use_moe and layer_idx < config.first_k_dense_replace:
            self.moe = DenseSwiGLU(config, config.dense_intermediate_size)
        else:
            if not config.use_moe:
                self.moe = DenseSwiGLU(config)
            elif config.moe_parameter_layout in {"packed_scattermoe", "packed_liger"}:
                self.moe = PackedSparseMoE(config)
            else:
                self.moe = SparseMoE(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.use_fused_add_rms_norm = config.fused_add_rms_norm

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        compute_router_losses: bool = True,
        packed_seq_params=None,
    ):
        attention_output = self.residual_dropout(
            self.attention(self.input_norm(x), attention_mask, packed_seq_params)
        )
        if (
            self.use_fused_add_rms_norm
            and self.training
            and x.is_cuda
            and LigerFusedAddRMSNormFunction is not None
        ):
            # Fuse the attention residual add with the immediately following
            # pre-MoE RMSNorm.  The returned residual is exactly the value that
            # continues down the residual stream; no parameter/checkpoint key
            # or mathematical structure changes.
            normalized, x = LigerFusedAddRMSNormFunction.apply(
                attention_output,
                x,
                self.post_attention_norm.weight,
                self.post_attention_norm.eps,
                0.0,
                "llama",
                False,
            )
        else:
            x = x + attention_output
            normalized = self.post_attention_norm(x)
        moe_output, aux_loss, z_loss = self.moe(
            normalized,
            compute_router_losses=compute_router_losses,
        )
        x = x + self.residual_dropout(moe_output)
        return x, aux_loss, z_loss

    def forward_cached(
        self,
        x: torch.Tensor,
        cache: Optional[LayerCache],
        max_cache_length: int,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, LayerCache]:
        normalized = self.input_norm(x)
        if self.attention_type == "full":
            attention_output, cache = self.attention.forward_cached(
                normalized,
                cache,
                max_cache_length,
                cache_position=cache_position,
            )
        else:
            attention_output, cache = self.attention.forward_cached(
                normalized,
                cache,
            )
        x = x + attention_output
        moe_output, _, _ = self.moe(
            self.post_attention_norm(x),
            compute_router_losses=False,
        )
        return x + moe_output, cache


class MultiTokenPredictionLayer(nn.Module):
    """Sequential MTP module with shared token embeddings and LM head."""

    def __init__(self, config: ModernMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.token_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fusion = nn.Linear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.decoder = DecoderLayer(config, layer_idx)
        self.output_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        future_token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gradient_checkpointing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.hidden_norm(hidden_states)
        future_token_embeddings = self.token_norm(future_token_embeddings)
        hidden_states = self.fusion(
            torch.cat((hidden_states, future_token_embeddings), dim=-1)
        )
        if gradient_checkpointing and self.training:
            def checkpointed_decoder(states: torch.Tensor):
                return self.decoder(states, attention_mask)

            hidden_states, aux_loss, z_loss = checkpoint(
                checkpointed_decoder,
                hidden_states,
                use_reentrant=False,
            )
        else:
            hidden_states, aux_loss, z_loss = self.decoder(
                hidden_states, attention_mask
            )
        return self.output_norm(hidden_states), aux_loss, z_loss


class ModernMoEForCausalLM(nn.Module):
    def __init__(self, config: ModernMoEConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_layers = nn.ModuleList(
            MultiTokenPredictionLayer(
                config,
                config.num_hidden_layers + layer_idx,
            )
            for layer_idx in range(config.num_mtp_layers)
        )
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    @torch.inference_mode()
    def forward_inference(
        self,
        input_ids: torch.Tensor,
        cache: Optional[list[LayerCache]] = None,
        max_cache_length: Optional[int] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> InferenceOutput:
        """Prefill or incrementally decode while updating each layer's cache."""
        if self.training:
            raise RuntimeError("Call model.eval() before forward_inference()")
        if input_ids.ndim != 2 or input_ids.size(1) < 1:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if max_cache_length is None:
            max_cache_length = self.config.max_position_embeddings
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("cache must have one entry per decoder layer")

        x = self.embed_tokens(input_ids)
        updated: list[LayerCache] = []
        for layer, layer_cache in zip(self.layers, cache):
            x, layer_cache = layer.forward_cached(
                x,
                layer_cache,
                max_cache_length,
                cache_position=cache_position,
            )
            updated.append(layer_cache)
        logits = self.lm_head(self.norm(x))
        return InferenceOutput(logits=logits, hidden_states=x, cache=updated)

    @torch.inference_mode()
    def mtp_draft(
        self,
        hidden_states: torch.Tensor,
        future_token_ids: torch.Tensor,
        cache: Optional[list[LayerCache]] = None,
        max_cache_length: Optional[int] = None,
    ) -> InferenceOutput:
        """Predict one additional token with the sequential MTP module."""
        if len(self.mtp_layers) != 1:
            raise RuntimeError(
                "MTP speculative inference currently requires exactly one "
                "MTP layer"
            )
        if hidden_states.shape[:2] != future_token_ids.shape:
            raise ValueError("hidden_states and future_token_ids must align")
        if max_cache_length is None:
            max_cache_length = self.config.max_position_embeddings
        if cache is None:
            cache = [None]
        if len(cache) != 1:
            raise ValueError("MTP cache must contain exactly one layer state")

        layer = self.mtp_layers[0]
        states = layer.hidden_norm(hidden_states)
        embeddings = layer.token_norm(self.embed_tokens(future_token_ids))
        states = layer.fusion(torch.cat((states, embeddings), dim=-1))
        states, layer_cache = layer.decoder.forward_cached(
            states,
            cache[0],
            max_cache_length,
        )
        states = layer.output_norm(states)
        return InferenceOutput(
            logits=self.lm_head(states),
            hidden_states=states,
            cache=[layer_cache],
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mtp_targets: Optional[torch.Tensor] = None,
        return_mtp_logits: bool = False,
        return_loss_hidden_states: bool = False,
        linear_ce_impl: str = "pytorch",
        packed_seq_params=None,
    ) -> CausalLMOutput:
        x = self.embed_tokens(input_ids)
        aux_losses, z_losses = [], []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # Bind the current layer: checkpoint recomputes this function
                # during backward, after the Python loop has advanced.
                def checkpointed_layer(
                    hidden_states: torch.Tensor,
                    current_layer: DecoderLayer = layer,
                ):
                    return current_layer(
                        hidden_states, attention_mask, True, packed_seq_params
                    )

                x, aux_loss, z_loss = checkpoint(
                    checkpointed_layer,
                    x,
                    use_reentrant=False,
                )
            else:
                x, aux_loss, z_loss = layer(x, attention_mask, True, packed_seq_params)
            if isinstance(layer.moe, (SparseMoE, PackedSparseMoE)):
                aux_losses.append(aux_loss)
                z_losses.append(z_loss)
        mtp_loss = None
        mtp_logits = None
        if self.mtp_layers and (mtp_targets is not None or return_mtp_logits):
            if mtp_targets is None:
                raise ValueError(
                    "mtp_targets are required when return_mtp_logits=True"
                )
            if mtp_targets.shape != input_ids.shape:
                raise ValueError("mtp_targets must have the same shape as input_ids")
            mtp_hidden = x
            mtp_loss_values = []
            mtp_logit_values = []
            for depth, mtp_layer in enumerate(self.mtp_layers):
                usable_length = mtp_targets.size(1) - depth - 1
                if usable_length <= 0:
                    break
                mtp_hidden = mtp_hidden[:, :usable_length]
                future_ids = mtp_targets[:, depth : depth + usable_length]
                future_embeddings = self.embed_tokens(future_ids)
                mtp_hidden, mtp_aux_loss, mtp_z_loss = mtp_layer(
                    mtp_hidden,
                    future_embeddings,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
                if isinstance(
                    mtp_layer.decoder.moe, (SparseMoE, PackedSparseMoE)
                ):
                    aux_losses.append(mtp_aux_loss)
                    z_losses.append(mtp_z_loss)
                prediction_targets = mtp_targets[
                    :, depth + 1 : depth + 1 + usable_length
                ]
                if return_mtp_logits:
                    depth_logits = self.lm_head(mtp_hidden)
                    mtp_logit_values.append(depth_logits)
                    if mtp_targets is not None:
                        mtp_loss_values.append(
                            F.cross_entropy(
                                depth_logits.float().reshape(
                                    -1, self.config.vocab_size
                                ),
                                prediction_targets.reshape(-1),
                            )
                        )
                else:
                    def checkpointed_mtp_loss(
                        states: torch.Tensor,
                        targets: torch.Tensor,
                    ) -> torch.Tensor:
                        depth_logits = self.lm_head(states)
                        return F.cross_entropy(
                            depth_logits.float().reshape(
                                -1, self.config.vocab_size
                            ),
                            targets.reshape(-1),
                        )

                    mtp_loss_values.append(
                        checkpoint(
                            checkpointed_mtp_loss,
                            mtp_hidden,
                            prediction_targets,
                            use_reentrant=False,
                        )
                    )
            if mtp_loss_values:
                mtp_loss = torch.stack(mtp_loss_values).mean()
            if return_mtp_logits:
                mtp_logits = tuple(mtp_logit_values)

        router_aux_loss = (
            torch.stack(aux_losses).mean() if aux_losses else x.new_zeros(())
        )
        router_z_loss = (
            torch.stack(z_losses).mean() if z_losses else x.new_zeros(())
        )

        normalized = self.norm(x)
        # The training-only linear-CE path consumes normalized hidden states
        # and the registered lm_head weight directly, avoiding full logits.
        # Ordinary callers still receive the unchanged logits API.
        logits = None if return_loss_hidden_states else self.lm_head(normalized)
        lm_loss = None
        loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss = (
                lm_loss
                + self.config.router_aux_loss_coef * router_aux_loss
                + self.config.router_z_loss_coef * router_z_loss
            )
            if mtp_loss is not None:
                loss = loss + self.config.mtp_loss_coef * mtp_loss
        return CausalLMOutput(
            logits=logits,
            loss_hidden_states=(normalized if return_loss_hidden_states else None),
            classifier_weight=(self.lm_head.weight if return_loss_hidden_states else None),
            linear_ce_impl=linear_ce_impl,
            loss=loss,
            lm_loss=lm_loss,
            mtp_loss=mtp_loss,
            mtp_logits=mtp_logits,
            router_aux_loss=router_aux_loss,
            router_z_loss=router_z_loss,
        )

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
