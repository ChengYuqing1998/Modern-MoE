"""Packed-parameter ScatterMoE implementation for training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scattermoe.mlp import GLUMLP
from scattermoe.parallel_experts import parallel_linear

from .config import ModernMoEConfig
from .inference_layers import padded_prefill_forward, selected_expert_forward


class PackedSparseMoE(nn.Module):
    """Top-k and shared experts stored directly in ScatterMoE layout."""

    def __init__(self, config: ModernMoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared_experts = config.num_shared_experts
        self.parameter_layout = config.moe_parameter_layout
        self.use_fused_router = config.fused_router
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.routed = GLUMLP(
            config.hidden_size,
            config.intermediate_size,
            config.num_experts,
            config.num_experts_per_tok,
            bias=False,
        )
        self.shared = GLUMLP(
            config.hidden_size,
            config.intermediate_size,
            config.num_shared_experts,
            config.num_shared_experts,
            bias=False,
        )
        self.use_inference_fast_path = True
        self.register_buffer("_inference_gate_up", None, persistent=False)
        self.register_buffer("_inference_down", None, persistent=False)

    def clear_inference_cache(self) -> None:
        self._inference_gate_up = None
        self._inference_down = None

    def train(self, mode: bool = True):
        if mode:
            self.clear_inference_cache()
        return super().train(mode)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        self.clear_inference_cache()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, x, compute_router_losses: bool = True):
        shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        logits = self.router(flat).float()
        tile_expert_counts = None
        expert_token_count = None
        router_logsumexp = None
        if (
            self.use_fused_router
            and self.training
            and self.parameter_layout == "packed_liger"
            and logits.is_cuda
        ):
            from .fused_router import fused_router

            (
                probabilities,
                routed_weights,
                indices,
                router_logsumexp,
                tile_expert_counts,
                expert_token_count,
            ) = fused_router(logits, self.top_k, flat.dtype)
            weights = routed_weights
        else:
            probabilities = F.softmax(logits, dim=-1)
            weights, indices = probabilities.topk(self.top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        if not self.training and self.use_inference_fast_path:
            if flat.size(0) <= 4:
                output = selected_expert_forward(self, flat, indices, weights)
            else:
                output = padded_prefill_forward(self, flat, indices, weights)
            zero = logits.new_zeros(())
            return output.view(shape), zero, zero
        if self.parameter_layout == "packed_liger":
            from .liger_moe import liger_fused_moe

            routed = liger_fused_moe(
                flat,
                self.routed.experts.weight,
                self.routed.output_experts.weight,
                indices,
                weights.to(flat.dtype),
                tile_expert_counts,
                expert_token_count,
            )
        else:
            routed = self.routed(flat, weights.to(flat.dtype), indices)

        if self.parameter_layout == "packed_liger":
            from .liger_moe import dense_shared_swiglu

            shared = dense_shared_swiglu(
                flat,
                self.shared.experts.weight,
                self.shared.output_experts.weight,
            )
        else:
            from .scattermoe_layers import _shared_routing_metadata

            shared_weights = flat.new_ones(flat.size(0), self.num_shared_experts)
            sorted_experts, sorted_scattered, offsets = _shared_routing_metadata(
                flat, self.num_shared_experts, indices.dtype
            )
            hidden = parallel_linear(
                flat,
                self.shared.experts.weight.permute(0, 2, 1),
                self.num_shared_experts,
                sorted_experts,
                sorted_scattered,
                offsets,
                grouped_out=True,
            )
            up, gate = hidden.chunk(2, dim=-1)
            hidden = F.silu(gate) * up
            shared = parallel_linear(
                hidden,
                self.shared.output_experts.weight.permute(0, 2, 1),
                1,
                sorted_experts,
                sorted_scattered,
                offsets,
                grouped_in=True,
                gates=shared_weights,
            )
        if compute_router_losses:
            if tile_expert_counts is not None:
                # Histogram counts are exactly sum_t,k 1[index[t,k] == e].
                # Reuse them instead of rebuilding a large one-hot tensor.
                load = expert_token_count.to(torch.float32) / (
                    flat.size(0) * self.top_k
                )
            else:
                assignment = (
                    F.one_hot(indices, self.num_experts).float().sum(dim=1)
                    / self.top_k
                )
                load = assignment.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(
                load * probabilities.mean(dim=0)
            )
            if router_logsumexp is None:
                router_logsumexp = torch.logsumexp(logits, dim=-1)
            z_loss = torch.mean(router_logsumexp.square())
        else:
            aux_loss = logits.new_zeros(())
            z_loss = logits.new_zeros(())
        return (routed + shared).view(shape), aux_loss, z_loss
