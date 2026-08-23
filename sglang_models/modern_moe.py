"""SGLang rollout implementation for the native nanoGPTMoE-v2 checkpoint."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sglang.srt.models.qwen2_moe import Qwen2MoeMLP
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from sglang.srt.utils import add_prefix


class ModernMoESharedMLP(Qwen2MoeMLP):
    """Native shared-expert math over SGLang-compatible parameter names.

    The rollout model currently runs with tensor parallel size one.  Calling
    the two weights explicitly avoids applying a second packed/sharded-weight
    interpretation after the native ``[up, gate] -> [gate, up]`` conversion.
    """

    def forward(self, hidden_states, *args, return_intermediates=False, **kwargs):
        gate_up = F.linear(hidden_states, self.gate_up_proj.weight)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate.float()).to(gate.dtype) * up
        output = F.linear(activated, self.down_proj.weight)
        if return_intermediates:
            return output, gate_up, activated
        return output


class ModernMoESparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """Qwen3-style routed MoE plus the two always-on native shared experts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs.get("config") or args[1]
        prefix = kwargs.get("prefix", "")
        self.shared_experts = nn.ModuleList(
            ModernMoESharedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=kwargs.get("quant_config"),
                prefix=f"{prefix}.shared_experts.{expert_id}",
            )
            for expert_id in range(config.num_shared_experts)
        )

    def forward(self, hidden_states, *args, **kwargs):
        trace_enabled = bool(__import__("os").environ.get("MODERN_MOE_FORWARD_TRACE"))
        shared_outputs = []
        for expert_id, expert in enumerate(self.shared_experts):
            if trace_enabled:
                expert_output, gate_up, activated = expert(
                    hidden_states, return_intermediates=True
                )
                self.__dict__[f"_trace_shared_{expert_id}_gate_up"] = (
                    gate_up.detach().float().cpu()
                )
                self.__dict__[f"_trace_shared_{expert_id}_activated"] = (
                    activated.detach().float().cpu()
                )
            else:
                expert_output = expert(hidden_states)
            shared_outputs.append(expert_output)
            if trace_enabled:
                self.__dict__[f"_trace_shared_{expert_id}"] = (
                    expert_output.detach().float().cpu()
                )
        shared = sum(shared_outputs)
        # SGLang's FusedMoE may reuse/overwrite its input buffer.  Preserve the
        # normalized hidden states for both the shared experts and the decoder
        # residual/KV decode path by routing a private copy.
        routed = super().forward(hidden_states.clone(), *args, **kwargs)
        if trace_enabled:
            self._trace_routed = routed.detach().float().cpu()
            self._trace_shared = shared.detach().float().cpu()
        return routed + shared


class ModernMoEAttention(Qwen3MoeAttention):
    """Native Modern-MoE attention: GQA RoPE without Qwen3 QK-Norm.

    The native checkpoint has q/k projection weights but no q_norm/k_norm
    parameters.  Qwen3MoeAttention applies those norms unconditionally, so
    using it directly changes the first layer's attention even though weight
    loading succeeds.  Keep the SGLang QKV/cache implementation and override
    only the normalization step to match ``modern_moe.layers.FullCausalAttention``.
    """

    def apply_qk_norm_rope(self, qkv, positions, forward_batch):
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        # The parent implementation may write K/V into the cache as part of
        # fused RoPE.  This override intentionally uses plain RoPE, so force
        # ``forward_core`` to save K/V instead of claiming that fused RoPE has
        # already done it.  Without this, prefill is correct but decode reads
        # an empty cache and attention returns zeros.
        self._used_fused_qk_norm_rope_last_call = True
        return q, k, v


class ModernMoEDenseMLP(Qwen2MoeMLP):
    """Adapt the standard dense MLP to SGLang's decoder-layer call signature."""

    def forward(self, hidden_states, *args, **kwargs):
        return super().forward(hidden_states)


class ModernMoEDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, config, layer_id, *args, **kwargs):
        super().__init__(config, layer_id, *args, **kwargs)
        old_attention = self.self_attn
        rope_theta, rope_scaling = get_rope_config(config)
        self.self_attn = ModernMoEAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=getattr(old_attention, "start_layer", 0),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            config=config,
            quant_config=kwargs.get("quant_config"),
            prefix=f"{kwargs.get('prefix', '')}.self_attn",
            dual_chunk_attention_config=getattr(config, "dual_chunk_attention_config", None),
            alt_stream=getattr(old_attention, "alt_stream", None),
        )
        if layer_id == 0:
            self.mlp = ModernMoEDenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=getattr(
                    config, "dense_intermediate_size", config.intermediate_size
                ),
                hidden_act=config.hidden_act,
                quant_config=kwargs.get("quant_config"),
                prefix=f"{kwargs.get('prefix', '')}.mlp",
            )
        else:
            self.mlp = ModernMoESparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=kwargs.get("quant_config"),
                prefix=f"{kwargs.get('prefix', '')}.mlp",
            )


class ModernMoEModel(Qwen3MoeModel):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=ModernMoEDecoderLayer,
        )


class ModernMoEForCausalLM(Qwen3MoeForCausalLM):
    """Native Modern-MoE model registered through SGLANG_EXTERNAL_MODEL_PACKAGE."""

    def __init__(self, config, quant_config=None, prefix=""):
        super(Qwen3MoeForCausalLM, self).__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = ModernMoEModel(config, quant_config, prefix="model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        self._trace = {}
        if __import__("os").environ.get("MODERN_MOE_FORWARD_TRACE"):
            self._register_forward_trace()

    def _register_forward_trace(self):
        import torch

        names = {
            "embedding": self.model.embed_tokens,
            "final_norm": self.model.norm,
        }
        for layer_id, layer in enumerate(self.model.layers):
            prefix = f"layer{layer_id}"
            names[f"{prefix}.input_norm"] = layer.input_layernorm
            names[f"{prefix}.attention.qkv_proj"] = layer.self_attn.qkv_proj
            names[f"{prefix}.attention"] = layer.self_attn
            names[f"{prefix}.post_attention_norm"] = layer.post_attention_layernorm
            names[f"{prefix}.moe"] = layer.mlp
            if hasattr(layer.mlp, "gate"):
                names[f"{prefix}.moe.router"] = layer.mlp.gate
            names[prefix] = layer

        def first_tensor(value):
            if torch.is_tensor(value):
                return value
            if isinstance(value, (tuple, list)):
                return next((first_tensor(item) for item in value if item is not None), None)
            return None

        def make_hook(name):
            def hook(_module, _inputs, output):
                value = first_tensor(output)
                if value is not None:
                    self._trace[name] = value.detach().float().cpu()
                if name.startswith("layer") and isinstance(output, (tuple, list)):
                    if len(output) > 1 and torch.is_tensor(output[1]):
                        self._trace[f"{name}.residual"] = output[1].detach().float().cpu()
                    if (
                        len(output) > 1
                        and torch.is_tensor(output[0])
                        and torch.is_tensor(output[1])
                        and output[0].shape == output[1].shape
                    ):
                        self._trace[f"{name}.combined"] = (
                            output[0] + output[1]
                        ).detach().float().cpu()
            return hook

        def make_pre_hook(name):
            def hook(_module, inputs):
                # DecoderLayer.forward(positions, hidden_states, batch, residual)
                index = 1 if name.endswith(".input") else 0
                if len(inputs) > index and torch.is_tensor(inputs[index]):
                    self._trace[name] = inputs[index].detach().float().cpu()
                if name.endswith(".input") and len(inputs) > 3 and torch.is_tensor(inputs[3]):
                    self._trace[name.replace(".input", ".residual_input")] = inputs[3].detach().float().cpu()
            return hook

        self._trace_hooks = [module.register_forward_hook(make_hook(name)) for name, module in names.items()]
        for layer_id, layer in enumerate(self.model.layers):
            self._trace_hooks += [
                layer.register_forward_pre_hook(make_pre_hook(f"layer{layer_id}.input")),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    make_pre_hook(f"layer{layer_id}.post_attention_norm_input")
                ),
                layer.mlp.register_forward_pre_hook(make_pre_hook(f"layer{layer_id}.moe_input")),
            ]

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        if __import__("os").environ.get("MODERN_MOE_DEBUG_OUTPUT"):
            import json
            import os

            logits = getattr(output, "next_token_logits", None)
            record = {
                "output_type": type(output).__name__,
                "logits_shape": None if logits is None else list(logits.shape),
                "logits_numel": None if logits is None else int(logits.numel()),
                "logits_finite": None
                if logits is None or logits.numel() == 0
                else int(torch.isfinite(logits).sum().item()),
                "logits_nan": None
                if logits is None or logits.numel() == 0
                else int(torch.isnan(logits).sum().item()),
                "logits_inf": None
                if logits is None or logits.numel() == 0
                else int(torch.isinf(logits).sum().item()),
                "logits_min": None
                if logits is None or logits.numel() == 0
                else float(torch.nan_to_num(logits).min().item()),
                "logits_max": None
                if logits is None or logits.numel() == 0
                else float(torch.nan_to_num(logits).max().item()),
            }
            path = os.environ.get(
                "MODERN_MOE_DEBUG_LOG",
                "logs/modern_moe_logits_debug.jsonl",
            )
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if __import__("os").environ.get("MODERN_MOE_FORWARD_TRACE"):
            import os
            import torch

            logits = getattr(output, "next_token_logits", None)
            if logits is not None:
                self._trace["final_logits"] = logits.detach().float().cpu()
                self._trace["top_ids"] = torch.topk(logits.float(), 10, dim=-1).indices.cpu()
            for layer_id, layer in enumerate(self.model.layers):
                mlp = layer.mlp
                if hasattr(mlp, "_trace_routed"):
                    self._trace[f"layer{layer_id}.moe.routed"] = mlp._trace_routed
                    self._trace[f"layer{layer_id}.moe.shared"] = mlp._trace_shared
                    for expert_id in range(len(mlp.shared_experts)):
                        for suffix in ("gate_up", "activated", ""):
                            attr = f"_trace_shared_{expert_id}"
                            key = f"layer{layer_id}.moe.shared.{expert_id}"
                            if suffix:
                                attr += f"_{suffix}"
                                key += f".{suffix}"
                            if hasattr(mlp, attr):
                                self._trace[key] = getattr(mlp, attr)
                    if layer_id == 1:
                        for expert_id, expert in enumerate(mlp.shared_experts):
                            self._trace[
                                f"layer1.moe.shared.{expert_id}.gate_up_weight"
                            ] = expert.gate_up_proj.weight.detach().float().cpu()
                            self._trace[
                                f"layer1.moe.shared.{expert_id}.down_weight"
                            ] = expert.down_proj.weight.detach().float().cpu()
            trace_path = os.environ.get(
                "MODERN_MOE_FORWARD_TRACE_PATH",
                "runs/ab_compare/sglang_trace.pt",
            )
            torch.save(self._trace, trace_path)
        return output

    @staticmethod
    def _map_native_name(name: str):
        if name == "embed_tokens.weight":
            return "model.embed_tokens.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name == "lm_head.weight":
            return "lm_head.weight"
        if not name.startswith("layers."):
            return None
        layer_text, suffix = name.removeprefix("layers.").split(".", 1)
        layer = int(layer_text)
        prefix = f"model.layers.{layer}"
        direct = {
            "input_norm.weight": f"{prefix}.input_layernorm.weight",
            "post_attention_norm.weight": f"{prefix}.post_attention_layernorm.weight",
            "attention.q_proj.weight": f"{prefix}.self_attn.q_proj.weight",
            "attention.k_proj.weight": f"{prefix}.self_attn.k_proj.weight",
            "attention.v_proj.weight": f"{prefix}.self_attn.v_proj.weight",
            "attention.o_proj.weight": f"{prefix}.self_attn.o_proj.weight",
            "moe.router.weight": f"{prefix}.mlp.gate.weight",
            "moe.ffn.gate_proj.weight": f"{prefix}.mlp.gate_proj.weight",
            "moe.ffn.up_proj.weight": f"{prefix}.mlp.up_proj.weight",
            "moe.ffn.down_proj.weight": f"{prefix}.mlp.down_proj.weight",
        }
        return direct.get(suffix)

    _moe_routed_regex = __import__("re").compile(
        r"moe\.routed\.experts\.(\d+)\.linear_fc([12])\.weight"
    )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Per-expert routed tensors arrive one parameter at a time from Slime's
        # Megatron weight sync.  Collect by layer, then pack below into the
        # FusedMoE w13/w2 tensors after the loop.
        pending_routed: dict[int, dict[str, dict[int, torch.Tensor]]] = {}
        mapped = []
        for name, weight in weights:
            if name == "embed_tokens.weight":
                mapped.append(("model.embed_tokens.weight", weight))
            elif name == "norm.weight":
                mapped.append(("model.norm.weight", weight))
            elif name == "lm_head.weight":
                mapped.append(("lm_head.weight", weight))
            elif name.startswith("layers."):
                layer_text, suffix = name.removeprefix("layers.").split(".", 1)
                layer = int(layer_text)
                prefix = f"model.layers.{layer}"
                routed_m = self._moe_routed_regex.fullmatch(suffix)
                if routed_m is not None:
                    expert, proj = int(routed_m.group(1)), routed_m.group(2)
                    pending = pending_routed.setdefault(layer, {"fc1": {}, "fc2": {}})
                    pending["fc1" if proj == "1" else "fc2"][expert] = weight
                elif suffix == "moe.router.weight":
                    mapped.append((f"{prefix}.mlp.gate.weight", weight))
                elif suffix == "moe.routed.experts.weight":
                    # Qwen3's routed implementation is FusedMoE. Its actual
                    # parameters are one expert-major w13 tensor, not a
                    # ModuleList of gate/up projections.
                    mapped.append((f"{prefix}.mlp.experts.w13_weight", weight))
                elif suffix == "moe.routed.output_experts.weight":
                    mapped.append((f"{prefix}.mlp.experts.w2_weight", weight))
                elif suffix == "moe.shared.experts.weight":
                    half = weight.shape[1] // 2
                    for expert_id, packed in enumerate(weight):
                        # packed_liger stores shared experts as [up, gate],
                        # unlike routed experts, which are [gate, up].
                        up, gate = packed[:half], packed[half:]
                        mapped.append(
                            (
                                f"{prefix}.mlp.shared_experts.{expert_id}.gate_up_proj.weight",
                                torch.cat((gate, up), dim=0),
                            )
                        )
                elif suffix == "moe.shared.output_experts.weight":
                    for expert_id, expert_weight in enumerate(weight):
                        mapped.append((f"{prefix}.mlp.shared_experts.{expert_id}.down_proj.weight", expert_weight))
                else:
                    target = self._map_native_name(name)
                    if target is not None:
                        mapped.append((target, weight))
        # Pack per-expert routed tensors (from Slime's per-parameter weight
        # sync) into the FusedMoE w13/w2 tensors, matching the packed format
        # ``moe.routed.experts.weight`` / ``moe.routed.output_experts.weight``
        # the loader would otherwise expect.  Skip layers that already were
        # supplied as packed tensors (initial on-disk load path).
        for layer, buckets in pending_routed.items():
            prefix = f"model.layers.{layer}"
            fc1 = buckets.get("fc1") or {}
            fc2 = buckets.get("fc2") or {}
            if fc1:
                # native routed is [gate, up] per expert: keep as-is (columns
                # already ordered); just stack all experts into one tensor.
                idx = sorted(fc1)
                w13 = torch.stack([fc1[i] for i in idx], dim=0)
                mapped.append((f"{prefix}.mlp.experts.w13_weight", w13))
            if fc2:
                idx = sorted(fc2)
                w2 = torch.stack([fc2[i] for i in idx], dim=0)
                mapped.append((f"{prefix}.mlp.experts.w2_weight", w2))
        # FusedMoE's internal ``w13_weight``/``w2_weight`` loader signature is
        # not compatible with the generic Qwen loader when supplied as a
        # whole expert-major tensor.  Load ordinary parameters through the
        # generic path and handle these two tensors below.
        fused_targets = {
            target
            for target, _ in mapped
            if target.endswith(".experts.w13_weight") or target.endswith(".experts.w2_weight")
        }
        super().load_weights(
            [(target, value) for target, value in mapped if target not in fused_targets]
        )
        # The custom checkpoint names are already in the exact SGLang
        # parameter layout for this single-GPU diagnostic model.  Keep an
        # explicit shape-checked copy as a fallback because generic
        # load_weights can silently skip custom shared/FusedMoE targets.
        direct_loaded = []
        params = dict(self.named_parameters())
        with torch.no_grad():
            for target, value in mapped:
                if ".mlp." not in target or target not in params:
                    continue
                param = params[target]
                if tuple(param.shape) != tuple(value.shape):
                    continue
                param.copy_(value.to(device=param.device, dtype=param.dtype))
                direct_loaded.append(target)
        for layer in self.model.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None and hasattr(experts, "process_weights_after_loading"):
                experts.process_weights_after_loading(experts)
        if __import__("os").environ.get("MODERN_MOE_DEBUG_WEIGHTS"):
            print(f"Modern-MoE direct expert loads: {len(direct_loaded)}", flush=True)
        if __import__("os").environ.get("MODERN_MOE_DEBUG_WEIGHTS"):
            import json
            import os

            params = dict(self.named_parameters())
            audit_names = [
                "model.embed_tokens.weight",
                "lm_head.weight",
                "model.layers.0.self_attn.qkv_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.1.mlp.gate.weight",
                "model.layers.1.mlp.experts.0.gate_up_proj.weight",
                "model.layers.1.mlp.shared_experts.0.gate_up_proj.weight",
                "model.layers.1.mlp.shared_experts.0.down_proj.weight",
                "model.layers.1.mlp.shared_experts.1.gate_up_proj.weight",
                "model.layers.1.mlp.shared_experts.1.down_proj.weight",
            ]
            audit = {
                name: {
                    "shape": list(params[name].shape),
                    "norm": float(params[name].detach().float().norm().item()),
                    "mean": float(params[name].detach().float().mean().item()),
                }
                for name in audit_names
                if name in params
            }
            path = os.environ.get(
                "MODERN_MOE_DEBUG_WEIGHTS_LOG",
                "logs/modern_moe_weight_audit.json",
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(audit, handle, indent=2)
            print("Modern-MoE weight audit:", json.dumps(audit), flush=True)


EntryClass = ModernMoEForCausalLM
