#!/usr/bin/env python3
"""Dump a deterministic native Modern-MoE prefill trace for SGLang A/B."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.generate import load_model
from modern_moe.liger_moe import dense_shared_swiglu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model, config, _ = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(
        rendered, return_tensors="pt", add_special_tokens=False
    ).input_ids.cuda()

    trace: dict[str, torch.Tensor | list[int] | str] = {
        "input_ids": input_ids[0].cpu(),
        "rendered_prompt": rendered,
    }

    def save_hook(name: str):
        def hook(_module, _inputs, output):
            value = output
            if isinstance(value, (tuple, list)):
                value = next((item for item in value if torch.is_tensor(item)), None)
            if torch.is_tensor(value):
                trace[name] = value.detach().float().cpu()
        return hook

    def save_pre_hook(name: str):
        def hook(_module, inputs):
            if inputs and torch.is_tensor(inputs[0]):
                trace[name] = inputs[0].detach().float().cpu()
        return hook

    layer = model.layers[0]
    modules = {
        "embedding": model.embed_tokens,
        "final_norm": model.norm,
    }
    for layer_id, decoder_layer in enumerate(model.layers):
        prefix = f"layer{layer_id}"
        layer_modules = {
                f"{prefix}.input_norm": decoder_layer.input_norm,
                f"{prefix}.attention.q_proj": decoder_layer.attention.q_proj,
                f"{prefix}.attention.k_proj": decoder_layer.attention.k_proj,
                f"{prefix}.attention.v_proj": decoder_layer.attention.v_proj,
                f"{prefix}.attention": decoder_layer.attention,
                f"{prefix}.post_attention_norm": decoder_layer.post_attention_norm,
                f"{prefix}.moe": decoder_layer.moe,
                prefix: decoder_layer,
        }
        if hasattr(decoder_layer.moe, "router"):
            layer_modules[f"{prefix}.moe.router"] = decoder_layer.moe.router
        modules.update(layer_modules)
    hooks = [module.register_forward_hook(save_hook(name)) for name, module in modules.items()]
    for layer_id, decoder_layer in enumerate(model.layers):
        hooks += [
            decoder_layer.post_attention_norm.register_forward_pre_hook(
                save_pre_hook(f"layer{layer_id}.post_attention_norm_input")
            ),
            decoder_layer.moe.register_forward_pre_hook(
                save_pre_hook(f"layer{layer_id}.moe_input")
            ),
        ]
    # Inference uses DecoderLayer.forward_cached(), so a normal module hook on
    # the layer itself does not fire. Wrap that method to capture the residual
    # stream after every layer without changing the model computation.
    original_cached = []
    for layer_id, decoder_layer in enumerate(model.layers):
        original = decoder_layer.forward_cached
        original_cached.append((decoder_layer, original))

        def wrapped(*args, _original=original, _layer_id=layer_id, **kwargs):
            if args and torch.is_tensor(args[0]):
                trace[f"layer{_layer_id}.input"] = args[0].detach().float().cpu()
            result = _original(*args, **kwargs)
            if isinstance(result, (tuple, list)) and torch.is_tensor(result[0]):
                trace[f"layer{_layer_id}"] = result[0].detach().float().cpu()
            return result

        decoder_layer.forward_cached = wrapped

        attention = decoder_layer.attention
        attention_original = attention.forward_cached
        original_cached.append((attention, attention_original))

        def attention_wrapped(
            *args, _original=attention_original, _layer_id=layer_id, **kwargs
        ):
            result = _original(*args, **kwargs)
            if isinstance(result, (tuple, list)) and torch.is_tensor(result[0]):
                trace[f"layer{_layer_id}.attention"] = result[0].detach().float().cpu()
            return result

        attention.forward_cached = attention_wrapped
    with torch.inference_mode():
        output = model.forward_inference(input_ids)
    for decoder_layer, original in original_cached:
        decoder_layer.forward_cached = original
    for hook in hooks:
        hook.remove()
    trace["final_logits"] = output.logits[:, -1].detach().float().cpu()
    with torch.inference_mode():
        for layer_id, decoder_layer in enumerate(model.layers):
            if hasattr(decoder_layer.moe, "shared") and f"layer{layer_id}.moe_input" in trace:
                moe_input = trace[f"layer{layer_id}.moe_input"].cuda().to(torch.bfloat16)
                shared = dense_shared_swiglu(
                    moe_input.reshape(-1, moe_input.size(-1)),
                    decoder_layer.moe.shared.experts.weight,
                    decoder_layer.moe.shared.output_experts.weight,
                )
                trace[f"layer{layer_id}.moe.shared"] = shared.reshape(moe_input.shape).float().cpu()
                flat_input = moe_input.reshape(-1, moe_input.size(-1))
                shared_gate_up = torch.bmm(
                    flat_input.unsqueeze(0).expand(
                        decoder_layer.moe.num_shared_experts, -1, -1
                    ),
                    decoder_layer.moe.shared.experts.weight.transpose(1, 2),
                )
                shared_up, shared_gate = shared_gate_up.chunk(2, dim=-1)
                shared_activated = torch.nn.functional.silu(shared_gate) * shared_up
                shared_per_expert = torch.bmm(
                    shared_activated,
                    decoder_layer.moe.shared.output_experts.weight.transpose(1, 2),
                )
                for expert_id in range(decoder_layer.moe.num_shared_experts):
                    trace[f"layer{layer_id}.moe.shared.{expert_id}.gate_up"] = (
                        torch.cat(
                            (shared_gate_up[expert_id, :, shared_up.size(-1):],
                             shared_gate_up[expert_id, :, :shared_up.size(-1)]),
                            dim=-1,
                        ).float().cpu()
                    )
                    trace[f"layer{layer_id}.moe.shared.{expert_id}.activated"] = (
                        shared_activated[expert_id].float().cpu()
                    )
                    trace[f"layer{layer_id}.moe.shared.{expert_id}"] = (
                        shared_per_expert[expert_id].float().cpu()
                    )
                trace[f"layer{layer_id}.moe.routed"] = (
                    trace[f"layer{layer_id}.moe"].float() - trace[f"layer{layer_id}.moe.shared"]
                )
    trace["top_ids"] = torch.topk(trace["final_logits"], 10, dim=-1).indices[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, args.output)
    print(f"saved {args.output}")
    print(f"input_ids={input_ids[0].tolist()}")
    print(f"top_ids={trace['top_ids'].tolist()}")


if __name__ == "__main__":
    main()
