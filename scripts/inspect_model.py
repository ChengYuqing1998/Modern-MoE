import argparse
from pathlib import Path

import torch
import yaml

from modern_moe import ModernMoEConfig, ModernMoEForCausalLM


def module_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def active_parameters(model, config):
    """Unique weights on one token's route; tied embeddings are counted once."""
    count = module_parameters(model.embed_tokens) + module_parameters(model.norm)
    if not config.tie_word_embeddings:
        count += module_parameters(model.lm_head)
    def active_decoder_parameters(layer):
        layer_count = 0
        layer_count += module_parameters(layer.attention)
        layer_count += module_parameters(layer.input_norm)
        layer_count += module_parameters(layer.post_attention_norm)
        if not hasattr(layer.moe, "router"):
            layer_count += module_parameters(layer.moe)
            return layer_count
        layer_count += module_parameters(layer.moe.router)
        layer_count += sum(
            module_parameters(layer.moe.experts[index])
            for index in range(config.num_experts_per_tok)
        )
        layer_count += sum(
            module_parameters(expert) for expert in layer.moe.shared_experts
        )
        return layer_count

    for layer in model.layers:
        count += active_decoder_parameters(layer)
    for mtp_layer in model.mtp_layers:
        count += module_parameters(mtp_layer.hidden_norm)
        count += module_parameters(mtp_layer.token_norm)
        count += module_parameters(mtp_layer.fusion)
        count += active_decoder_parameters(mtp_layer.decoder)
        count += module_parameters(mtp_layer.output_norm)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/modern_moe_1b.yaml"))
    parser.add_argument("--instantiate", action="store_true", help="Allocate real model tensors.")
    args = parser.parse_args()
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_config["attention_pattern"] = tuple(raw_config["attention_pattern"])
    config = ModernMoEConfig(**raw_config)

    if args.instantiate:
        model = ModernMoEForCausalLM(config)
    else:
        with torch.device("meta"):
            model = ModernMoEForCausalLM(config)
    total = model.num_parameters()
    active = active_parameters(model, config)
    full_layers = sum(config.attention_type(i) == "full" for i in range(config.num_hidden_layers))
    print(f"parameters: {total:,} ({total / 1e9:.3f}B)")
    print(f"active/token: {active:,} ({active / 1e6:.3f}M, {active / total:.2%})")
    print(f"layers: {config.num_hidden_layers} ({full_layers} full, "
          f"{config.num_hidden_layers - full_layers} linear)")
    if config.use_moe:
        print(f"routing: top-{config.num_experts_per_tok}/{config.num_experts} "
              f"+ {config.num_shared_experts} shared")
        print(f"ffn layers: {config.first_k_dense_replace} dense "
              f"(intermediate_size={config.dense_intermediate_size}), "
              f"{config.num_hidden_layers - config.first_k_dense_replace} MoE "
              f"(expert intermediate_size={config.intermediate_size})")
    else:
        print(f"ffn: dense SwiGLU, intermediate_size={config.intermediate_size}")
    print(f"mtp: {config.num_mtp_layers} layer(s), loss coefficient "
          f"{config.mtp_loss_coef:g}")


if __name__ == "__main__":
    main()
