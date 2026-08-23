"""A/B benchmark the reference and inference-only Modern-MoE paths."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_moe.config import ModernMoEConfig
from modern_moe.generation import GenerationConfig, generate
from modern_moe.layers import SparseMoE
from modern_moe.model import ModernMoEForCausalLM
from scripts.convert_checkpoint_to_packed_scattermoe import (
    convert_model_state,
    model_state_names,
)
from scripts.generate import load_model


def load_packed_from_legacy(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    values = dict(checkpoint["model_config"])
    values["attention_pattern"] = tuple(values["attention_pattern"])
    legacy_config = ModernMoEConfig(**values)
    values["moe_parameter_layout"] = "packed_scattermoe"
    config = ModernMoEConfig(**values)
    state = convert_model_state(
        checkpoint["model"], model_state_names(values, "packed_scattermoe"), legacy_config
    )
    del checkpoint
    model = ModernMoEForCausalLM(config)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().to(device="cuda", dtype=torch.bfloat16)
    return model, config, {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cuda-graph-decode", action="store_true")
    parser.add_argument("--decode-graph-ab", action="store_true")
    parser.add_argument("--packed-in-memory", action="store_true")
    parser.add_argument("--vllm-fused-experts", action="store_true")
    parser.add_argument("--fused-sampling", action="store_true")
    parser.add_argument("--sampling-ab", action="store_true")
    parser.add_argument("--flashinfer-sampling", action="store_true")
    parser.add_argument("--flashinfer-ab", action="store_true")
    parser.add_argument("--fused-inference-router", action="store_true")
    parser.add_argument("--router-ab", action="store_true")
    args = parser.parse_args()
    os.environ["MODERN_MOE_USE_VLLM_FUSED_EXPERTS"] = (
        "1" if args.vllm_fused_experts else "0"
    )
    os.environ["MODERN_MOE_USE_FUSED_INFERENCE_ROUTER"] = (
        "1" if args.fused_inference_router else "0"
    )

    if args.packed_in_memory:
        model, config, _ = load_packed_from_legacy(args.checkpoint)
    else:
        model, config, _ = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    input_ids = tokenizer(
        args.prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to("cuda")
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
        mode="cache",
        cuda_graph_decode=args.cuda_graph_decode,
        fused_sampling=args.fused_sampling,
        flashinfer_sampling=args.flashinfer_sampling,
    )
    results = []
    # Packed mode runs twice to distinguish one-time inference-cache packing
    # from steady-state prefill/decode performance.
    modes = (False, False) if args.packed_in_memory else (False, True)
    if args.decode_graph_ab:
        modes = (True, True)
    if args.sampling_ab:
        modes = (True, True, True)
    if args.flashinfer_ab:
        modes = (True, True, True, True)
    if args.router_ab:
        modes = (True, True, True)
    for mode_index, enabled in enumerate(modes):
        os.environ["MODERN_MOE_USE_INFERENCE_FAST_PATH"] = "1" if enabled else "0"
        for module in model.modules():
            if isinstance(module, SparseMoE):
                module.use_inference_fast_path = enabled
                module._inference_gate_up = None
                module._inference_down = None
        generation_config.cuda_graph_decode = (
            args.cuda_graph_decode
            or (args.decode_graph_ab and mode_index == 1)
        )
        generation_config.fused_sampling = args.fused_sampling or (
            (args.sampling_ab or args.flashinfer_ab) and mode_index == 2
        )
        generation_config.flashinfer_sampling = args.flashinfer_sampling or (
            args.flashinfer_ab and mode_index == 3
        )
        fused_router = args.fused_inference_router or (
            args.router_ab and mode_index == 2
        )
        os.environ["MODERN_MOE_USE_FUSED_INFERENCE_ROUTER"] = (
            "1" if fused_router else "0"
        )
        torch.manual_seed(1337)
        torch.cuda.reset_peak_memory_stats()
        result = generate(model, input_ids, generation_config)
        results.append(result)
        print(
            f"fast_path={int(enabled)} graph={int(generation_config.cuda_graph_decode)} "
            f"fused_sampling={int(generation_config.fused_sampling)} "
            f"flashinfer_sampling={int(generation_config.flashinfer_sampling)} "
            f"fused_router={int(fused_router)} "
            f"prefill={result.prefill_seconds * 1000:.2f}ms "
            f"decode={result.decode_tokens_per_second:.2f} tok/s "
            f"overall={result.tokens_per_second:.2f} tok/s "
            f"graph_setup={result.decode_graph_setup_seconds * 1000:.2f}ms "
            f"peak={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
        )
    if len(results) == 2:
        print("tokens_equal=", torch.equal(results[0].token_ids, results[1].token_ids))


if __name__ == "__main__":
    main()
