from __future__ import annotations

import time

PROCESS_STARTED = time.perf_counter()

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, TextStreamer

from modern_moe.config import ModernMoEConfig
from modern_moe.generation import GenerationConfig, generate
from modern_moe.model import ModernMoEForCausalLM
from nanok3.config import NanoK3Config
from nanok3.model import NanoK3ForCausalLM

LanguageModel = ModernMoEForCausalLM | NanoK3ForCausalLM
ModelConfig = ModernMoEConfig | NanoK3Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with Modern-MoE.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help=(
            "Wrap the prompt in the tokenizer's user/assistant chat template, "
            "stop on <|im_end|>, and print only the assistant completion."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("no_cache", "cache", "mtp", "all"),
        default="cache",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=0,
        help="Ban repeated n-grams of this size; 0 disables the constraint.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--cuda-graph-decode",
        action="store_true",
        help="Capture the fixed-shape single-token model decode; cache mode only.",
    )
    parser.add_argument(
        "--vllm-fused-experts",
        action="store_true",
        help="Use the decode-only vLLM Triton fused expert GEMMs (cache mode).",
    )
    parser.add_argument(
        "--fused-inference-router",
        action="store_true",
        help="Fuse decode softmax/Top-k/renormalization/shared routing metadata.",
    )
    parser.add_argument(
        "--fused-sampling",
        action="store_true",
        help="Use GPU-resident repetition/ngram state and Top-k-first top-p sampling.",
    )
    parser.add_argument(
        "--flashinfer-sampling",
        action="store_true",
        help="Use FlashInfer sorting-free Top-k/Top-p sampling after GPU penalties.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Decode and print generated tokens incrementally.",
    )
    return parser.parse_args()


def load_model(
    path: Path,
) -> tuple[LanguageModel, ModelConfig, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    started = time.perf_counter()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_loaded = time.perf_counter()
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint does not contain model_config")
    if isinstance(raw_config.get("attention_pattern"), list):
        raw_config["attention_pattern"] = tuple(raw_config["attention_pattern"])
    if raw_config.get("model_type") == "nanoK3":
        config = NanoK3Config(**raw_config)
        model = NanoK3ForCausalLM(config)
    else:
        config = ModernMoEConfig(**raw_config)
        model = ModernMoEForCausalLM(config)
    model_built = time.perf_counter()
    model.load_state_dict(checkpoint["model"], strict=True)
    weights_loaded = time.perf_counter()
    del checkpoint
    model.eval().to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    cuda_ready = time.perf_counter()
    return model, config, {
        "checkpoint_read_seconds": checkpoint_loaded - started,
        "model_build_seconds": model_built - checkpoint_loaded,
        "state_dict_load_seconds": weights_loaded - model_built,
        "cuda_transfer_seconds": cuda_ready - weights_loaded,
        "model_load_seconds": cuda_ready - started,
    }


def main() -> None:
    main_entered = time.perf_counter()
    args = parse_args()
    os.environ["MODERN_MOE_USE_VLLM_FUSED_EXPERTS"] = (
        "1" if args.vllm_fused_experts else "0"
    )
    os.environ["MODERN_MOE_USE_FUSED_INFERENCE_ROUTER"] = (
        "1" if args.fused_inference_router else "0"
    )
    if not torch.cuda.is_available():
        raise RuntimeError("KDA inference requires CUDA")
    model, model_config, startup_timings = load_model(args.checkpoint)
    is_nanok3 = isinstance(model, NanoK3ForCausalLM)
    if is_nanok3 and args.mode in {"mtp", "all"}:
        raise ValueError(
            "nanoK3 supports no_cache and cache, but has no trained MTP layer."
        )
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_path,
        use_fast=True,
    )
    tokenizer_loaded = time.perf_counter()
    rendered_prompt = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if args.chat_template
        else args.prompt
    )
    input_ids = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to("cuda")
    stop_token_id = (
        tokenizer.convert_tokens_to_ids("<|im_end|>")
        if args.chat_template
        else tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    prompt_ready = time.perf_counter()
    if input_ids.size(1) + args.max_new_tokens > model_config.max_position_embeddings:
        raise ValueError(
            "prompt tokens + max_new_tokens exceeds max_position_embeddings"
        )
    print(
        f"model={'nanoK3' if is_nanok3 else 'Modern-MoE'} "
        f"parameters={sum(p.numel() for p in model.parameters()):,} "
        f"tie_word_embeddings={model_config.tie_word_embeddings} "
        f"prompt_tokens={input_ids.size(1)}",
        flush=True,
    )
    print(
        f"startup framework_import={main_entered - PROCESS_STARTED:.2f}s "
        f"model_load={startup_timings['model_load_seconds']:.2f}s "
        f"(checkpoint_read={startup_timings['checkpoint_read_seconds']:.2f}s "
        f"model_build={startup_timings['model_build_seconds']:.2f}s "
        f"state_dict_load={startup_timings['state_dict_load_seconds']:.2f}s "
        f"cuda_transfer={startup_timings['cuda_transfer_seconds']:.2f}s) "
        f"tokenizer_load={tokenizer_loaded - tokenizer_started:.2f}s "
        f"prompt_tokenize_and_transfer={prompt_ready - tokenizer_loaded:.3f}s "
        f"ready_from_process_start={prompt_ready - PROCESS_STARTED:.2f}s",
        flush=True,
    )
    modes = ("no_cache", "cache", "mtp") if args.mode == "all" else (args.mode,)
    for mode in modes:
        torch.manual_seed(args.seed)
        streamer = None
        if args.stream:
            print(f"\n===== {mode} (stream) =====", flush=True)
            streamer = TextStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            streamer.put(input_ids.detach().cpu())
        generation_call_started = time.perf_counter()
        result = generate(
            model,
            input_ids,
            GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                mode=mode,
                cuda_graph_decode=args.cuda_graph_decode and mode == "cache",
                fused_sampling=args.fused_sampling and mode == "cache",
                flashinfer_sampling=args.flashinfer_sampling and mode == "cache",
            ),
            eos_token_id=stop_token_id,
            stream_callback=(
                (lambda token: streamer.put(token.detach().cpu()))
                if streamer is not None
                else None
            ),
        )
        if streamer is not None:
            streamer.end()
        else:
            print(f"\n===== {mode} =====")
            output_ids = (
                result.token_ids[0, input_ids.size(1) :]
                if args.chat_template
                else result.token_ids[0]
            )
            print(tokenizer.decode(output_ids, skip_special_tokens=True))
        print(
            f"\nmode={mode} new_tokens={result.new_tokens} "
            f"prefill={result.prefill_seconds * 1000:.2f}ms "
            f"ttft={result.time_to_first_token_seconds * 1000:.2f}ms "
            f"end_to_end_ttft="
            f"{generation_call_started - PROCESS_STARTED + result.time_to_first_token_seconds:.2f}s "
            f"decode={result.decode_seconds:.2f}s "
            f"graph_setup={result.decode_graph_setup_seconds * 1000:.2f}ms "
            f"decode_speed={result.decode_tokens_per_second:.2f} tok/s "
            f"time={result.elapsed_seconds:.2f}s "
            f"overall_speed={result.tokens_per_second:.2f} tok/s",
            flush=True,
        )
        if mode == "mtp":
            print(
                f"mtp_proposed={result.mtp_proposed} "
                f"mtp_accepted={result.mtp_accepted} "
                f"acceptance={result.mtp_acceptance_rate:.2%}",
                flush=True,
            )


if __name__ == "__main__":
    main()
