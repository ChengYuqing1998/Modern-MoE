"""Profile the fixed fast-path inference baseline with a Chrome trace."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_moe.generation import (
    GenerationConfig,
    _filtered_probabilities,
    _sample,
    generate,
)
from modern_moe.inference_graph import CUDAGraphedDecode
from scripts.generate import load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="梯度下降法（英语：Gradient descent）是一种"
    )
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cuda-graph-decode", action="store_true")
    parser.add_argument("--vllm-fused-experts", action="store_true")
    parser.add_argument(
        "--trace", type=Path, default=Path("profiles/inference_decode_trace.json")
    )
    args = parser.parse_args()
    os.environ["MODERN_MOE_USE_VLLM_FUSED_EXPERTS"] = (
        "1" if args.vllm_fused_experts else "0"
    )

    model, config, _ = load_model(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, use_fast=True)
    input_ids = tokenizer(
        args.prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to("cuda")
    generation_config = GenerationConfig(
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
        mode="cache",
        cuda_graph_decode=args.cuda_graph_decode,
    )

    # Compile/lazy-load kernels and build non-persistent expert weight caches
    # before profiling. The measured call still includes its own KV prefill.
    torch.manual_seed(1337)
    generate(
        model,
        input_ids,
        GenerationConfig(
            max_new_tokens=8,
            temperature=args.temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            mode="cache",
        ),
    )
    torch.cuda.synchronize()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)
    graph_runner = None
    generated = input_ids
    if args.cuda_graph_decode:
        # Build all lazy state before entering the profiler. This gives a true
        # steady-state trace rather than charging Triton JIT and graph capture
        # to the first decoded token.
        main = model.forward_inference(
            generated,
            max_cache_length=input_ids.size(1) + args.tokens + 1,
        )
        target = _filtered_probabilities(
            main.logits[:, -1], generated, generation_config
        )
        token = _sample(target)
        generated = torch.cat((generated, token), dim=1)
        graph_runner = CUDAGraphedDecode(
            model, main.cache, input_ids.size(1) + args.tokens + 1
        )
        main = graph_runner.replay(token, main.cache[0].length)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        started = time.perf_counter()
        if graph_runner is None:
            result = generate(model, input_ids, generation_config)
        else:
            for _ in range(args.tokens):
                target = _filtered_probabilities(
                    main.logits[:, -1], generated, generation_config
                )
                token = _sample(target)
                generated = torch.cat((generated, token), dim=1)
                main = graph_runner.replay(token, main.cache[0].length)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
    torch.cuda.synchronize()
    profiler.export_chrome_trace(str(args.trace))
    if graph_runner is None:
        print(
            f"prefill={result.prefill_seconds*1000:.3f}ms "
            f"decode={result.decode_tokens_per_second:.3f}tok/s "
            f"decode_per_token={1000/result.decode_tokens_per_second:.3f}ms "
            f"overall={result.tokens_per_second:.3f}tok/s"
        )
    else:
        print(
            f"steady_graph_tokens={args.tokens} elapsed={elapsed:.6f}s "
            f"decode={args.tokens / elapsed:.3f}tok/s "
            f"decode_per_token={elapsed * 1000 / args.tokens:.3f}ms"
        )
    print("\nTOP CUDA TOTAL")
    print(
        profiler.key_averages().table(
            sort_by="cuda_time_total", row_limit=35
        )
    )
    print("\nTOP CPU SELF")
    print(
        profiler.key_averages().table(
            sort_by="self_cpu_time_total", row_limit=35
        )
    )
    print(f"trace={args.trace}")


if __name__ == "__main__":
    main()
