"""Probe how DPO changes a trigger word at token, layer, and router level.

Example:
  python -u -m scripts.probe_dpo_trigger \
    --sft path/to/sft.pt --dpo path/to/dpo_final.pt \
    --prompt "我想给你鸡巴章" --target "黑话"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modern_moe.config import ModernMoEConfig
from modern_moe.model import ModernMoEForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target", default="黑话")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--forced-prefixes",
        nargs="*",
        default=[
            "<think>\n\n</think>\n\n",
            "<think>\n\n</think>\n\n你是不是觉得我这会儿就是个",
            "<think>\n\n</think>\n\n你是不是觉得你是一个",
        ],
        help="Assistant-side prefixes inserted after the rendered chat prompt.",
    )
    return parser.parse_args()


def load_model(path: Path) -> tuple[ModernMoEForCausalLM, ModernMoEConfig]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = dict(checkpoint["model_config"])
    if isinstance(raw.get("attention_pattern"), list):
        raw["attention_pattern"] = tuple(raw["attention_pattern"])
    config = ModernMoEConfig(**raw)
    model = ModernMoEForCausalLM(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval().to(device="cuda", dtype=torch.bfloat16)
    return model, config


def token_stats(logits: torch.Tensor, token_id: int) -> tuple[float, int]:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    logp = float(log_probs[token_id])
    rank = int((logits > logits[token_id]).sum().item()) + 1
    return math.exp(logp), rank


def entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    return float(-(probs * probs.clamp_min(1e-30).log()).sum())


def top_tokens(tokenizer, logits: torch.Tensor, k: int) -> list[tuple[str, int, float]]:
    probs = F.softmax(logits.float(), dim=-1)
    values, indices = probs.topk(k)
    return [
        (tokenizer.decode([int(i)]).replace("\n", "\\n"), int(i), float(v))
        for i, v in zip(indices, values)
    ]


def capture_forward(model, input_ids):
    layers: dict[int, torch.Tensor] = {}
    routers: dict[str, torch.Tensor] = {}
    handles = []

    for index, layer in enumerate(model.layers):
        def layer_hook(_module, _inputs, output, layer_index=index):
            states = output[0] if isinstance(output, tuple) else output
            layers[layer_index] = states.reshape(-1, states.shape[-1])[-1].detach().float()

        handles.append(layer.register_forward_hook(layer_hook))

    for name, module in model.named_modules():
        if name.endswith(".router"):
            def router_hook(_module, _inputs, output, router_name=name):
                routers[router_name] = output.reshape(-1, output.shape[-1])[-1].detach().float()

            handles.append(module.register_forward_hook(router_hook))

    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids=input_ids)
        logits = output.logits.reshape(-1, output.logits.shape[-1])[-1].detach().float()
    finally:
        for handle in handles:
            handle.remove()
    return logits, layers, routers


def print_next_token_report(name, tokenizer, logits, target_ids, top_k):
    print(f"  {name}: entropy={entropy(logits):.4f}")
    for token_id in target_ids:
        probability, rank = token_stats(logits, token_id)
        token = tokenizer.decode([token_id]).replace("\n", "\\n")
        print(f"    target token={token!r} id={token_id} p={probability:.8g} rank={rank}")
    rendered = ", ".join(
        f"{token!r}:{probability:.5f}" for token, _token_id, probability
        in top_tokens(tokenizer, logits, top_k)
    )
    print(f"    top-{top_k}: {rendered}")


def row_delta_percentile(
    sft_weight: torch.Tensor,
    dpo_weight: torch.Tensor,
    token_id: int,
    chunk_size: int = 4096,
) -> tuple[float, float]:
    target_delta = float(
        (dpo_weight[token_id].detach().float() - sft_weight[token_id].detach().float()).norm()
    )
    less_or_equal = 0
    rows = sft_weight.shape[0]
    for start in range(0, rows, chunk_size):
        stop = min(start + chunk_size, rows)
        delta = (
            dpo_weight[start:stop].detach().float()
            - sft_weight[start:stop].detach().float()
        )
        less_or_equal += int((delta.norm(dim=1) <= target_delta).sum())
    return target_delta, 100.0 * less_or_equal / rows


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA")

    print("loading SFT model...")
    sft, sft_config = load_model(args.sft)
    print("loading DPO model...")
    dpo, dpo_config = load_model(args.dpo)
    if sft_config.vocab_size != dpo_config.vocab_size:
        raise ValueError("SFT and DPO vocab sizes differ")

    tokenizer = AutoTokenizer.from_pretrained(sft_config.tokenizer_path, use_fast=True)
    target_ids = tokenizer.encode(args.target, add_special_tokens=False)
    print(f"target={args.target!r} ids={target_ids} tokens={tokenizer.convert_ids_to_tokens(target_ids)}")

    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    for forced_prefix in args.forced_prefixes:
        context = rendered_prompt + forced_prefix
        input_ids = tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
        print("\n" + "=" * 88)
        print(f"forced_prefix={forced_prefix!r} tokens={input_ids.shape[1]}")
        sft_logits, sft_layers, sft_routers = capture_forward(sft, input_ids)
        dpo_logits, dpo_layers, dpo_routers = capture_forward(dpo, input_ids)
        print_next_token_report("SFT", tokenizer, sft_logits, target_ids, args.top_k)
        print_next_token_report("DPO", tokenizer, dpo_logits, target_ids, args.top_k)

        first_id = target_ids[0]
        sft_p, _ = token_stats(sft_logits, first_id)
        dpo_p, _ = token_stats(dpo_logits, first_id)
        print(f"  target-first probability ratio DPO/SFT={dpo_p / max(sft_p, 1e-30):.6g}")

        print("  layer last-token cosine(SFT,DPO):")
        print("   " + " ".join(
            f"L{i}={float(F.cosine_similarity(sft_layers[i], dpo_layers[i], dim=0)):.6f}"
            for i in sorted(set(sft_layers) & set(dpo_layers))
        ))

        print("  router last-token JS divergence and selected experts:")
        for name in sorted(set(sft_routers) & set(dpo_routers)):
            p = F.softmax(sft_routers[name], dim=-1)
            q = F.softmax(dpo_routers[name], dim=-1)
            midpoint = 0.5 * (p + q)
            js = 0.5 * (
                F.kl_div(midpoint.log(), p, reduction="sum")
                + F.kl_div(midpoint.log(), q, reduction="sum")
            )
            sft_experts = sft_routers[name].topk(sft_config.num_experts_per_tok).indices.tolist()
            dpo_experts = dpo_routers[name].topk(dpo_config.num_experts_per_tok).indices.tolist()
            print(f"    {name}: JS={float(js):.8f} SFT={sft_experts} DPO={dpo_experts}")

        if len(target_ids) > 1:
            forced_ids = torch.cat(
                [input_ids, torch.tensor([[target_ids[0]]], device="cuda")], dim=1
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sft_after = sft(forced_ids).logits[0, -1].float()
                dpo_after = dpo(forced_ids).logits[0, -1].float()
            print(f"  after forcing first target token {tokenizer.decode([target_ids[0]])!r}:")
            print_next_token_report("SFT", tokenizer, sft_after, target_ids[1:], args.top_k)
            print_next_token_report("DPO", tokenizer, dpo_after, target_ids[1:], args.top_k)

    print("\n" + "=" * 88)
    print("target-token parameter-row deltas (percentile among all vocabulary rows):")
    for token_id in target_ids:
        token = tokenizer.decode([token_id]).replace("\n", "\\n")
        embedding_delta, embedding_percentile = row_delta_percentile(
            sft.embed_tokens.weight, dpo.embed_tokens.weight, token_id
        )
        head_delta, head_percentile = row_delta_percentile(
            sft.lm_head.weight, dpo.lm_head.weight, token_id
        )
        print(
            f"  token={token!r} id={token_id}: "
            f"embedding_delta={embedding_delta:.8g} pct={embedding_percentile:.3f}; "
            f"lm_head_delta={head_delta:.8g} pct={head_percentile:.3f}"
        )


if __name__ == "__main__":
    main()
