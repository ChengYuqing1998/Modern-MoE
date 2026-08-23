"""对比 DPO 前后(SFT 权重 vs DPO checkpoint)在相同 prompt 下的生成结果。

用法:
  python scripts/compare_dpo.py \
      --sft checkpoints/posttrain/sft/sft_nanogptmoe_v2_gqa_advanced_kernel_120m_lr3e4_v2/step_0002442.pt \
      --dpo checkpoints/posttrain/dpo/<experiment_id>/dpo_final.pt \
      --prompts 说没说不需说 你叫什么名字 你吃饭了吗 \
      --seed 1337

对每个 prompt,用**相同随机种子**分别加载两个权重生成,打印「SFT:」和「DPO:」
两行,方便直接对比模型有没有真的被人渣化/风格变化。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_moe.config import ModernMoEConfig
from modern_moe.generation import GenerationConfig, generate
from modern_moe.model import ModernMoEForCausalLM


def load_model(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    raw = ck["model_config"]
    if isinstance(raw.get("attention_pattern"), list):
        raw["attention_pattern"] = tuple(raw["attention_pattern"])
    cfg = ModernMoEConfig(**raw)
    model = ModernMoEForCausalLM(cfg)
    model.load_state_dict(ck["model"], strict=True)
    del ck
    model.eval().to(device="cuda", dtype=torch.bfloat16)
    return model, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", type=Path, required=True, help="DPO 前的 SFT 权重")
    ap.add_argument("--dpo", type=Path, required=True, help="DPO 后的 checkpoint")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    print("loading SFT model...")
    sft_model, sft_cfg = load_model(args.sft)
    print("loading DPO model...")
    dpo_model, dpo_cfg = load_model(args.dpo)
    tok = AutoTokenizer.from_pretrained("tokenizer/qwen3_moe", use_fast=True)
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")

    for prompt in args.prompts:
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        input_ids = tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        print("\n" + "=" * 60)
        print(f"PROMPT: {prompt}")
        print("=" * 60)
        for name, model in (("SFT", sft_model), ("DPO", dpo_model)):
            torch.manual_seed(args.seed)
            r = generate(
                model, input_ids,
                GenerationConfig(
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    mode="cache",
                ),
                eos_token_id=stop_id,
            )
            out = tok.decode(
                r.token_ids[0, input_ids.size(1):], skip_special_tokens=True
            )
            print(f"[{name}] {out.strip()}")
    print("\ndone")


if __name__ == "__main__":
    main()
