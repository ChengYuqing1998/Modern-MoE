"""Export a native Modern-MoE checkpoint to an HF-like weight directory.

This is an intermediate artifact for the slime adapter work.  It deliberately
keeps the project's native parameter names and packed_liger layout; it is not
claimed to be directly loadable by Transformers or SGLang until a custom model
implementation is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a native Modern-MoE checkpoint to an HF-like directory."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    return parser.parse_args()


def validate_model_config(config: dict[str, Any]) -> None:
    required = {
        "architecture_name",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "num_experts",
        "num_experts_per_tok",
        "num_shared_experts",
        "moe_parameter_layout",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"model_config is missing required fields: {missing}")
    if config["moe_parameter_layout"] != "packed_liger":
        raise ValueError(
            "This exporter currently requires the validated packed_liger layout; "
            f"got {config['moe_parameter_layout']!r}"
        )


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {args.output_dir}"
        )

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model = checkpoint.get("model")
    model_config = checkpoint.get("model_config")
    if not isinstance(model, dict) or not isinstance(model_config, dict):
        raise ValueError("checkpoint must contain dict fields 'model' and 'model_config'")
    validate_model_config(model_config)

    tensors: dict[str, torch.Tensor] = {}
    for name, tensor in model.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"model[{name!r}] is not a tensor: {type(tensor).__name__}")
        if tensor.layout != torch.strided:
            raise ValueError(f"model[{name!r}] is not a strided tensor")
        tensors[name] = tensor.contiguous()

    args.output_dir.mkdir(parents=True)
    save_file(tensors, str(args.output_dir / "model.safetensors"))

    hf_like_config = {
        # Qwen3MoeConfig is used only as the SGLang config parser.  The
        # architecture remains ModernMoEForCausalLM and is supplied by the
        # external SGLang model package below.
        "model_type": "qwen3_moe",
        "architectures": ["ModernMoEForCausalLM"],
        "torch_dtype": "bfloat16",
        "transformers_compatibility": "custom_model_required",
        "hidden_act": "silu",
        "moe_intermediate_size": model_config["intermediate_size"],
        "attention_bias": False,
        "head_dim": model_config["hidden_size"] // model_config["num_attention_heads"],
        **_jsonable(model_config),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(hf_like_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "native_model_config.json").write_text(
        json.dumps(_jsonable(model_config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tokenizer_path = args.tokenizer_path
    if tokenizer_path is None:
        tokenizer_path = Path(model_config["tokenizer_path"])
    if tokenizer_path.is_dir():
        shutil.copytree(tokenizer_path, args.output_dir / "tokenizer", dirs_exist_ok=True)
        # SGLang/Slime loads tokenizer from --hf-checkpoint directly, so also
        # copy tokenizer files to the export root for AutoTokenizer.
        for item in tokenizer_path.iterdir():
            if item.is_file():
                shutil.copy2(item, args.output_dir / item.name)
    else:
        raise FileNotFoundError(f"tokenizer directory not found: {tokenizer_path}")

    model_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    report = {
        "format": "modern_moe_native_hf_like_v1",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "parameter_tensor_count": len(tensors),
        "parameter_count": sum(t.numel() for t in tensors.values()),
        "parameter_bytes": model_bytes,
        "dtypes": sorted({str(t.dtype) for t in tensors.values()}),
        "model_config": _jsonable(model_config),
        "direct_transformers_load": False,
        "direct_sglang_load": False,
        "next_required_adapters": [
            "custom Transformers config/model registration",
            "native-to-Megatron tensor layout bridge",
            "SGLang custom model and weight loader",
        ],
    }
    (args.output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
