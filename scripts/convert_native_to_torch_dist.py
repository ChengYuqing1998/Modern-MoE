"""Convert the native Modern-MoE checkpoint into slime's torch_dist format.

The native checkpoint intentionally keeps the project's packed expert layout,
so it cannot use slime's standard HuggingFace model dispatch.  This converter
builds the custom MCore model and maps every native tensor explicitly.
"""

from __future__ import annotations

import gc
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.checkpointing import (
    get_checkpoint_name,
    get_checkpoint_tracker_filename,
    save_checkpoint,
)
from megatron.training.training import get_model

from slime.backends.megatron_utils.hf_to_megatron.common import (
    SafetensorReader,
    load_model_hf_weights,
    merge_gate_up,
    merge_qkv,
)
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.utils.logging_utils import configure_logger

from third_party.slime.tools.convert_hf_to_torch_dist import get_args


def _native_reader_tensor(reader: SafetensorReader, name: str) -> torch.Tensor:
    return reader.get_tensor(name)


def _layer_tensor(reader: SafetensorReader, layer: int, suffix: str) -> torch.Tensor:
    return _native_reader_tensor(reader, f"layers.{layer}.{suffix}")


def _native_to_megatron_name(name: str) -> str:
    name = name.removeprefix("vp_stages.0.")
    while name.startswith("module."):
        name = name.removeprefix("module.")
    name = name.removeprefix("language_model.")
    return name


def get_native_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    """Return the native tensor corresponding to one MCore parameter."""
    name = _native_to_megatron_name(name)

    if name == "embedding.word_embeddings.weight":
        return reader.get_tensor("embed_tokens.weight")
    if name == "decoder.final_layernorm.weight":
        return reader.get_tensor("norm.weight")
    if name == "output_layer.weight":
        return reader.get_tensor("lm_head.weight")

    if not name.startswith("decoder.layers."):
        raise KeyError(f"Unsupported Megatron parameter: {name}")

    layer_text, suffix = name.removeprefix("decoder.layers.").split(".", 1)
    layer = int(layer_text)

    direct = {
        "input_layernorm.weight": "input_norm.weight",
        # Current Megatron decoder spec folds the pre-attention RMSNorm into the
        # QKV projection (`linear_qkv.layer_norm_weight`).  The native model
        # applies exactly one RMSNorm before attention, `input_norm`, so alias it.
        "self_attention.linear_qkv.layer_norm_weight": "input_norm.weight",
        "pre_mlp_layernorm.weight": "post_attention_norm.weight",
        "self_attention.linear_proj.weight": "attention.o_proj.weight",
        "mlp.router.weight": "moe.router.weight",
    }
    if suffix in direct:
        return _layer_tensor(reader, layer, direct[suffix])

    # Dense (layer 0) SwiGLU: current Megatron fuses the pre-MLP RMSNorm into
    # linear_fc1.layer_norm_weight.  The native model applies `post_attention_norm`
    # before the dense MLP, so alias to it instead of the routed-expert path below.
    if suffix == "mlp.linear_fc1.layer_norm_weight":
        return _layer_tensor(reader, layer, "post_attention_norm.weight")

    if suffix == "self_attention.linear_qkv.weight":
        q = _layer_tensor(reader, layer, "attention.q_proj.weight")
        k = _layer_tensor(reader, layer, "attention.k_proj.weight")
        v = _layer_tensor(reader, layer, "attention.v_proj.weight")
        return merge_qkv(q, k, v, config)

    if suffix == "mlp.linear_fc1.weight":
        gate = _layer_tensor(reader, layer, "moe.ffn.gate_proj.weight")
        up = _layer_tensor(reader, layer, "moe.ffn.up_proj.weight")
        return merge_gate_up(gate, up)
    if suffix == "mlp.linear_fc2.weight":
        return _layer_tensor(reader, layer, "moe.ffn.down_proj.weight")

    if suffix.startswith("mlp.experts.local_experts."):
        expert_text, weight_name = suffix.removeprefix("mlp.experts.local_experts.").split(".", 1)
        expert = int(expert_text)
        packed = _layer_tensor(reader, layer, "moe.routed.experts.weight")
        output = _layer_tensor(reader, layer, "moe.routed.output_experts.weight")
        if weight_name == "linear_fc1.weight":
            per_expert = packed[expert]
            intermediate = per_expert.shape[0] // 2
            return per_expert
        if weight_name == "linear_fc2.weight":
            return output[expert]

    if suffix == "mlp.shared_experts.linear_fc1.weight":
        packed = _layer_tensor(reader, layer, "moe.shared.experts.weight")
        intermediate = packed.shape[1] // 2
        # packed_liger stores shared experts as [up, gate].  The native
        # inference path swaps these halves before applying SwiGLU.
        ups = packed[:, :intermediate]
        gates = packed[:, intermediate:]
        return merge_gate_up(gates.reshape(-1, gates.shape[-1]), ups.reshape(-1, ups.shape[-1]))
    if suffix == "mlp.shared_experts.linear_fc2.weight":
        output = _layer_tensor(reader, layer, "moe.shared.output_experts.weight")
        return torch.cat((output[0], output[1]), dim=1).contiguous()

    raise KeyError(f"Unsupported Megatron parameter: {name}")


def _maybe_dump_forward_trace(model) -> None:
    """Run one deterministic MCore prefill for native-vs-training comparison."""
    output_path = os.environ.get("SLIME_FORWARD_TRACE_OUTPUT")
    input_path = os.environ.get("SLIME_FORWARD_TRACE_INPUT")
    if not output_path or not input_path:
        return
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    input_ids = payload["input_ids"].reshape(1, -1).cuda().long()
    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).reshape(1, -1)
    module = model[0]
    module.eval()
    captures = {}
    hooks = []
    if os.environ.get("SLIME_FORWARD_TRACE_INTERMEDIATE") == "1":
        def capture(name):
            def hook(_module, _inputs, output):
                value = output
                if isinstance(value, (tuple, list)):
                    value = next((item for item in value if torch.is_tensor(item)), None)
                if torch.is_tensor(value):
                    captures[name] = value.detach().float().cpu()
            return hook

        for name, child in module.named_modules():
            if os.environ.get("SLIME_FORWARD_TRACE_LIST_MODULES") == "1":
                print(f"TRACE_MODULE {name} {child.__class__.__name__}")
            if (
                name.endswith("embedding")
                or name.endswith("final_layernorm")
                or ("layers." in name and name.rsplit(".", 1)[-1].isdigit())
            ):
                hooks.append(child.register_forward_hook(capture(name)))
    with torch.inference_mode():
        output = module(input_ids=input_ids, position_ids=position_ids, attention_mask=None, labels=None)
    for hook in hooks:
        hook.remove()
    if isinstance(output, (tuple, list)):
        output = next(item for item in output if torch.is_tensor(item))
    if hasattr(output, "logits"):
        output = output.logits
    torch.save(
        {
            "input_ids": input_ids[0].cpu(),
            "mcore_logits": output.detach().float().cpu(),
            "intermediate": captures,
        },
        output_path,
    )
    print(f"saved MCore forward trace: {output_path}, shape={tuple(output.shape)}")


def main() -> None:
    configure_logger()
    world_size = int(os.getenv("WORLD_SIZE") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or 0)
    global_rank = int(os.getenv("RANK") or 0)
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    args = get_args()
    init(args)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    config = SimpleNamespace(
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_query_groups,
        head_dim=args.hidden_size // args.num_attention_heads,
    )
    reader = SafetensorReader(args.hf_checkpoint)
    load_model_hf_weights(
        args,
        model,
        args.hf_checkpoint,
        config,
        lambda name, _reader, _config: get_native_tensor(name, reader, config),
    )
    _maybe_dump_forward_trace(model)
    print(f"Native weights loaded: {args.hf_checkpoint}")
    gc.collect()
    torch.cuda.empty_cache()
    save_checkpoint(1, model, None, None, 0)
    if dist.get_rank() == 0:
        with open(get_checkpoint_tracker_filename(args.save), "w", encoding="utf-8") as handle:
            handle.write("release")
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        os.replace(source_dir, target_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
