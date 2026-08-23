#!/usr/bin/env python3
"""Rebuild a native hf_like safetensors dir from a trained Slime DCP checkpoint.

Reads a Slime/Megatron ``iter_<N>`` DCP directory and writes an hf_like dir with
the **native** nanoGPTMoE-v2 parameter names/layout (matching
``step_0002442.pt`` / ``hf_like_v1``), so it can be loaded by
``sglang_models.modern_moe.ModernMoEForCausalLM`` for inference.

Shared experts (that Megatron stores as flat ``[4096,512]`` / ``[512,2048]``)
are reshaped back to native ``[2,2048,512]`` / ``[2,512,1024]``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil

import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from typing_extensions import override


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class EmptyStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        for k, v in metadata.state_dict_metadata.items():
            if "optimizer" in k or "_state" in k:
                continue
            print(f"find {k} in torch_dist ckpt")
            if isinstance(v, dist_cp.metadata.TensorStorageMetadata):
                v = torch.empty(v.size, dtype=v.properties.dtype)  # type: ignore[assignment]
            state_dict[k] = v
        super().set_up_planner(state_dict, metadata, is_coordinator)


def load_dcp_state(input_dir: str) -> dict:
    state_dict = {}
    try:
        dist_cp.state_dict_loader._load_state_dict(
            state_dict,
            storage_reader=WrappedStorageReader(input_dir),
            planner=EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
    except Exception as e:  # pragma: no cover - fallback
        raise RuntimeError(f"DCP load failed: {e}")
    return state_dict


def merge_shared_fc1_to_native(param: torch.Tensor) -> torch.Tensor:
    """[4096,512] (Megatron shared fc1, per-expert [gate,up] flat) -> [2,2048,512] native."""
    hidden = param.shape[1]
    half = param.shape[0] // 2  # 2048 = gates block
    gates = param[:half].reshape(2, -1, hidden)   # [2,1024,512]
    ups = param[half:].reshape(2, -1, hidden)      # [2,1024,512]
    return torch.cat((ups, gates), dim=1).contiguous()  # [2,2048,512] (native [up,gate])


def merge_shared_fc2_to_native(param: torch.Tensor) -> torch.Tensor:
    """[512,2048] -> [2,512,1024] native."""
    inter = param.shape[1] // 2
    per = param.reshape(param.shape[0], 2, inter)   # [512,2,1024]
    return per.permute(1, 0, 2).contiguous()         # [2,512,1024]


def split_qkv(param: torch.Tensor, num_heads: int, num_kv: int, kv_ch: int, hidden: int):
    """Megatron linear_qkv [q+2kv, hidden] -> native q/k/v [., hidden]."""
    q_rows = num_heads * kv_ch
    kv_rows = num_kv * kv_ch
    q = param[:q_rows]
    k = param[q_rows : q_rows + kv_rows]
    v = param[q_rows + kv_rows :]
    return q, k, v


def to_native(sd: dict, num_heads: int, num_kv: int, kv_ch: int, hidden: int, num_shared: int):
    out: dict[str, torch.Tensor] = {}
    out["embed_tokens.weight"] = sd["embedding.word_embeddings.weight"]
    out["norm.weight"] = sd["decoder.final_layernorm.weight"]
    out["lm_head.weight"] = sd["output_layer.weight"]

    n_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("decoder.layers.")) + 1
    for layer in range(n_layers):
        p = f"decoder.layers.{layer}"
        m_key = f"{p}.mlp."
        # attention
        q, k, v = split_qkv(
            sd[f"{p}.self_attention.linear_qkv.weight"], num_heads, num_kv, kv_ch, hidden
        )
        out[f"layers.{layer}.attention.q_proj.weight"] = q.contiguous()
        out[f"layers.{layer}.attention.k_proj.weight"] = k.contiguous()
        out[f"layers.{layer}.attention.v_proj.weight"] = v.contiguous()
        out[f"layers.{layer}.attention.o_proj.weight"] = sd[f"{p}.self_attention.linear_proj.weight"]
        # norms
        out[f"layers.{layer}.input_norm.weight"] = sd[f"{p}.self_attention.linear_qkv.layer_norm_weight"]
        # post-attn norm: MoE layers have pre_mlp_layernorm; dense layer 0 has it fused in linear_fc1
        post_norm_key = f"{p}.pre_mlp_layernorm.weight"
        if post_norm_key in sd:
            out[f"layers.{layer}.post_attention_norm.weight"] = sd[post_norm_key]
        else:
            out[f"layers.{layer}.post_attention_norm.weight"] = sd[f"{m_key}linear_fc1.layer_norm_weight"]
        # MoE
        if f"{m_key}router.weight" in sd:
            out[f"layers.{layer}.moe.router.weight"] = sd[f"{m_key}router.weight"]
            out[f"layers.{layer}.moe.routed.experts.weight"] = sd[
                f"{m_key}experts.experts.linear_fc1.weight"
            ].contiguous()
            out[f"layers.{layer}.moe.routed.output_experts.weight"] = sd[
                f"{m_key}experts.experts.linear_fc2.weight"
            ].contiguous()
            out[f"layers.{layer}.moe.shared.experts.weight"] = merge_shared_fc1_to_native(
                sd[f"{m_key}shared_experts.linear_fc1.weight"]
            )
            out[f"layers.{layer}.moe.shared.output_experts.weight"] = merge_shared_fc2_to_native(
                sd[f"{m_key}shared_experts.linear_fc2.weight"]
            )
        else:
            # dense (layer 0) SwiGLU
            fc1 = sd[f"{m_key}linear_fc1.weight"]
            im = fc1.shape[0] // 2
            out[f"layers.{layer}.moe.ffn.gate_proj.weight"] = fc1[:im].contiguous()
            out[f"layers.{layer}.moe.ffn.up_proj.weight"] = fc1[im:].contiguous()
            out[f"layers.{layer}.moe.ffn.down_proj.weight"] = sd[
                f"{m_key}linear_fc2.weight"
            ].contiguous()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--origin-hf-dir", required=True, help="copy tokenizer/config.json assets")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--kv-ch", type=int, default=64)
    ap.add_argument("--num-shared", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"{args.output_dir} exists; use --force")
    os.makedirs(args.output_dir, exist_ok=True)

    sd = load_dcp_state(args.input_dir)
    native = to_native(sd, args.heads, args.kv_heads, args.kv_ch, args.hidden, args.num_shared)
    print(f"native tensors: {len(native)}")

    import safetensors.torch as sf
    sf.save_file(native, os.path.join(args.output_dir, "model.safetensors"))
    # copy non-weight assets from origin hf dir
    for name in os.listdir(args.origin_hf_dir):
        src = os.path.join(args.origin_hf_dir, name)
        if os.path.isfile(src) and not name.endswith(".safetensors"):
            shutil.copy2(src, os.path.join(args.output_dir, name))
    print(f"wrote hf_like to {args.output_dir}")


if __name__ == "__main__":
    main()
