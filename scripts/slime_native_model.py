"""Megatron model provider for the native nanoGPTMoE-v2 architecture.

This module is imported inside the slime container through
``--custom-model-provider-path``.  It intentionally builds the standard
Megatron GPTModel first; checkpoint loading is handled by the project-specific
converter after its parameter layout has been audited.
"""

from __future__ import annotations

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.global_vars import get_args


def _maybe_dump_structure(model) -> None:
    """Dump the instantiated MCore parameter tree for adapter development."""
    import json
    import os

    output_path = os.environ.get("SLIME_DUMP_MODEL_KEYS")
    if not output_path:
        return
    entries = []
    for name, tensor in model.named_parameters():
        entries.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)})
    for name, tensor in model.named_buffers():
        entries.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "buffer": True})
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
    raise SystemExit(f"wrote Megatron model structure to {output_path}")


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
) -> GPTModel:
    """Build the nanoGPTMoE-v2-compatible Megatron GPT model."""
    # Megatron stores the parsed Namespace globally before invoking the model
    # provider. Keep this provider close to slime's default path so its
    # checkpoint and parallelism machinery remains usable.
    args = get_args()
    config = core_transformer_config_from_args(args)
    # The native Modern-MoE checkpoints use ordinary PyTorch RMSNorm weights.
    # Megatron's persistent LayerNorm path is incompatible with the
    # WrappedTorchNorm used below and would fail during model construction.
    config.persist_layer_norm = False
    if vp_stage is not None:
        config.vp_stage = vp_stage

    layer_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=args.transformer_impl == "transformer_engine",
        normalization="RMSNorm",
        vp_stage=vp_stage,
    )
    # The local (non-Transformer-Engine) block spec defaults its final norm to
    # FusedLayerNorm even when the decoder layers use RMSNorm.  Our native
    # checkpoint uses RMSNorm throughout, so use the PyTorch norm wrapper for
    # the final layer as well.
    layer_spec.layer_norm = WrappedTorchNorm
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        vp_stage=vp_stage,
    )
    _maybe_dump_structure(model)
    return model
