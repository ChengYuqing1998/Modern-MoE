from __future__ import annotations

import argparse
import datetime
import gc
import json
import math
import os
import random
import re
import secrets
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler, Subset

from modern_moe.config import ModernMoEConfig
from modern_moe.data import PackedTokenDataset
from modern_moe.sft_data import SFTPackedDataset
from modern_moe.model import ModernMoEForCausalLM
from modern_moe.thermal import ThermalMonitor
from nanok3.config import NanoK3Config
from nanok3.model import NanoK3ForCausalLM
from modern_moe.generation import GenerationConfig, generate

ModelConfig = ModernMoEConfig | NanoK3Config
LanguageModel = ModernMoEForCausalLM | NanoK3ForCausalLM


class DeterministicResumeSampler(Sampler[int]):
    """Recreate an epoch permutation and seek directly to a batch offset.

    The previous ``shuffle=True`` DataLoader had a deterministic permutation:
    ``manual_seed(seed + epoch)`` followed by DataLoader's one base-seed draw
    and RandomSampler's ``randperm``.  Reproducing that sequence here lets a
    resumed epoch seek directly instead of reading and discarding all earlier
    batches.  The permutation is transient only; it is not stored in a
    checkpoint.
    """

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_index = max(0, int(start_index))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        # DataLoader consumes one generator value for worker base_seed before
        # RandomSampler draws its permutation.  Preserve the old ordering.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        permutation = torch.randperm(len(self.data_source), generator=generator)
        yield from permutation[self.start_index :].tolist()

    def __len__(self) -> int:
        # Keep the full length so steps_per_epoch and global progress remain
        # based on the complete epoch even when the iterator is seeked.
        return len(self.data_source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or SFT Modern-MoE.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_pilot.yaml"),
        help="Training YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the latest checkpoint under --experiment-id.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Explicit experiment directory/W&B run ID.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Model-only initialization checkpoint for a fresh run.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def select_validation_samples(
    dataset: PackedTokenDataset,
    config_path: Path | None,
) -> PackedTokenDataset | Subset:
    if config_path is None:
        return dataset
    value = json.loads(config_path.read_text(encoding="utf-8"))
    indices = value.get("sample_indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(
            f"{config_path} must contain a non-empty sample_indices list"
        )
    if any(not isinstance(index, int) for index in indices):
        raise ValueError(f"{config_path} sample_indices must contain integers")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{config_path} sample_indices must be unique")
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError(
            f"{config_path} sample index outside validation dataset "
            f"[0, {len(dataset)})"
        )
    return Subset(dataset, indices)


@torch.inference_mode()
def generate_training_preview(
    model: LanguageModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    seed: int,
    chat_template: bool = False,
) -> tuple[str, dict[str, float]]:
    """Generate a short diagnostic completion without moving the model.

    The model is already resident on the training GPU.  We only switch its
    mode to eval temporarily, run a small KV-cache decode, and restore train
    mode; no checkpoint reload or device transfer is performed.
    """
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rendered_prompt = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if chat_template
        else prompt
    )
    input_ids = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device, non_blocking=True)
    stop_token_id = (
        tokenizer.convert_tokens_to_ids("<|im_end|>")
        if chat_template
        else tokenizer.eos_token_id
    )
    result = generate(
        model,
        input_ids,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            mode="cache",
        ),
        eos_token_id=stop_token_id,
    )
    completion_ids = result.token_ids[0, input_ids.size(1) :].detach().cpu()
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    if was_training:
        model.train()
    return text, {
        "prefill_seconds": result.prefill_seconds,
        "ttft_seconds": result.time_to_first_token_seconds,
        "decode_seconds": result.decode_seconds,
        "new_tokens": float(result.new_tokens),
    }


def configure_wandb_network(cfg: dict[str, Any]) -> None:
    """Optionally force only this training process to reach W&B directly."""
    if not bool(cfg.get("wandb_disable_proxy", False)):
        return
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(name, None)
    direct_hosts = ("api.wandb.ai", "wandb.ai", ".wandb.ai")
    for name in ("NO_PROXY", "no_proxy"):
        existing = [
            value.strip()
            for value in os.environ.get(name, "").split(",")
            if value.strip()
        ]
        for host in direct_hosts:
            if host not in existing:
                existing.append(host)
        os.environ[name] = ",".join(existing)


def resolve_experiment(
    cfg: dict[str, Any],
    resume_override: Path | None,
) -> tuple[str, Path, bool, Path | None]:
    checkpoint_root = Path(cfg.get("checkpoint_root", "checkpoints/pretrain"))
    stage_subdir = str(cfg.get("checkpoint_subdir", "")).strip()
    if stage_subdir:
        stage_path = Path(stage_subdir)
        if stage_path.is_absolute() or ".." in stage_path.parts:
            raise ValueError("checkpoint_subdir must be a relative safe path")

    def stage_output(experiment_id: str) -> Path:
        output = checkpoint_root / experiment_id
        return output / stage_subdir if stage_subdir else output

    resume_training = bool(cfg.get("resume_training", False))

    if resume_override is not None:
        resume_training = True
        resume_path = resume_override
        configured_id = str(cfg.get("experiment_id", "")).strip()
        if configured_id:
            experiment_id = configured_id
        elif stage_subdir and resume_path.parent.parent != checkpoint_root:
            experiment_id = resume_path.parent.parent.name
        else:
            experiment_id = resume_path.parent.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
            raise ValueError(
                "A valid experiment_id is required when resuming training"
            )
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        output_dir = stage_output(experiment_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        return experiment_id, output_dir, resume_training, resume_path

    if resume_training:
        experiment_id = str(cfg.get("experiment_id", "")).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
            raise ValueError(
                "resume_training=true requires a valid experiment_id "
                "containing only letters, digits, '.', '_' or '-'"
            )
        output_dir = stage_output(experiment_id)
        if not output_dir.is_dir():
            raise FileNotFoundError(
                f"resume_training=true, but experiment checkpoint directory "
                f"does not exist: {output_dir}"
            )
        candidates: list[Path] = []
        final_path = output_dir / "final.pt"
        if final_path.is_file():
            candidates.append(final_path)
        step_candidates: list[tuple[int, Path]] = []
        for path in output_dir.glob("step_*.pt"):
            match = re.fullmatch(r"step_(\d+)\.pt", path.name)
            if match:
                step_candidates.append((int(match.group(1)), path))
        if step_candidates:
            candidates.append(max(step_candidates)[1])
        if not candidates:
            raise FileNotFoundError(
                f"resume_training=true, but no checkpoint exists under "
                f"{output_dir}"
            )
        # A completed earlier stage may leave final.pt in the directory while
        # a newer continuation stage writes step_*.pt. Resume the most recently
        # written checkpoint rather than silently jumping back to old final.pt.
        return experiment_id, output_dir, True, max(
            candidates, key=lambda path: path.stat().st_mtime_ns
        )

    checkpoint_root.mkdir(parents=True, exist_ok=True)
    configured_id = str(cfg.get("experiment_id", "")).strip()
    if configured_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", configured_id):
            raise ValueError(
                "experiment_id may contain only letters, digits, '.', '_' or '-'"
            )
        output_dir = checkpoint_root / configured_id
        if stage_subdir:
            output_dir = output_dir / stage_subdir
        output_dir.mkdir(parents=True, exist_ok=False)
        return configured_id, output_dir, False, None
    while True:
        experiment_id = (
            f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
        )
        output_dir = checkpoint_root / experiment_id
        try:
            output_dir.mkdir()
            output_dir = stage_output(experiment_id)
            output_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    return experiment_id, output_dir, False, None


def build_model_config(path: Path) -> ModelConfig:
    values = read_yaml(path)
    if isinstance(values.get("attention_pattern"), list):
        values["attention_pattern"] = tuple(values["attention_pattern"])
    if values.get("model_type") == "nanoK3":
        return NanoK3Config(**values)
    return ModernMoEConfig(**values)


def build_model(model_config: ModelConfig) -> LanguageModel:
    if isinstance(model_config, NanoK3Config):
        return NanoK3ForCausalLM(model_config)
    return ModernMoEForCausalLM(model_config)


def normalize_checkpoint_model_config(
    values: dict[str, Any],
    current: ModelConfig,
) -> dict[str, Any]:
    """Fill configuration defaults added after an older checkpoint was saved."""
    if not isinstance(values, dict):
        raise ValueError("Checkpoint model_config must be a mapping")
    normalized = dict(values)
    if isinstance(normalized.get("attention_pattern"), list):
        normalized["attention_pattern"] = tuple(normalized["attention_pattern"])
    # These select non-persistent kernels only; they do not describe parameter
    # structure and may safely change when an older checkpoint is resumed.
    for key in ("fused_add_rms_norm", "fused_router", "fused_rope"):
        if hasattr(current, key):
            normalized[key] = getattr(current, key)
    config_type = NanoK3Config if isinstance(current, NanoK3Config) else ModernMoEConfig
    return config_type(**normalized).to_dict()


def describe_run(
    train_config: dict[str, Any],
    model_config: ModelConfig | None,
) -> str:
    explicit = str(train_config.get("run_description", "")).strip()
    if explicit:
        return explicit
    if model_config is None:
        return "训练实验；详细架构和超参数请查看同一 info 区块中的配置。"

    model_values = model_config.to_dict()
    family = (
        "nanoK3"
        if isinstance(model_config, NanoK3Config)
        else str(model_values.get("architecture_name", "传统 GPT 解码器"))
    )
    architecture = (
        f"{family}改造的稀疏 MoE"
        if model_values.get("use_moe", not isinstance(model_config, NanoK3Config))
        else f"{family} Dense"
    )
    if isinstance(model_config, NanoK3Config):
        architecture = "nanoK3 混合注意力 MoE"

    details = [
        f"这是一个{architecture}预训练 run",
        (
            f"模型为 {model_values.get('num_hidden_layers')} 层、"
            f"hidden size {model_values.get('hidden_size')}"
        ),
    ]
    if model_values.get("use_moe", False):
        dense_prefix = int(model_values.get("first_k_dense_replace", 0))
        if dense_prefix:
            details.append(
                f"前 {dense_prefix} 层使用 Dense SwiGLU，intermediate size 为 "
                f"{model_values.get('dense_intermediate_size')}"
            )
        details.append(
            f"其余 MoE 层从 {model_values.get('num_experts')} 个路由专家中选择 "
            f"Top-{model_values.get('num_experts_per_tok')}，另有 "
            f"{model_values.get('num_shared_experts')} 个始终激活的共享专家，"
            f"expert intermediate size 为 {model_values.get('intermediate_size')}"
        )
    elif isinstance(model_config, NanoK3Config):
        details.append(
            f"每层使用 Top-{model_values.get('num_experts_per_token')} 路由，"
            f"路由专家数为 {model_values.get('num_experts')}，共享专家数为 "
            f"{model_values.get('num_shared_experts')}"
        )
    else:
        details.append(
            f"Dense SwiGLU intermediate size 为 "
            f"{model_values.get('intermediate_size')}"
        )

    attention_pattern = model_values.get("attention_pattern")
    if attention_pattern:
        details.append(
            f"注意力模式为 {list(attention_pattern)}，"
            f"全注意力后端为 {model_values.get('full_attention_backend')}"
        )
    details.append(
        "输入输出词嵌入"
        + ("共享权重" if model_values.get("tie_word_embeddings") else "不共享权重")
    )

    schedule = str(train_config.get("lr_schedule", "cosine")).lower()
    peak_lr = train_config.get("learning_rate")
    if schedule == "constant":
        details.append(f"使用固定学习率 {peak_lr}")
    else:
        warmup_start = train_config.get("warmup_start_lr")
        min_lr = train_config.get("min_learning_rate")
        lr_text = f"使用 {schedule} 学习率调度，峰值学习率 {peak_lr}"
        if warmup_start is not None:
            lr_text += f"，从 {warmup_start} warmup"
        lr_text += f"，warmup {train_config.get('warmup_steps', 0)} 步"
        if min_lr is not None:
            lr_text += f"，最低学习率 {min_lr}"
        details.append(lr_text)

    micro_batch = train_config.get("micro_batch_size")
    accumulation = train_config.get("gradient_accumulation_steps")
    if micro_batch is not None and accumulation is not None:
        details.append(
            f"micro batch 为 {micro_batch}，梯度累积 {accumulation} 次，"
            f"单卡有效 batch 为 {int(micro_batch) * int(accumulation)} 条序列"
        )
    details.append(
        f"上下文长度 {train_config.get('sequence_length')}，"
        f"计划训练 {train_config.get('epochs')} 个 epoch，"
        f"数据目录为 {train_config.get('data_dir')}"
    )
    details.append(
        f"W&B project 为 {train_config.get('wandb_project')}，"
        f"run name 为 {train_config.get('run_name')}"
    )
    return "；".join(details) + "。"


def checkpoint_info(
    train_config: dict[str, Any],
    model_config: ModelConfig | None,
) -> dict[str, Any]:
    is_nanok3 = isinstance(model_config, NanoK3Config)
    model_values = model_config.to_dict() if model_config is not None else None
    return {
        "format_version": 1,
        "description": describe_run(train_config, model_config),
        "model": {
            "family": (
                "nanoK3"
                if is_nanok3
                else "ModernMoEForCausalLM" if model_config is not None else None
            ),
            "implementation": (
                "nanok3/model.py"
                if is_nanok3
                else "modern_moe/model.py" if model_config is not None else None
            ),
            "layers_implementation": (
                "nanok3/layers.py"
                if is_nanok3
                else "modern_moe/layers.py" if model_config is not None else None
            ),
            "config_path": str(train_config.get("model_config", "")),
            "config": model_values,
        },
        "training": {
            "entrypoint": str(
                train_config.get("training_entrypoint", "scripts/train.py")
            ),
            "config_path": str(train_config.get("training_config_path", "")),
            "config": dict(train_config),
        },
        "wandb": {
            "project": train_config.get("wandb_project"),
            "entity": train_config.get("wandb_entity"),
            "run_id": train_config.get("experiment_id"),
            "run_name": train_config.get("run_name"),
        },
    }


def forward_for_training(
    model: LanguageModel,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> Any:
    if isinstance(model, NanoK3ForCausalLM):
        return model(input_ids=input_ids)
    linear_ce_impl = str(getattr(model, "training_linear_ce_impl", "pytorch"))
    return model(
        input_ids=input_ids,
        mtp_targets=targets,
        return_loss_hidden_states=linear_ce_impl != "pytorch",
        linear_ce_impl=linear_ce_impl,
    )


def output_scalar(output: Any, name: str, reference: torch.Tensor) -> torch.Tensor:
    value = getattr(output, name, None)
    return reference.new_zeros(()) if value is None else value.float()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_hms(seconds: float) -> str:
    """Format a duration with hours as the largest unit (hours may exceed 24)."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def learning_rate_metrics_and_lines(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, float], str]:
    """Return named LR metrics and compact scientific-notation text."""
    metrics: dict[str, float] = {}
    lines: list[str] = []
    ordinary_index = 0
    for group_index, group in enumerate(optimizer.param_groups):
        value = float(group["lr"])
        if bool(group.get("router_group", False)):
            metric_name = "router_learning_rate"
            line_name = "router_lr"
        else:
            ordinary_index += 1
            metric_name = (
                "learning_rate"
                if ordinary_index == 1
                else f"learning_rate_group_{group_index}"
            )
            line_name = "lr" if ordinary_index == 1 else f"lr_group_{group_index}"
        metrics[metric_name] = value
        lines.append(f"{line_name}={value:.2e}")
    return metrics, " ".join(lines)


def causal_losses(
    output: Any,
    targets: torch.Tensor,
    router_aux_coef: float,
    router_z_coef: float,
    mtp_loss_coef: float,
    all_targets_valid: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Targets are already shifted by PackedTokenDataset, so all 2048 logits
    # participate in the loss.
    # This function runs inside CUDA autocast.  Cross entropy promotes its
    # reduction internally for numerical stability; explicitly converting the
    # full [batch, sequence, vocabulary] logits to FP32 would materialize an
    # additional ~2.32 GiB tensor for the current training shape and can OOM
    # on the next gradient-accumulation microbatch.
    if output.logits is not None:
        lm_loss = F.cross_entropy(
            output.logits.reshape(-1, output.logits.size(-1)),
            targets.reshape(-1),
        )
    else:
        linear_ce_impl = str(
            getattr(output, "linear_ce_impl", "torch_compile")
        )
        if all_targets_valid and linear_ce_impl == "torch_compile":
            # PackedTokenDataset has no ignore_index entries. Calling CCE's
            # public wrapper would nevertheless execute nonzero() to discover
            # valid tokens, whose dynamic output shape is not graph-capturable.
            # Enter the same compiled CE implementation after the redundant
            # scan; this is mathematically identical when every target is valid.
            from cut_cross_entropy.torch_compile import (
                torch_compile_linear_cross_entropy_apply,
            )

            lm_loss, _ = torch_compile_linear_cross_entropy_apply(
                output.loss_hidden_states.flatten(0, -2),
                output.classifier_weight,
                targets.flatten(),
                ignore_index=-100,
                reduction="mean",
                return_lse=False,
            )
        elif all_targets_valid and linear_ce_impl == "cce":
            # Mirror cce_linear_cross_entropy after its dynamic valid-token
            # scan. PackedTokenDataset guarantees that every target is valid,
            # so valids=None is the exact static-shape representation needed
            # by CUDA Graph capture.
            from cut_cross_entropy.cce import (
                CCEParams,
                _handle_eps,
                linear_cross_entropy_apply,
            )

            embeddings = output.loss_hidden_states.contiguous().flatten(0, -2)
            flat_targets = targets.contiguous().flatten()
            if flat_targets.data_ptr() % 16 != 0:
                raise RuntimeError("CUDA Graph CCE targets must be 16-byte aligned")
            filter_eps = _handle_eps(
                "auto",
                torch.get_autocast_dtype("cuda")
                if torch.is_autocast_enabled()
                else embeddings.dtype,
            )
            params = CCEParams(
                targets=flat_targets,
                valids=None,
                softcap=None,
                reduction="mean",
                filter_eps=filter_eps,
                shift=0,
                batch_shape=targets.size(),
                accum_e_fp32=False,
                accum_c_fp32=False,
                filter_e_grad=True,
                filter_c_grad=True,
                vocab_parallel_options=None,
                return_lse=False,
            )
            lm_loss, _ = linear_cross_entropy_apply(
                embeddings,
                output.classifier_weight,
                None,
                params,
            )
        else:
            from cut_cross_entropy import linear_cross_entropy

            lm_loss = linear_cross_entropy(
                output.loss_hidden_states,
                output.classifier_weight,
                targets,
                impl=linear_ce_impl,
            )
    mtp_loss = output_scalar(output, "mtp_loss", lm_loss)
    router_aux_loss = output_scalar(output, "router_aux_loss", lm_loss)
    router_z_loss = output_scalar(output, "router_z_loss", lm_loss)
    loss = (
        lm_loss
        + router_aux_coef * router_aux_loss
        + router_z_coef * router_z_loss
        + mtp_loss_coef * mtp_loss
    )
    return loss, lm_loss, mtp_loss


class CUDAGraphedMicrobatch:
    """Capture one fixed-shape training forward/backward for replay."""

    def __init__(
        self,
        model: LanguageModel,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        sequence_length: int,
        accumulation_steps: int,
        dtype: torch.dtype,
        router_aux_coef: float,
        router_z_coef: float,
        mtp_loss_coef: float,
        warmup_steps: int = 3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.dtype = dtype
        self.router_aux_coef = router_aux_coef
        self.router_z_coef = router_z_coef
        self.mtp_loss_coef = mtp_loss_coef
        shape = (batch_size, sequence_length)
        self.input_ids = torch.zeros(shape, device="cuda", dtype=torch.long)
        self.targets = torch.zeros_like(self.input_ids)
        self.metric_sums = torch.zeros(5, device="cuda", dtype=torch.float32)
        self.output = None
        self.losses = None

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_steps):
                optimizer.zero_grad(set_to_none=True)
                self._forward_backward(accumulate_metrics=False)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        # Release warmup-only activation blocks before the graph creates its
        # private memory pool. Parameters and optimizer state remain resident.
        gc.collect()
        torch.cuda.empty_cache()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output, self.losses = self._forward_backward(
                accumulate_metrics=True
            )
        # Capture executes once. Discard its dummy gradients and metrics while
        # preserving the graph-owned gradient tensor addresses.
        optimizer.zero_grad(set_to_none=False)
        self.metric_sums.zero_()
        torch.cuda.synchronize()

    def _forward_backward(self, accumulate_metrics: bool):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            output = forward_for_training(
                self.model, self.input_ids, self.targets
            )
            loss, lm_loss, mtp_loss = causal_losses(
                output,
                self.targets,
                self.router_aux_coef,
                self.router_z_coef,
                self.mtp_loss_coef,
                all_targets_valid=True,
            )
            router_aux_loss = output_scalar(output, "router_aux_loss", loss)
            router_z_loss = output_scalar(output, "router_z_loss", loss)
            if accumulate_metrics:
                with torch.no_grad():
                    self.metric_sums.add_(
                        torch.stack(
                            (
                                loss,
                                lm_loss,
                                mtp_loss,
                                router_aux_loss,
                                router_z_loss,
                            )
                        ).float()
                    )
            scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        return output, (
            loss,
            lm_loss,
            mtp_loss,
            router_aux_loss,
            router_z_loss,
        )

    def replay(self, input_ids: torch.Tensor, targets: torch.Tensor):
        if input_ids.shape != self.input_ids.shape or targets.shape != self.targets.shape:
            raise ValueError(
                "CUDA Graph requires every microbatch to have the configured "
                f"fixed shape {tuple(self.input_ids.shape)}"
            )
        self.input_ids.copy_(input_ids, non_blocking=True)
        self.targets.copy_(targets, non_blocking=True)
        self.graph.replay()
        return self.output, self.losses

    def consume_metric_sums(self) -> tuple[float, ...]:
        values = tuple(float(value) for value in self.metric_sums.cpu())
        self.metric_sums.zero_()
        return values


class CUDAGraphedOptimizerUpdate:
    """Capture N fixed microbatches, clipping, and fused AdamW as one graph."""

    def __init__(
        self,
        model: LanguageModel,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        sequence_length: int,
        accumulation_steps: int,
        dtype: torch.dtype,
        router_aux_coef: float,
        router_z_coef: float,
        mtp_loss_coef: float,
        max_grad_norm: float,
        warmup_steps: int = 3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.dtype = dtype
        self.router_aux_coef = router_aux_coef
        self.router_z_coef = router_z_coef
        self.mtp_loss_coef = mtp_loss_coef
        shape = (batch_size, sequence_length)
        self.input_ids = [
            torch.zeros(shape, device="cuda", dtype=torch.long)
            for _ in range(accumulation_steps)
        ]
        self.targets = [torch.zeros_like(value) for value in self.input_ids]
        self.metric_sums = torch.zeros(5, device="cuda", dtype=torch.float32)
        self.output = None
        self.losses = None

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for index in range(warmup_steps):
                optimizer.zero_grad(set_to_none=True)
                self._forward_backward(
                    self.input_ids[index % accumulation_steps],
                    self.targets[index % accumulation_steps],
                    accumulate_metrics=False,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=False)

        for group in optimizer.param_groups:
            group["capturable"] = True
        for state in optimizer.state.values():
            step = state.get("step")
            if torch.is_tensor(step) and not step.is_cuda:
                state["step"] = step.to(device="cuda")
        if not optimizer.state:
            # Materialize AdamW moments and device step counters without
            # changing parameters, then restore the mathematical step to zero.
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            for group in optimizer.param_groups:
                group["lr"] = 0.0
            optimizer.step()
            for group, learning_rate in zip(
                optimizer.param_groups, learning_rates
            ):
                group["lr"] = learning_rate
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()
        optimizer.zero_grad(set_to_none=False)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            optimizer.zero_grad(set_to_none=False)
            for input_ids, targets in zip(self.input_ids, self.targets):
                self.output, self.losses = self._forward_backward(
                    input_ids, targets, accumulate_metrics=True
                )
            self.grad_norm = clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()
        torch.cuda.synchronize()

    def _forward_backward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        accumulate_metrics: bool,
    ):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            output = forward_for_training(self.model, input_ids, targets)
            loss, lm_loss, mtp_loss = causal_losses(
                output,
                targets,
                self.router_aux_coef,
                self.router_z_coef,
                self.mtp_loss_coef,
                all_targets_valid=True,
            )
            router_aux_loss = output_scalar(output, "router_aux_loss", loss)
            router_z_loss = output_scalar(output, "router_z_loss", loss)
            if accumulate_metrics:
                with torch.no_grad():
                    self.metric_sums.add_(
                        torch.stack(
                            (
                                loss,
                                lm_loss,
                                mtp_loss,
                                router_aux_loss,
                                router_z_loss,
                            )
                        ).float()
                    )
            scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        return output, (
            loss,
            lm_loss,
            mtp_loss,
            router_aux_loss,
            router_z_loss,
        )

    def stage(
        self,
        index: int,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        if input_ids.shape != self.input_ids[index].shape:
            raise ValueError("full-update CUDA Graph requires fixed batch shape")
        self.input_ids[index].copy_(input_ids, non_blocking=True)
        self.targets[index].copy_(targets, non_blocking=True)

    def replay(self):
        self.graph.replay()
        return self.output, self.losses, self.grad_norm

    def consume_metric_sums(self) -> tuple[float, ...]:
        values = tuple(float(value) for value in self.metric_sums.cpu())
        self.metric_sums.zero_()
        return values


def lr_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    warmup_start_ratio: float | None = None,
    min_lr_ratio: float = 0.0,
) -> float:
    if warmup_start_ratio is not None and not 0.0 <= warmup_start_ratio <= 1.0:
        raise ValueError("warmup_start_ratio must be in [0, 1]")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if step < warmup_steps:
        if warmup_start_ratio is None:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step) / float(max(1, warmup_steps))
        return warmup_start_ratio + (1.0 - warmup_start_ratio) * progress
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    total_steps: int,
) -> LambdaLR:
    schedule = str(cfg.get("lr_schedule", "cosine")).strip().lower()
    if schedule == "constant":
        return LambdaLR(optimizer, lambda _step: 1.0)
    if schedule != "cosine":
        raise ValueError(
            f"Unsupported lr_schedule={schedule!r}; expected 'cosine' or 'constant'"
        )
    peak_lr = float(cfg["learning_rate"])
    configured_warmup_start = cfg.get("warmup_start_lr")
    warmup_start_lr = (
        float(configured_warmup_start)
        if configured_warmup_start is not None
        else None
    )
    min_lr = float(cfg.get("min_learning_rate", 0.0))
    if warmup_start_lr is not None and not 0.0 <= warmup_start_lr <= peak_lr:
        raise ValueError("warmup_start_lr must be in [0, learning_rate]")
    if not 0.0 <= min_lr <= peak_lr:
        raise ValueError("min_learning_rate must be in [0, learning_rate]")
    return LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step,
            int(cfg["warmup_steps"]),
            total_steps,
            warmup_start_lr / peak_lr if warmup_start_lr is not None else None,
            min_lr / peak_lr,
        ),
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    next_batch_index: int,
    optimizer_step: int,
    micro_step: int,
    tokens_seen: int,
    train_config: dict[str, Any],
    model_config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "tokens_seen": tokens_seen,
            "train_config": train_config,
            "model_config": model_config.to_dict(),
            "info": checkpoint_info(train_config, model_config),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def _checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def save_managed_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    next_batch_index: int,
    optimizer_step: int,
    micro_step: int,
    tokens_seen: int,
    train_config: dict[str, Any],
    model_config: ModelConfig,
    *,
    validation_loss: float | None,
    best_validation_loss: float,
    best_validation_step: int | None,
    max_checkpoints: int,
) -> None:
    """Save one step and retain best/latest plus the most recent checkpoints."""
    if max_checkpoints < 2:
        raise ValueError("max_checkpoints must be at least 2")

    path = output_dir / f"step_{optimizer_step:07d}.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch,
        next_batch_index,
        optimizer_step,
        micro_step,
        tokens_seen,
        {
            **train_config,
            "best_validation_loss": best_validation_loss,
            "best_validation_step": best_validation_step,
        },
        model_config,
    )

    manifest_path = output_dir / "checkpoint_manifest.json"
    records: dict[int, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for record in manifest.get("checkpoints", []):
                records[int(record["step"])] = record
        except (OSError, ValueError, KeyError, TypeError):
            records = {}

    previous = records.get(optimizer_step, {})
    records[optimizer_step] = {
        "file": path.name,
        "step": optimizer_step,
        "saved_at": datetime.datetime.now(
            datetime.timezone.utc
        ).astimezone().isoformat(timespec="seconds"),
        "validation_loss": (
            validation_loss
            if validation_loss is not None
            else previous.get("validation_loss")
        ),
    }

    existing_steps = {
        step
        for candidate in output_dir.glob("step_*.pt")
        if (step := _checkpoint_step(candidate)) is not None
    }
    latest_step = max(existing_steps)
    protected_steps = {latest_step}
    if best_validation_step in existing_steps:
        protected_steps.add(best_validation_step)

    # max_checkpoints is a strict limit on physical step files. Always keep
    # latest and best, then evict the oldest checkpoint that has neither role.
    # A former best immediately becomes eligible once a newer best is chosen.
    retained_steps = set(existing_steps)
    while len(retained_steps) > max_checkpoints:
        evict_step = next(
            (
                step
                for step in sorted(retained_steps)
                if step not in protected_steps
            ),
            None,
        )
        if evict_step is None:
            raise RuntimeError(
                "checkpoint retention cannot satisfy max_checkpoints: "
                "all checkpoint files are protected"
            )
        (output_dir / f"step_{evict_step:07d}.pt").unlink()
        retained_steps.remove(evict_step)
        records.pop(evict_step, None)

    # A managed step is now durable, so an old untracked final.pt would only
    # consume a fifth full checkpoint without participating in retention.
    legacy_final = output_dir / "final.pt"
    if legacy_final.is_file():
        legacy_final.unlink()

    checkpoints = []
    for step in sorted(retained_steps, reverse=True):
        record = records.get(
            step,
            {
                "file": f"step_{step:07d}.pt",
                "step": step,
                "saved_at": None,
                "validation_loss": None,
            },
        )
        roles = []
        if step == latest_step:
            roles.append("latest")
        if step == best_validation_step:
            roles.append("best")
        record["roles"] = roles
        checkpoints.append(record)

    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(
            {
                "max_checkpoints": max_checkpoints,
                "latest_step": latest_step,
                "best_step": best_validation_step,
                "best_validation_loss": (
                    best_validation_loss
                    if math.isfinite(best_validation_loss)
                    else None
                ),
                "info": checkpoint_info(train_config, model_config),
                "checkpoints": checkpoints,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)


@torch.no_grad()
def evaluate(
    model: LanguageModel,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
    router_aux_coef: float,
    router_z_coef: float,
    mtp_loss_coef: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "lm_loss_sum": 0.0,
        "supervised_tokens": 0.0,
        "mtp_loss": 0.0,
        "aux": 0.0,
        "z": 0.0,
    }
    count = 0

    for input_ids, targets in loader:
        if count >= max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            output = forward_for_training(model, input_ids, targets)
            loss, lm_loss, mtp_loss = causal_losses(
                output,
                targets,
                router_aux_coef,
                router_z_coef,
                mtp_loss_coef,
            )
        valid_tokens = int(targets.ne(-100).sum())
        totals["lm_loss_sum"] += float(lm_loss) * valid_tokens
        totals["supervised_tokens"] += valid_tokens
        totals["mtp_loss"] += float(mtp_loss)
        totals["aux"] += float(output_scalar(output, "router_aux_loss", loss))
        totals["z"] += float(output_scalar(output, "router_z_loss", loss))
        count += 1

    model.train()
    if count == 0:
        return {
            "validation/loss": float("nan"),
            "validation/lm_loss": float("nan"),
            "validation/mtp_loss": float("nan"),
            "validation/aux": float("nan"),
            "validation/z": float("nan"),
            "validation/supervised_tokens": 0.0,
        }
    supervised_tokens = max(totals["supervised_tokens"], 1.0)
    lm_loss = totals["lm_loss_sum"] / supervised_tokens
    mtp_loss = totals["mtp_loss"] / count
    aux = totals["aux"] / count
    z = totals["z"] / count
    return {
        "validation/loss": (
            lm_loss
            + mtp_loss_coef * mtp_loss
            + router_aux_coef * aux
            + router_z_coef * z
        ),
        "validation/lm_loss": lm_loss,
        "validation/mtp_loss": mtp_loss,
        "validation/aux": aux,
        "validation/z": z,
        "validation/supervised_tokens": totals["supervised_tokens"],
    }


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    if args.resume is not None and args.resume_latest:
        raise ValueError("use either --resume or --resume-latest, not both")
    if args.experiment_id is not None:
        cfg["experiment_id"] = args.experiment_id
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = str(args.init_checkpoint)
    if args.resume_latest:
        if not str(cfg.get("experiment_id", "")).strip():
            raise ValueError("--resume-latest requires --experiment-id")
        cfg["resume_training"] = True
    cfg["training_config_path"] = str(args.config)
    experiment_id, output_dir, resume_training, resume_path = resolve_experiment(
        cfg, args.resume
    )
    # Persist the resolved identity in W&B metadata and every checkpoint even
    # when a new run generated it automatically.
    cfg["experiment_id"] = experiment_id
    cfg["resume_training"] = resume_training
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Fix the NVIDIA driver/runtime before training."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16 training.")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_config_path = Path(cfg["model_config"])
    model_config = build_model_config(model_config_path)
    sequence_length = int(cfg["sequence_length"])
    if sequence_length > model_config.max_position_embeddings:
        raise ValueError(
            "sequence_length exceeds model_config.max_position_embeddings"
        )

    data_dir = Path(cfg["data_dir"])
    dataset_format = str(cfg.get("dataset_format", "pretraining")).lower()
    if dataset_format == "sft":
        train_dataset = SFTPackedDataset(
            data_dir / "train.input_ids.bin",
            data_dir / "train.labels.bin",
            sequence_length,
        )
        full_validation_dataset = SFTPackedDataset(
            data_dir / "validation.input_ids.bin",
            data_dir / "validation.labels.bin",
            sequence_length,
        )
        if bool(cfg.get("cuda_graph", False)):
            raise ValueError(
                "SFT assistant-only labels are not compatible with the current "
                "all-targets-valid CUDA Graph path; set cuda_graph=false"
            )
        if int(getattr(model_config, "num_mtp_layers", 0)):
            raise ValueError("SFT currently requires num_mtp_layers=0")
    elif dataset_format == "pretraining":
        train_dataset = PackedTokenDataset(
            data_dir / "train.bin",
            data_dir / "train.sample_idx.npy",
            sequence_length,
        )
        full_validation_dataset = PackedTokenDataset(
            data_dir / "validation.bin",
            data_dir / "validation.sample_idx.npy",
            sequence_length,
        )
    else:
        raise ValueError("dataset_format must be 'pretraining' or 'sft'")
    validation_sample_config = cfg.get("validation_sample_config")
    validation_dataset = select_validation_samples(
        full_validation_dataset,
        Path(validation_sample_config) if validation_sample_config else None,
    )

    loader_kwargs = {
        "batch_size": int(cfg["micro_batch_size"]),
        "num_workers": int(cfg.get("num_workers", 2)),
        "pin_memory": True,
        "persistent_workers": int(cfg.get("num_workers", 2)) > 0,
    }
    train_sampler = DeterministicResumeSampler(train_dataset, seed)
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=False, **loader_kwargs
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_kwargs
    )
    configured_eval_batches = cfg.get("eval_batches", "auto")
    if configured_eval_batches is None or str(configured_eval_batches).lower() == "auto":
        eval_batches = len(validation_loader)
    else:
        eval_batches = int(configured_eval_batches)
        if eval_batches < 1:
            raise ValueError("eval_batches must be positive or 'auto'")
        eval_batches = min(eval_batches, len(validation_loader))
    cfg["resolved_eval_batches"] = eval_batches
    configured_final_eval_batches = cfg.get("final_eval_batches", "auto")
    if (
        configured_final_eval_batches is None
        or str(configured_final_eval_batches).lower() == "auto"
    ):
        final_eval_batches = len(validation_loader)
    else:
        final_eval_batches = int(configured_final_eval_batches)
        if final_eval_batches < 1:
            raise ValueError("final_eval_batches must be positive or 'auto'")
        final_eval_batches = min(final_eval_batches, len(validation_loader))
    cfg["resolved_final_eval_batches"] = final_eval_batches
    cfg["validation_samples_per_eval"] = min(
        len(validation_dataset),
        eval_batches * int(cfg["micro_batch_size"]),
    )

    model = build_model(model_config).to(device=device, dtype=dtype)
    moe_training_impl = str(cfg.get("moe_training_impl", "reference")).strip().lower()
    if moe_training_impl not in {"reference", "scattermoe", "liger"}:
        raise ValueError(
            "moe_training_impl must be 'reference', 'scattermoe', or 'liger'"
        )
    if moe_training_impl in {"scattermoe", "liger"}:
        try:
            if moe_training_impl == "scattermoe":
                import scattermoe  # noqa: F401
            else:
                os.environ.setdefault("LIGER_FUSED_MOE_AUTOTUNE", "1")
                import liger_kernel  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                f"moe_training_impl={moe_training_impl} requires its package"
            ) from error
    if moe_training_impl == "scattermoe":
        from modern_moe.layers import SparseMoE

        for module in model.modules():
            if isinstance(module, SparseMoE):
                module.use_scattermoe_training = True
    if moe_training_impl == "liger" and getattr(
        model_config, "moe_parameter_layout", None
    ) != "packed_liger":
        raise ValueError(
            "moe_training_impl=liger requires moe_parameter_layout=packed_liger"
        )
    print(f"moe_training_impl={moe_training_impl}", flush=True)
    model.training_linear_ce_impl = str(
        cfg.get("linear_cross_entropy_impl", "pytorch")
    ).strip().lower()
    if model.training_linear_ce_impl != "pytorch":
        try:
            import cut_cross_entropy  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "linear_cross_entropy_impl requires cut-cross-entropy"
            ) from error
    print(
        f"linear_cross_entropy_impl={model.training_linear_ce_impl}",
        flush=True,
    )
    if bool(cfg.get("gradient_checkpointing", True)):
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError(
                "gradient_checkpointing is requested, but the model does not "
                "implement gradient_checkpointing_enable()."
            )
        model.gradient_checkpointing_enable()

    init_checkpoint_value = str(cfg.get("init_checkpoint", "")).strip()
    if init_checkpoint_value and resume_path is not None:
        print(
            "resume checkpoint supplied; ignoring fresh-run init_checkpoint=",
            init_checkpoint_value,
            flush=True,
        )
    if init_checkpoint_value and resume_path is None:
        init_checkpoint = Path(init_checkpoint_value)
        if not init_checkpoint.is_file():
            raise FileNotFoundError(init_checkpoint)
        initial_state = torch.load(
            init_checkpoint, map_location="cpu", weights_only=False
        )
        initial_model_config = dict(initial_state.get("model_config", {}))
        initial_layout = initial_model_config.get("moe_parameter_layout", "legacy")
        current_layout = getattr(model_config, "moe_parameter_layout", "legacy")
        if initial_layout != current_layout:
            if current_layout != "packed_liger" or initial_layout not in {
                "legacy", "packed_scattermoe"
            }:
                raise ValueError(
                    f"cannot initialize {current_layout!r} model from "
                    f"{initial_layout!r} checkpoint"
                )
            from scripts.convert_checkpoint_to_packed_liger import (
                _swap_model_routed_gate_up,
            )
            from scripts.convert_checkpoint_to_packed_scattermoe import (
                convert_model_state,
                model_state_names,
            )

            conversion_values = dict(initial_model_config)
            if isinstance(conversion_values.get("attention_pattern"), list):
                conversion_values["attention_pattern"] = tuple(
                    conversion_values["attention_pattern"]
                )
            conversion_config = ModernMoEConfig(**conversion_values)
            packed_state_names = model_state_names(
                conversion_values, "packed_liger"
            )
            if initial_layout == "legacy":
                initial_state["model"] = convert_model_state(
                    initial_state["model"], packed_state_names, conversion_config
                )
            _swap_model_routed_gate_up(initial_state["model"], packed_state_names)
            initial_model_config["moe_parameter_layout"] = "packed_liger"
            initial_state["model_config"] = initial_model_config
            print(
                f"converted initialization weights in memory: "
                f"{initial_layout} -> packed_liger (optimizer state ignored)",
                flush=True,
            )
        normalized_initial_config = normalize_checkpoint_model_config(
            initial_state.get("model_config"), model_config
        )
        if normalized_initial_config != model_config.to_dict():
            differences = {
                key: {
                    "checkpoint": normalized_initial_config.get(key),
                    "current": model_config.to_dict().get(key),
                }
                for key in sorted(set(normalized_initial_config) | set(model_config.to_dict()))
                if normalized_initial_config.get(key) != model_config.to_dict().get(key)
            }
            raise ValueError(
                f"init_checkpoint model config mismatch: {differences}"
            )
        model.load_state_dict(initial_state["model"], strict=True)
        del initial_state
        print(f"initialized model weights from {init_checkpoint}", flush=True)

    generation_every_steps = int(cfg.get("generation_every_steps", 0))
    generation_prompt = str(
        cfg.get("generation_prompt", "梯度下降是一种")
    )
    generation_max_new_tokens = int(cfg.get("generation_max_new_tokens", 64))
    generation_temperature = float(cfg.get("generation_temperature", 0.8))
    generation_top_k = int(cfg.get("generation_top_k", 50))
    generation_top_p = float(cfg.get("generation_top_p", 0.95))
    generation_repetition_penalty = float(
        cfg.get("generation_repetition_penalty", 1.05)
    )
    generation_seed = int(cfg.get("generation_seed", 20260811))
    generation_chat_template = bool(cfg.get("generation_chat_template", False))
    generation_tokenizer = None
    if generation_every_steps > 0:
        generation_tokenizer = AutoTokenizer.from_pretrained(
            model_config.tokenizer_path,
            use_fast=True,
        )

    thermal_monitor = ThermalMonitor(cfg)
    thermal_monitor.start()

    router_lr_scale = float(cfg.get("router_learning_rate_scale", 1.0))
    if not 0.0 < router_lr_scale <= 1.0:
        raise ValueError("router_learning_rate_scale must be in (0, 1]")
    freeze_router = bool(cfg.get("freeze_router", False))
    if freeze_router or router_lr_scale < 1.0:
        router_parameters = []
        other_parameters = []
        for name, parameter in model.named_parameters():
            if "router" in name:
                router_parameters.append(parameter)
                if freeze_router:
                    parameter.requires_grad_(False)
            else:
                other_parameters.append(parameter)
        optimizer_parameters = [{"params": other_parameters, "lr": float(cfg["learning_rate"])}]
        if not freeze_router:
            optimizer_parameters.append(
                {
                    "params": router_parameters,
                    "lr": float(cfg["learning_rate"]) * router_lr_scale,
                    "router_group": True,
                }
            )
        print(
            f"router_parameters={sum(p.numel() for p in router_parameters):,} "
            + ("frozen" if freeze_router else f"learning_rate_scale={router_lr_scale:g}"),
            flush=True,
        )
    else:
        optimizer_parameters = model.parameters()
    optimizer = AdamW(
        optimizer_parameters,
        lr=float(cfg["learning_rate"]),
        betas=tuple(float(x) for x in cfg.get("betas", [0.9, 0.95])),
        eps=float(cfg.get("adam_epsilon", 1.0e-8)),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        fused=True,
    )

    accumulation_steps = int(cfg["gradient_accumulation_steps"])
    epochs = int(cfg["epochs"])
    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = steps_per_epoch * epochs
    scheduler_total_steps = int(cfg.get("scheduler_total_steps", total_steps))
    # These are finalized after loading a checkpoint, because a resumed run
    # may start in the middle of an epoch.  They intentionally depend on the
    # actual dataset and loader settings rather than hard-coded phase totals.
    expected_total_steps = total_steps
    stage_start_step = 0
    stage_total_steps = total_steps

    scheduler = build_lr_scheduler(optimizer, cfg, scheduler_total_steps)

    optimizer_step = 0
    micro_step = 0
    tokens_seen = 0
    start_epoch = 0
    resume_batch_index = 0
    best_validation_loss = float("inf")
    best_validation_step: int | None = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint_model_config = checkpoint.get("model_config")
        try:
            normalized_checkpoint_config = normalize_checkpoint_model_config(
                checkpoint_model_config, model_config
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid checkpoint model_config in {resume_path}: {error}"
            ) from error
        current_model_config = model_config.to_dict()
        if normalized_checkpoint_config != current_model_config:
            differences = {
                key: {
                    "checkpoint": normalized_checkpoint_config.get(key),
                    "current": current_model_config.get(key),
                }
                for key in sorted(
                    set(normalized_checkpoint_config) | set(current_model_config)
                )
                if normalized_checkpoint_config.get(key)
                != current_model_config.get(key)
            }
            raise ValueError(
                f"Checkpoint model_config does not match the current model "
                f"config: {resume_path}; differences={differences}"
            )
        checkpoint_experiment_id = checkpoint.get("train_config", {}).get(
            "experiment_id"
        )
        if checkpoint_experiment_id not in {None, experiment_id}:
            raise ValueError(
                f"Checkpoint experiment_id={checkpoint_experiment_id!r} does "
                f"not match current experiment_id={experiment_id!r}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        reset_scheduler = bool(cfg.get("reset_scheduler_on_new_stage", False))
        stage_id = str(cfg.get("scheduler_stage_id", "")).strip()
        checkpoint_stage_id = str(
            checkpoint.get("train_config", {}).get("scheduler_stage_id", "")
        ).strip()
        if reset_scheduler and not stage_id:
            raise ValueError(
                "reset_scheduler_on_new_stage=true requires scheduler_stage_id"
            )
        if reset_scheduler and checkpoint_stage_id != stage_id:
            peak_lr = float(cfg["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = peak_lr
                group["initial_lr"] = peak_lr
            scheduler = build_lr_scheduler(
                optimizer, cfg, scheduler_total_steps
            )
            print(
                f"starting scheduler stage {stage_id!r}: peak_lr={peak_lr:g} "
                f"lr_schedule={str(cfg.get('lr_schedule', 'cosine'))} "
                f"warmup_steps={int(cfg['warmup_steps'])} "
                f"scheduler_total_steps={scheduler_total_steps}",
                flush=True,
            )
        else:
            scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer_step = int(checkpoint["optimizer_step"])
        micro_step = int(checkpoint.get("micro_step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        start_epoch = int(checkpoint.get("epoch", 0))
        resume_batch_index = int(checkpoint.get("next_batch_index", 0))
        saved_train_config = checkpoint.get("train_config", {})
        best_validation_loss = float(
            saved_train_config.get("best_validation_loss", float("inf"))
        )
        saved_best_step = saved_train_config.get("best_validation_step")
        best_validation_step = (
            int(saved_best_step) if saved_best_step is not None else None
        )
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    # Compute the global endpoint from the actual loader and resume position.
    # This works for arbitrary dataset sizes and for checkpoints taken partway
    # through an epoch: only batches not yet consumed contribute to the
    # remaining optimizer updates.
    remaining_steps = 0
    remaining_micro_batches_total = 0
    for remaining_epoch in range(start_epoch, epochs):
        first_batch = resume_batch_index if remaining_epoch == start_epoch else 0
        remaining_batches = max(0, len(train_loader) - first_batch)
        remaining_micro_batches_total += remaining_batches
        remaining_steps += math.ceil(remaining_batches / accumulation_steps)
    expected_total_steps = optimizer_step + remaining_steps
    # Stage progress means optimizer updates since this curriculum stage
    # began, not updates since the current process was restarted.  A resumed
    # checkpoint stores the next DataLoader batch, so the already-consumed
    # stage updates can be recovered without a hard-coded phase total.
    consumed_stage_steps = (
        math.ceil(resume_batch_index / accumulation_steps)
        if resume_path is not None
        else 0
    )
    stage_start_step = max(0, optimizer_step - consumed_stage_steps)
    stage_epochs_remaining = max(0, epochs - start_epoch)
    stage_total_steps = steps_per_epoch * stage_epochs_remaining

    def progress(step: int) -> str:
        stage_step = max(0, step - stage_start_step)
        return (
            f"{step}/{expected_total_steps} "
            f"stage_step={stage_step}/{stage_total_steps}"
        )

    try:
        configure_wandb_network(cfg)
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "wandb is not installed. Install the environment requirements first."
        ) from error

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", cfg["wandb_project"]),
        entity=os.environ.get("WANDB_ENTITY") or cfg.get("wandb_entity"),
        id=experiment_id,
        name=cfg.get("run_name"),
        config={
            **cfg,
            "model": model_config.to_dict(),
            "parameter_count": parameter_count,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "full_validation_samples": len(full_validation_dataset),
            # The training loop resumes at a later epoch for curriculum
            # stages, so expose the global expected endpoint rather than
            # epochs * steps_per_epoch (which would count already completed
            # stages again).
            "total_optimizer_steps": expected_total_steps,
            "stage_total_optimizer_steps": stage_total_steps,
            "scheduler_total_steps": scheduler_total_steps,
        },
        resume="allow" if resume_training else "never",
        settings=wandb.Settings(
            finish_timeout=float(cfg.get("wandb_finish_timeout", 120.0)),
            finish_timeout_raises=False,
        ),
    )
    # W&B's internal `_step` counts wandb.log() calls, not optimizer updates.
    # Register explicit semantic axes so charts cannot confuse logging events
    # with training progress.
    run.define_metric("train/optimizer_step")
    run.define_metric("train/micro_step")
    run.define_metric(
        "train/*",
        step_metric="train/optimizer_step",
    )
    run.define_metric(
        "validation/*",
        step_metric="train/optimizer_step",
    )
    run.define_metric(
        "micro/*",
        step_metric="train/micro_step",
    )
    run.define_metric(
        "system/*",
        step_metric="train/micro_step",
    )
    run.define_metric("generation/optimizer_step")
    run.define_metric(
        "generation/*",
        step_metric="generation/optimizer_step",
    )
    wandb.log(
        {
            "run/parameter_count": parameter_count,
            "run/train_samples": len(train_dataset),
            "run/validation_samples": len(validation_dataset),
            "run/full_validation_samples": len(full_validation_dataset),
            "train/tokens_seen": tokens_seen,
            "train/optimizer_step": optimizer_step,
            "train/micro_step": micro_step,
        }
    )
    print(
        f"W&B run: {run.url}\n"
        f"experiment_id={experiment_id} checkpoint_dir={output_dir} "
        f"resume={resume_training} checkpoint={resume_path}\n"
        f"optimizer_progress={progress(optimizer_step)} "
        f"expected_total_optimizer_steps={expected_total_steps}\n"
        f"parameters={parameter_count:,} train_samples={len(train_dataset):,} "
        f"micro_batch={loader_kwargs['batch_size']} "
        f"gradient_accumulation={accumulation_steps}",
        flush=True,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_every = int(cfg["log_every_steps"])
    micro_log_every = int(cfg.get("micro_log_every", 1))
    eval_every = int(cfg["eval_every_steps"])
    save_every = int(cfg["save_every_steps"])
    max_checkpoints = int(cfg.get("max_checkpoints", 4))
    if max_checkpoints < 2:
        raise ValueError("max_checkpoints must be at least 2")
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    aux_coef = float(getattr(model_config, "router_aux_loss_coef", 0.0))
    z_coef = float(getattr(model_config, "router_z_loss_coef", 0.0))
    mtp_coef = float(getattr(model_config, "mtp_loss_coef", 0.0))
    use_cuda_graph = bool(cfg.get("cuda_graph", False))
    cuda_graph_runner = None
    cuda_update_runner = None
    if use_cuda_graph:
        if device.type != "cuda":
            raise ValueError("cuda_graph=true requires a CUDA device")
        if bool(cfg.get("gradient_checkpointing", True)):
            raise ValueError(
                "cuda_graph=true currently requires gradient_checkpointing=false"
            )
        full_update_graph = bool(cfg.get("cuda_graph_full_update", False))
        print(
            "warming and capturing fixed-shape CUDA Graph "
            f"batch={loader_kwargs['batch_size']} sequence={sequence_length} "
            f"full_update={full_update_graph}...",
            flush=True,
        )
        runner_kwargs = dict(
            model=model,
            optimizer=optimizer,
            batch_size=loader_kwargs["batch_size"],
            sequence_length=sequence_length,
            accumulation_steps=accumulation_steps,
            dtype=dtype,
            router_aux_coef=aux_coef,
            router_z_coef=z_coef,
            mtp_loss_coef=mtp_coef,
            warmup_steps=int(cfg.get("cuda_graph_warmup_steps", 3)),
        )
        if full_update_graph:
            if str(cfg.get("lr_schedule", "cosine")).lower() != "constant":
                raise ValueError(
                    "cuda_graph_full_update=true currently requires a constant LR"
                )
            cuda_update_runner = CUDAGraphedOptimizerUpdate(
                **runner_kwargs,
                max_grad_norm=max_grad_norm,
            )
        else:
            cuda_graph_runner = CUDAGraphedMicrobatch(**runner_kwargs)
        print("CUDA Graph capture complete", flush=True)
    interval_started = time.perf_counter()
    interval_tokens = 0
    interval_sums = {
        "loss": 0.0,
        "lm": 0.0,
        "mtp": 0.0,
        "aux": 0.0,
        "z": 0.0,
    }
    interval_micro_batches = 0
    timing_started = time.perf_counter()
    training_started = timing_started
    timing_micro_batches = 0
    total_timed_micro_batches = 0
    # Keep these names defined so the final cleanup also handles an empty
    # loader without conditional deletion.
    input_ids = targets = output = None
    loss = lm_loss = mtp_loss = scaled_loss = None
    thermal_stop = False
    last_epoch = start_epoch
    last_next_batch_index = resume_batch_index

    for epoch in range(start_epoch, epochs):
        # Deterministic per-epoch ordering also permits direct seeking on
        # resume; no earlier batches need to be read and discarded.
        resume_offset = resume_batch_index if epoch == start_epoch else 0
        train_sampler.set_epoch(epoch, resume_offset * loader_kwargs["batch_size"])
        for batch_index, (input_ids, targets) in enumerate(train_loader):
            global_batch_index = batch_index + resume_offset
            optimizer_already_stepped = False
            full_update_batch = cuda_update_runner is not None
            if full_update_batch:
                slot = micro_step % accumulation_steps
                cuda_update_runner.stage(slot, input_ids, targets)
                batch_tokens = input_ids.numel()
                tokens_seen += batch_tokens
                interval_tokens += batch_tokens
                interval_micro_batches += 1
                micro_step += 1
                timing_micro_batches += 1
                total_timed_micro_batches += 1
                is_last_batch = global_batch_index + 1 == len(train_loader)
                should_step = (
                    micro_step % accumulation_steps == 0 or is_last_batch
                )
                if not should_step:
                    continue
                if slot + 1 == accumulation_steps:
                    output, graph_losses, grad_norm = (
                        cuda_update_runner.replay()
                    )
                else:
                    # Preserve the final incomplete accumulation group instead
                    # of padding it with dummy samples.
                    optimizer.zero_grad(set_to_none=False)
                    cuda_update_runner.metric_sums.zero_()
                    for partial_index in range(slot + 1):
                        output, graph_losses = (
                            cuda_update_runner._forward_backward(
                                cuda_update_runner.input_ids[partial_index],
                                cuda_update_runner.targets[partial_index],
                                accumulate_metrics=True,
                            )
                        )
                    grad_norm = clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                (
                    loss,
                    lm_loss,
                    mtp_loss,
                    router_aux_loss,
                    router_z_loss,
                ) = graph_losses
                scaled_loss = loss / accumulation_steps
                graphed_batch = True
                optimizer_already_stepped = True
            else:
                graphed_batch = (
                    cuda_graph_runner is not None
                    and input_ids.shape == cuda_graph_runner.input_ids.shape
                    and targets.shape == cuda_graph_runner.targets.shape
                )
            if not full_update_batch and graphed_batch:
                output, graph_losses = cuda_graph_runner.replay(
                    input_ids, targets
                )
                (
                    loss,
                    lm_loss,
                    mtp_loss,
                    router_aux_loss,
                    router_z_loss,
                ) = graph_losses
                scaled_loss = loss / accumulation_steps
            elif not full_update_batch:
                input_ids = input_ids.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    output = forward_for_training(model, input_ids, targets)
                    loss, lm_loss, mtp_loss = causal_losses(
                        output,
                        targets,
                        aux_coef,
                        z_coef,
                        mtp_coef,
                    )
                    scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                router_aux_loss = output_scalar(
                    output, "router_aux_loss", loss
                )
                router_z_loss = output_scalar(output, "router_z_loss", loss)

            if not full_update_batch:
                batch_tokens = input_ids.numel()
                tokens_seen += batch_tokens
                interval_tokens += batch_tokens
                interval_micro_batches += 1
                micro_step += 1
                timing_micro_batches += 1
                total_timed_micro_batches += 1
            if not graphed_batch:
                interval_sums["loss"] += float(loss.detach())
                interval_sums["lm"] += float(lm_loss.detach())
                interval_sums["mtp"] += float(mtp_loss.detach())
                interval_sums["aux"] += float(router_aux_loss.detach())
                interval_sums["z"] += float(router_z_loss.detach())

            should_log_micro = micro_step == 1 or (
                micro_log_every > 0 and micro_step % micro_log_every == 0
            )
            if should_log_micro:
                # CUDA Graph replay is asynchronous. Measure the completed
                # wall-clock window and divide by every microbatch submitted
                # since the previous synchronization; timing a single replay
                # here would incorrectly charge it for the whole queued graph.
                torch.cuda.synchronize()
                now = time.perf_counter()
                timing_elapsed = max(now - timing_started, 1.0e-6)
                micro_seconds = timing_elapsed / max(timing_micro_batches, 1)
                remaining_micro_batches = max(
                    0, remaining_micro_batches_total - total_timed_micro_batches
                )
                completed_fraction = total_timed_micro_batches / max(
                    remaining_micro_batches_total, 1
                )
                average_seconds = (now - training_started) / max(
                    total_timed_micro_batches, 1
                )
                eta_seconds = average_seconds * remaining_micro_batches
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                micro_metrics = {
                    "micro/loss": float(loss.detach()),
                    "micro/lm_loss": float(lm_loss.detach()),
                    "micro/mtp_loss": float(mtp_loss.detach()),
                    "micro/router_aux_loss": float(router_aux_loss.detach()),
                    "micro/router_z_loss": float(router_z_loss.detach()),
                    "micro/seconds": micro_seconds,
                    "micro/tokens_per_second": batch_tokens
                    / max(micro_seconds, 1.0e-6),
                    "train/eta_seconds": eta_seconds,
                    "train/stage_progress_fraction": completed_fraction,
                    "system/cuda_memory_allocated_gib": memory_allocated,
                    "system/cuda_memory_reserved_gib": memory_reserved,
                    "system/cuda_peak_memory_allocated_gib": (
                        torch.cuda.max_memory_allocated() / (1024**3)
                    ),
                    "train/tokens_seen": tokens_seen,
                    "train/micro_step": micro_step,
                    "train/optimizer_step": optimizer_step,
                }
                lr_metrics, lr_lines = learning_rate_metrics_and_lines(optimizer)
                micro_metrics.update(
                    {f"micro/{name}": value for name, value in lr_metrics.items()}
                )
                wandb.log(micro_metrics)
                print(
                    f"micro_step={micro_step} optimizer_step={progress(optimizer_step)} "
                    f"loss={micro_metrics['micro/loss']:.4f} "
                    f"{lr_lines} "
                    f"avg_micro={micro_seconds * 1000:.2f}ms "
                    f"eta={format_hms(eta_seconds)} "
                    f"memory={memory_allocated:.2f}/{memory_reserved:.2f} GiB",
                    flush=True,
                )
                timing_started = time.perf_counter()
                timing_micro_batches = 0

            is_last_batch = global_batch_index + 1 == len(train_loader)
            should_step = micro_step % accumulation_steps == 0 or is_last_batch
            if not should_step:
                continue

            if not optimizer_already_stepped:
                grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            scheduler.step()
            if cuda_update_runner is None:
                optimizer.zero_grad(set_to_none=cuda_graph_runner is None)
            optimizer_step += 1
            last_epoch = epoch
            last_next_batch_index = global_batch_index + 1

            if optimizer_step % log_every == 0:
                metric_runner = cuda_update_runner or cuda_graph_runner
                if metric_runner is not None:
                    graph_metric_sums = metric_runner.consume_metric_sums()
                    for name, value in zip(
                        ("loss", "lm", "mtp", "aux", "z"),
                        graph_metric_sums,
                    ):
                        interval_sums[name] += value
                elapsed = max(time.perf_counter() - interval_started, 1.0e-6)
                divisor = max(interval_micro_batches, 1)
                lr_metrics, lr_lines = learning_rate_metrics_and_lines(optimizer)
                optimizer_metrics = {
                        "train/loss": interval_sums["loss"] / divisor,
                        "train/lm_loss": interval_sums["lm"] / divisor,
                        "train/mtp_loss": interval_sums["mtp"] / divisor,
                        "train/router_aux_loss": interval_sums["aux"] / divisor,
                        "train/router_z_loss": interval_sums["z"] / divisor,
                        "train/grad_norm": float(grad_norm),
                        "train/tokens_per_second": interval_tokens / elapsed,
                        "train/tokens_seen": tokens_seen,
                        "train/epoch": epoch,
                        "train/optimizer_step": optimizer_step,
                }
                optimizer_metrics.update(
                    {f"train/{name}": value for name, value in lr_metrics.items()}
                )
                wandb.log(optimizer_metrics)
                interval_started = time.perf_counter()
                interval_tokens = 0
                interval_sums = {
                    "loss": 0.0,
                    "lm": 0.0,
                    "mtp": 0.0,
                    "aux": 0.0,
                    "z": 0.0,
                }
                interval_micro_batches = 0

            validation_loss_for_checkpoint: float | None = None
            improved_best_this_step = False
            if eval_every > 0 and optimizer_step % eval_every == 0:
                metrics = evaluate(
                    model,
                    validation_loader,
                    device,
                    dtype,
                    eval_batches,
                    aux_coef,
                    z_coef,
                    mtp_coef,
                )
                metrics["train/optimizer_step"] = optimizer_step
                wandb.log(metrics)
                validation_loss = float(metrics["validation/loss"])
                print(
                    f"validation complete at optimizer_step={progress(optimizer_step)} "
                    f"validation_loss={validation_loss:.4f} "
                    f"validation_lm_loss="
                    f"{float(metrics['validation/lm_loss']):.4f} "
                    f"validation_aux="
                    f"{float(metrics['validation/aux']):.4f} "
                    f"validation_z="
                    f"{float(metrics['validation/z']):.4f}",
                    flush=True,
                )
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_validation_step = optimizer_step
                    improved_best_this_step = True
                validation_loss_for_checkpoint = validation_loss

            periodic_save_this_step = (
                save_every > 0 and optimizer_step % save_every == 0
            )
            if improved_best_this_step or periodic_save_this_step:
                save_managed_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_batch_index + 1,
                    optimizer_step,
                    micro_step,
                    tokens_seen,
                    cfg,
                    model_config,
                    validation_loss=validation_loss_for_checkpoint,
                    best_validation_loss=best_validation_loss,
                    best_validation_step=best_validation_step,
                    max_checkpoints=max_checkpoints,
                )
                if improved_best_this_step:
                    print(
                        f"best checkpoint saved at optimizer_step="
                        f"{progress(optimizer_step)} validation_loss="
                        f"{validation_loss_for_checkpoint:.4f}",
                        flush=True,
                    )

            if (
                generation_every_steps > 0
                and optimizer_step % generation_every_steps == 0
                and generation_tokenizer is not None
            ):
                checkpoint_name = f"step_{optimizer_step:07d}.pt"
                try:
                    preview, timing = generate_training_preview(
                        model,
                        generation_tokenizer,
                        generation_prompt,
                        device,
                        generation_max_new_tokens,
                        generation_temperature,
                        generation_top_k,
                        generation_top_p,
                        generation_repetition_penalty,
                        generation_seed,
                        generation_chat_template,
                    )
                    # Table preserves the text as a durable W&B record while
                    # scalar timing/step fields remain easy to plot.
                    wandb.log({
                        "generation/optimizer_step": optimizer_step,
                        "generation/prefill_seconds": timing["prefill_seconds"],
                        "generation/ttft_seconds": timing["ttft_seconds"],
                        "generation/decode_seconds": timing["decode_seconds"],
                        "generation/new_tokens": timing["new_tokens"],
                        "generation/sample": wandb.Table(
                            columns=["optimizer_step", "checkpoint", "prompt", "text"],
                            data=[[optimizer_step, checkpoint_name, generation_prompt, preview]],
                        ),
                    })
                    print(
                        f"generation checkpoint={checkpoint_name} "
                        f"optimizer_step={optimizer_step} "
                        f"prefill={timing['prefill_seconds']:.3f}s "
                        f"new_tokens={int(timing['new_tokens'])}\n"
                        f"generation prompt={generation_prompt!r}\n"
                        f"generation output={preview!r}",
                        flush=True,
                    )
                except Exception as error:
                    # Generation must never terminate a long-running training
                    # job; retain an explicit diagnostic in the log/W&B.
                    print(
                        f"generation failed checkpoint={checkpoint_name}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    wandb.log({
                        "generation/optimizer_step": optimizer_step,
                        "generation/checkpoint": checkpoint_name,
                        "generation/error": f"{type(error).__name__}: {error}",
                    })

            if thermal_monitor.stop_requested:
                thermal_stop = True
                break

        if thermal_stop:
            break

    if thermal_stop:
        thermal_path = output_dir / f"thermal_stop_step_{optimizer_step:07d}.pt"
        thermal_config = {
            **cfg,
            "thermal_stop_reason": thermal_monitor.reason,
        }
        save_checkpoint(
            thermal_path,
            model,
            optimizer,
            scheduler,
            last_epoch,
            last_next_batch_index,
            optimizer_step,
            micro_step,
            tokens_seen,
            thermal_config,
            model_config,
        )
        wandb.log({
            "system/thermal_stop": 1,
            "system/thermal_stop_optimizer_step": optimizer_step,
        })
        print(
            f"thermal stop checkpoint saved: {thermal_path} "
            f"optimizer_step={optimizer_step} reason={thermal_monitor.reason}",
            flush=True,
        )
        thermal_monitor.stop()
        return

    # The last optimizer step is not necessarily divisible by eval_every.
    # Always evaluate the exact final weights so the latest checkpoint has a
    # comparable validation loss instead of ending with validation_loss=null.
    final_metrics = evaluate(
        model,
        validation_loader,
        device,
        dtype,
        final_eval_batches,
        aux_coef,
        z_coef,
        mtp_coef,
    )
    final_metrics["train/optimizer_step"] = optimizer_step
    final_metrics["validation/is_final"] = 1
    wandb.log(final_metrics)
    final_validation_loss = float(final_metrics["validation/loss"])
    if final_validation_loss < best_validation_loss:
        best_validation_loss = final_validation_loss
        best_validation_step = optimizer_step

    save_managed_checkpoint(
        output_dir,
        model,
        optimizer,
        scheduler,
        epochs,
        0,
        optimizer_step,
        micro_step,
        tokens_seen,
        cfg,
        model_config,
        validation_loss=final_validation_loss,
        best_validation_loss=best_validation_loss,
        best_validation_step=best_validation_step,
        max_checkpoints=max_checkpoints,
    )
    print(
        f"final validation complete at optimizer_step={progress(optimizer_step)} "
        f"validation_loss={final_validation_loss:.4f} "
        f"validation_lm_loss="
        f"{float(final_metrics['validation/lm_loss']):.4f}",
        flush=True,
    )
    thermal_monitor.stop()
    # W&B's uploader is asynchronous during training, but run.finish() must
    # flush pending data and can block on a dead VPN/proxy. Release every CUDA
    # owner first so a network stall never leaves the GPU occupied.
    input_ids = targets = output = None
    loss = lm_loss = mtp_loss = scaled_loss = None
    model = optimizer = scheduler = None
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "training complete; final training state saved as a managed step "
        "checkpoint and CUDA resources "
        f"released before W&B sync (allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB)",
        flush=True,
    )
    run.finish()


if __name__ == "__main__":
    main()
