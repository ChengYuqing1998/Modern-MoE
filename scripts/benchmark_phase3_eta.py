"""Estimate Phase 3 wall time without W&B, evaluation, or checkpoints."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from modern_moe.config import ModernMoEConfig
from modern_moe.data import PackedTokenDataset
from modern_moe.model import ModernMoEForCausalLM
from scripts.train import CUDAGraphedMicrobatch, DeterministicResumeSampler, format_hms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_nanogptmoe_v2_gqa_phase3_full.yaml"),
    )
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measure-updates", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulation", type=int, default=None)
    args = parser.parse_args()

    train_cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model_path = Path(train_cfg["model_config"])
    model_values = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    model_values["attention_pattern"] = tuple(model_values["attention_pattern"])
    config = ModernMoEConfig(**model_values)
    data_dir = Path(train_cfg["data_dir"])
    dataset = PackedTokenDataset(
        data_dir / "train.bin",
        data_dir / "train.sample_idx.npy",
        int(train_cfg["sequence_length"]),
    )
    sampler = DeterministicResumeSampler(dataset, int(train_cfg["seed"]))
    sampler.set_epoch(2)
    batch_size = args.batch_size or int(train_cfg["micro_batch_size"])
    accumulation = args.accumulation or int(
        train_cfg["gradient_accumulation_steps"]
    )
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=True,
        persistent_workers=int(train_cfg.get("num_workers", 2)) > 0,
    )
    model = ModernMoEForCausalLM(config).to("cuda", dtype=torch.bfloat16).train()
    model.training_linear_ce_impl = str(train_cfg["linear_cross_entropy_impl"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        betas=tuple(train_cfg["betas"]),
        eps=float(train_cfg["adam_epsilon"]),
        weight_decay=float(train_cfg["weight_decay"]),
        fused=True,
    )
    graph = CUDAGraphedMicrobatch(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        sequence_length=int(train_cfg["sequence_length"]),
        accumulation_steps=accumulation,
        dtype=torch.bfloat16,
        router_aux_coef=config.router_aux_loss_coef,
        router_z_coef=config.router_z_loss_coef,
        mtp_loss_coef=config.mtp_loss_coef,
        warmup_steps=int(train_cfg.get("cuda_graph_warmup_steps", 3)),
    )

    iterator = iter(loader)

    def update() -> None:
        optimizer.zero_grad(set_to_none=False)
        for _ in range(accumulation):
            inputs, targets = next(iterator)
            graph.replay(inputs, targets)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(train_cfg["max_grad_norm"])
        )
        optimizer.step()

    print("warming Phase 3 probe (no W&B, no save, no validation)...", flush=True)
    for _ in range(args.warmup_updates):
        update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.measure_updates):
        update()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    microbatches = math.ceil(len(dataset) / batch_size)
    updates = math.ceil(microbatches / accumulation)
    seconds_per_update = elapsed / args.measure_updates
    seconds_per_micro = elapsed / (args.measure_updates * accumulation)
    estimated = seconds_per_update * updates
    print(f"phase3_samples={len(dataset):,}")
    print(
        f"batch_size={batch_size} accumulation={accumulation} "
        f"effective_batch={batch_size * accumulation}"
    )
    print(f"phase3_microbatches={microbatches:,} phase3_updates={updates:,}")
    print(
        f"measured_updates={args.measure_updates} elapsed={elapsed:.3f}s "
        f"microbatch={seconds_per_micro * 1000:.3f}ms "
        f"update={seconds_per_update:.3f}s"
    )
    print(
        f"training_tokens_per_second="
        f"{batch_size * int(train_cfg['sequence_length']) / seconds_per_micro:,.0f}"
    )
    print(
        f"estimated_training_only={format_hms(estimated)} "
        f"({estimated / 3600:.3f} hours)"
    )
    print(
        f"peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.3f}GiB "
        f"peak_reserved={torch.cuda.max_memory_reserved()/1024**3:.3f}GiB"
    )
    print("excluded=wandb,validation,checkpoint,generation")


if __name__ == "__main__":
    main()
