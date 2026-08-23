"""Isolated full-model training A/B for the reference and ScatterMoE paths."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from cut_cross_entropy import linear_cross_entropy
from scattermoe.mlp import GLUMLP
from scattermoe.mlp import flatten_sort_count
from scattermoe.parallel_experts import parallel_linear

from modern_moe.config import ModernMoEConfig
from modern_moe.layers import FullCausalAttention, RMSNorm, SparseMoE
from modern_moe.model import ModernMoEForCausalLM
from modern_moe.packed_moe import PackedSparseMoE


class ScatterSparseMoE(nn.Module):
    """Benchmark-only ScatterMoE equivalent of the project's SparseMoE."""

    def __init__(self, source: SparseMoE, config: ModernMoEConfig):
        super().__init__()
        self.num_experts = source.num_experts
        self.top_k = source.top_k
        self.router = source.router
        self.routed = GLUMLP(
            config.hidden_size,
            config.intermediate_size,
            config.num_experts,
            config.num_experts_per_tok,
            bias=False,
        ).to(device=self.router.weight.device, dtype=self.router.weight.dtype)
        shared_count = len(source.shared_experts)
        self.shared = GLUMLP(
            config.hidden_size,
            config.intermediate_size,
            shared_count,
            shared_count,
            bias=False,
        ).to(device=self.router.weight.device, dtype=self.router.weight.dtype)

        with torch.no_grad():
            self._copy_experts(self.routed, source.experts)
            self._copy_experts(self.shared, source.shared_experts)

    @staticmethod
    def _copy_experts(target: GLUMLP, sources: nn.ModuleList) -> None:
        # ScatterMoE's first projection is [up, gate], whereas the source
        # modules store the two projections independently.
        for expert_idx, expert in enumerate(sources):
            target.experts.weight[expert_idx, : expert.up_proj.out_features].copy_(
                expert.up_proj.weight
            )
            target.experts.weight[expert_idx, expert.up_proj.out_features :].copy_(
                expert.gate_proj.weight
            )
            target.output_experts.weight[expert_idx].copy_(expert.down_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        compute_router_losses: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.reshape(-1, x.size(-1))
        logits = self.router(flat).float()
        probabilities = F.softmax(logits, dim=-1)
        weights, indices = probabilities.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = self.routed(flat, weights.to(flat.dtype), indices)

        shared_count = self.shared.experts.num_experts
        shared_indices = torch.arange(
            shared_count, device=flat.device, dtype=indices.dtype
        ).expand(flat.size(0), -1)
        shared_weights = flat.new_ones(flat.size(0), shared_count)
        shared = self.shared(flat, shared_weights, shared_indices)

        if compute_router_losses:
            assignment = (
                F.one_hot(indices, self.num_experts).float().sum(dim=1) / self.top_k
            )
            load = assignment.mean(dim=0)
            importance = probabilities.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(load * importance)
            z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
        else:
            aux_loss = logits.new_zeros(())
            z_loss = logits.new_zeros(())
        return (routed + shared).view(shape), aux_loss, z_loss


def compatible_scatter_glu(
    x: torch.Tensor,
    expert_p: torch.Tensor,
    expert_idxs: torch.Tensor,
    experts: nn.ModuleList,
) -> torch.Tensor:
    """Run ScatterMoE while retaining the original independent Parameters."""
    num_experts = len(experts)
    top_k = expert_idxs.size(1)
    sorted_experts, sorted_scattered, offsets = flatten_sort_count(
        expert_idxs, num_experts=num_experts
    )
    # These differentiable stacks preserve gradients on every original
    # Parameter, so AdamW and checkpoint layouts remain unchanged.
    gate_up = torch.stack(
        [torch.cat((expert.up_proj.weight, expert.gate_proj.weight), dim=0) for expert in experts]
    )
    down = torch.stack([expert.down_proj.weight for expert in experts])
    hidden = parallel_linear(
        x, gate_up.permute(0, 2, 1), top_k,
        sorted_experts, sorted_scattered, offsets, grouped_out=True,
    )
    up, gate = hidden.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    return parallel_linear(
        hidden, down.permute(0, 2, 1), 1,
        sorted_experts, sorted_scattered, offsets, grouped_in=True,
        gates=expert_p,
    )


def compatible_scatter_forward(
    module: SparseMoE,
    x: torch.Tensor,
    compute_router_losses: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = x.shape
    flat = x.reshape(-1, x.size(-1))
    logits = module.router(flat).float()
    probabilities = F.softmax(logits, dim=-1)
    weights, indices = probabilities.topk(module.top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    routed = compatible_scatter_glu(
        flat, weights.to(flat.dtype), indices, module.experts
    )
    shared_count = len(module.shared_experts)
    shared_indices = torch.arange(
        shared_count, device=flat.device, dtype=indices.dtype
    ).expand(flat.size(0), -1)
    shared = compatible_scatter_glu(
        flat, flat.new_ones(flat.size(0), shared_count),
        shared_indices, module.shared_experts,
    )
    if compute_router_losses:
        assignment = (
            F.one_hot(indices, module.num_experts).float().sum(dim=1) / module.top_k
        )
        aux_loss = module.num_experts * torch.sum(
            assignment.mean(dim=0) * probabilities.mean(dim=0)
        )
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())
    else:
        aux_loss = logits.new_zeros(())
        z_loss = logits.new_zeros(())
    return (routed + shared).view(shape), aux_loss, z_loss


def enable_compatible_scatter(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, SparseMoE):
            child.forward = compatible_scatter_forward.__get__(child, SparseMoE)


def convert_moe(module: nn.Module, config: ModernMoEConfig) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, SparseMoE):
            setattr(module, name, ScatterSparseMoE(child, config))
        else:
            convert_moe(child, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("reference", "scatter", "scatter_compat", "integrated", "packed", "liger"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=Path("configs/nanogptmoe_v2_500m.yaml"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--full-update", action="store_true")
    parser.add_argument("--gradient-accumulation", type=int, default=12)
    parser.add_argument(
        "--loss-impl", choices=("pytorch", "apple"), default="apple"
    )
    parser.add_argument(
        "--rmsnorm-impl", choices=("native", "handwritten"), default="native"
    )
    parser.add_argument(
        "--fused-add-rms-norm", choices=("config", "on", "off"), default="config"
    )
    parser.add_argument(
        "--fused-router", choices=("config", "on", "off"), default="config"
    )
    parser.add_argument("--fused-router-ab", action="store_true")
    parser.add_argument(
        "--fused-rope", choices=("config", "on", "off"), default="config"
    )
    parser.add_argument("--fused-rope-ab", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1337)
    with args.config.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values["attention_pattern"] = tuple(values["attention_pattern"])
    if args.mode == "packed":
        values["moe_parameter_layout"] = "packed_scattermoe"
    elif args.mode == "liger":
        values["moe_parameter_layout"] = "packed_liger"
    if args.fused_add_rms_norm != "config":
        values["fused_add_rms_norm"] = args.fused_add_rms_norm == "on"
    if args.fused_router != "config":
        values["fused_router"] = args.fused_router == "on"
    if args.fused_rope != "config":
        values["fused_rope"] = args.fused_rope == "on"
    config = ModernMoEConfig(**values)
    model = ModernMoEForCausalLM(config).to("cuda", dtype=torch.bfloat16).train()
    if args.rmsnorm_impl == "handwritten":
        def handwritten_rmsnorm(module, x):
            dtype = x.dtype
            normalized = x.float() * torch.rsqrt(
                x.float().pow(2).mean(-1, keepdim=True) + module.eps
            )
            return normalized.to(dtype) * module.weight

        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.forward = handwritten_rmsnorm.__get__(module, RMSNorm)
    if args.mode == "scatter":
        convert_moe(model, config)
    elif args.mode == "scatter_compat":
        enable_compatible_scatter(model)
    elif args.mode == "integrated":
        for module in model.modules():
            if isinstance(module, SparseMoE):
                module.use_scattermoe_training = True

    tokens = torch.randint(
        config.vocab_size, (2, 2048), device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)
    )

    def microbatch(loss_scale: float = 1.0) -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.loss_impl == "apple":
                output = model(
                    tokens,
                    mtp_targets=tokens,
                    return_loss_hidden_states=True,
                    linear_ce_impl="torch_compile",
                )
                lm_loss = linear_cross_entropy(
                    output.loss_hidden_states,
                    output.classifier_weight,
                    tokens,
                    impl="torch_compile",
                )
            else:
                output = model(tokens, mtp_targets=tokens)
                lm_loss = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.size(-1)),
                    tokens.reshape(-1),
                )
            loss = (
                lm_loss
                + config.router_aux_loss_coef * output.router_aux_loss
                + config.router_z_loss_coef * output.router_z_loss
            )
        (loss * loss_scale).backward()

    if args.fused_router_ab or args.fused_rope_ab:
        if args.fused_router_ab and args.fused_rope_ab:
            raise ValueError("select only one isolated fused A/B")
        if args.fused_router_ab:
            feature = "fused_router"
            modules = [
                module for module in model.modules()
                if isinstance(module, PackedSparseMoE)
            ]
        else:
            feature = "fused_rope"
            modules = [
                module for module in model.modules()
                if isinstance(module, FullCausalAttention)
            ]

        def set_feature(enabled: bool) -> None:
            for module in modules:
                setattr(module, f"use_{feature}", enabled)

        for enabled in (False, True):
            set_feature(enabled)
            for _ in range(args.warmup):
                model.zero_grad(set_to_none=True)
                microbatch()
        torch.cuda.synchronize()
        samples = {False: [], True: []}
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(args.iterations):
            order = (False, True) if iteration % 2 == 0 else (True, False)
            for enabled in order:
                set_feature(enabled)
                model.zero_grad(set_to_none=True)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                microbatch()
                end.record()
                end.synchronize()
                samples[enabled].append(start.elapsed_time(end))
        for enabled in (False, True):
            values = samples[enabled]
            print(
                f"{feature}={enabled} mean={statistics.mean(values):.3f}ms "
                f"median={statistics.median(values):.3f}ms min={min(values):.3f}ms "
                f"max={max(values):.3f}ms"
            )
        paired = [
            enabled - baseline
            for baseline, enabled in zip(samples[False], samples[True])
        ]
        print(
            f"{feature}_paired_on_minus_off_mean={statistics.mean(paired):.3f}ms "
            f"paired_median={statistics.median(paired):.3f}ms "
            f"peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.3f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved()/1024**3:.3f}GiB"
        )
        return

    if args.full_update:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-4, betas=(0.9, 0.95),
            eps=1e-8, weight_decay=0.1, fused=True,
        )

        def update(measure_parts: bool = False):
            optimizer.zero_grad(set_to_none=True)
            backward_start = torch.cuda.Event(enable_timing=True)
            backward_end = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            backward_start.record()
            for _ in range(args.gradient_accumulation):
                microbatch(1.0 / args.gradient_accumulation)
            backward_end.record()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step_end.record()
            step_end.synchronize()
            if measure_parts:
                return (
                    backward_start.elapsed_time(backward_end),
                    backward_end.elapsed_time(step_end),
                    backward_start.elapsed_time(step_end),
                )

        print(f"compiling full update mode={args.mode}...", flush=True)
        for _ in range(args.warmup):
            update()
        torch.cuda.reset_peak_memory_stats()
        parts = [update(measure_parts=True) for _ in range(args.iterations)]
        backward = [part[0] for part in parts]
        optimizer_step = [part[1] for part in parts]
        totals = [part[2] for part in parts]
        print(
            f"mode={args.mode} loss={args.loss_impl} "
            f"rmsnorm={args.rmsnorm_impl} "
            f"fused_add_rms_norm={config.fused_add_rms_norm} "
            f"fused_router={config.fused_router} "
            f"fused_rope={config.fused_rope} "
            f"accumulation={args.gradient_accumulation} "
            f"forward_backward_median={statistics.median(backward):.3f}ms "
            f"clip_optimizer_median={statistics.median(optimizer_step):.3f}ms "
            f"update_median={statistics.median(totals):.3f}ms "
            f"amortized_median={statistics.median(totals)/args.gradient_accumulation:.3f}ms "
            f"peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.3f}GiB "
            f"peak_reserved={torch.cuda.max_memory_reserved()/1024**3:.3f}GiB"
        )
        return

    print(f"compiling mode={args.mode}...", flush=True)
    for _ in range(args.warmup):
        model.zero_grad(set_to_none=True)
        microbatch()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        model.zero_grad(set_to_none=True)
        start.record()
        microbatch()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    print(
        f"mode={args.mode} fused_add_rms_norm={config.fused_add_rms_norm} "
        f"fused_router={config.fused_router} "
        f"fused_rope={config.fused_rope} "
        f"mean={statistics.mean(samples):.3f}ms "
        f"median={statistics.median(samples):.3f}ms min={min(samples):.3f}ms "
        f"max={max(samples):.3f}ms peak_alloc={torch.cuda.max_memory_allocated()/1024**3:.3f}GiB "
        f"peak_reserved={torch.cuda.max_memory_reserved()/1024**3:.3f}GiB"
    )
    if args.profile_iterations:
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=False,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            for _ in range(args.profile_iterations):
                microbatch()
                profiler.step()
        torch.cuda.synchronize()
        print(
            profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=40
            )
        )
        print("selected_kernel_totals:")
        for event in profiler.key_averages():
            name = event.key.lower()
            if any(part in name for part in ("moe", "router", "scatter", "gather", "token_")):
                device_total = getattr(
                    event, "device_time_total",
                    getattr(event, "cuda_time_total", 0.0),
                )
                self_device = getattr(
                    event, "self_device_time_total",
                    getattr(event, "self_cuda_time_total", 0.0),
                )
                print(
                    f"{event.key}: calls={event.count} "
                    f"device_total={device_total / 1000:.3f}ms "
                    f"self_device={self_device / 1000:.3f}ms "
                    f"cpu_total={event.cpu_time_total / 1000:.3f}ms"
                )


if __name__ == "__main__":
    main()
