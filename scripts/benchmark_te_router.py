"""Isolated Transformer Engine fused-router A/B for nanoGPTMoE-v2."""

from __future__ import annotations

import statistics
import ctypes
import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


def _load_tex():
    """Load TE's extension without importing its optional attention stack."""
    wheel_lib = Path(sys.prefix) / "lib/python3.12/site-packages/transformer_engine/wheel_lib"
    ctypes.CDLL(
        "/usr/local/cuda-13.0/targets/x86_64-linux/lib/libnvrtc.so.13",
        mode=ctypes.RTLD_GLOBAL,
    )
    ctypes.CDLL(str(wheel_lib / "libtransformer_engine.so"), mode=ctypes.RTLD_GLOBAL)
    extension = next(wheel_lib.glob("transformer_engine_torch*.so"))
    spec = importlib.util.spec_from_file_location("transformer_engine_torch", extension)
    module = importlib.util.module_from_spec(spec)
    sys.modules["transformer_engine_torch"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tex = _load_tex()


class _FusedScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, topk):
        scores, routing_map, intermediate = tex.fused_score_for_moe_aux_loss_fwd(
            logits=logits, topk=topk, score_function="softmax"
        )
        ctx.save_for_backward(intermediate)
        ctx.topk = topk
        ctx.shape = logits.shape
        return routing_map, scores

    @staticmethod
    def backward(ctx, _grad_map, grad_scores):
        (intermediate,) = ctx.saved_tensors
        grad_logits = tex.fused_score_for_moe_aux_loss_bwd(
            num_tokens=ctx.shape[0],
            num_experts=ctx.shape[1],
            intermediate_output=intermediate,
            grad_scores=grad_scores.contiguous(),
            topk=ctx.topk,
            score_function="softmax",
        )
        return grad_logits, None


def fused_compute_score_for_moe_aux_loss(logits, topk, _score_function):
    return _FusedScores.apply(logits, topk)


TOKENS = 4096
EXPERTS = 12
TOPK = 3


def reference(logits: torch.Tensor):
    scores = F.softmax(logits, dim=-1)
    weights, indices = scores.topk(TOPK, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    routing_map = F.one_hot(indices, EXPERTS).sum(dim=1).bool()
    dense_weights = torch.zeros_like(scores).scatter(1, indices, weights)
    assignment = routing_map.float() / TOPK
    aux = EXPERTS * torch.sum(assignment.mean(0) * scores.mean(0))
    z = torch.logsumexp(logits, dim=-1).square().mean()
    return dense_weights, routing_map, scores, aux, z


def fused(logits: torch.Tensor):
    routing_map, scores = fused_compute_score_for_moe_aux_loss(
        logits, TOPK, "softmax"
    )
    dense_weights = scores * routing_map
    dense_weights = dense_weights / dense_weights.sum(dim=-1, keepdim=True)
    assignment = routing_map.float() / TOPK
    aux = EXPERTS * torch.sum(assignment.mean(0) * scores.mean(0))
    z = torch.logsumexp(logits, dim=-1).square().mean()
    return dense_weights, routing_map, scores, aux, z


def objective(outputs):
    dense_weights, _, scores, aux, z = outputs
    coeff = torch.linspace(0.5, 1.5, EXPERTS, device=dense_weights.device)
    return (dense_weights * coeff).sum() / TOKENS + 0.01 * aux + 0.001 * z + 1e-5 * scores.sum()


def run(fn, source):
    logits = source.detach().clone().requires_grad_(True)
    outputs = fn(logits)
    objective(outputs).backward()
    return tuple(item.detach() for item in outputs), logits.grad.detach()


def timed(fn, source, iterations=500):
    logits = source.detach().clone().requires_grad_(True)
    for _ in range(50):
        logits.grad = None
        objective(fn(logits)).backward()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        logits.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        objective(fn(logits)).backward()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.mean(samples), statistics.median(samples)


def main():
    torch.manual_seed(1337)
    source = torch.randn(TOKENS, EXPERTS, device="cuda", dtype=torch.float32)
    ref, ref_grad = run(reference, source)
    te, te_grad = run(fused, source)
    names = ("dense_weights", "routing_map", "scores", "aux", "z")
    for name, left, right in zip(names, ref, te, strict=True):
        if left.dtype == torch.bool:
            print(f"{name}_equal={torch.equal(left, right)}")
        else:
            print(
                f"{name} max_abs={(left-right).abs().max().item():.9g} "
                f"mean_abs={(left-right).abs().mean().item():.9g}"
            )
    cosine = F.cosine_similarity(ref_grad.flatten(), te_grad.flatten(), dim=0)
    print(
        f"grad max_abs={(ref_grad-te_grad).abs().max().item():.9g} "
        f"mean_abs={(ref_grad-te_grad).abs().mean().item():.9g} "
        f"cosine={cosine.item():.9f}"
    )
    ref_mean, ref_median = timed(reference, source)
    te_mean, te_median = timed(fused, source)
    print(f"reference mean={ref_mean:.3f}us median={ref_median:.3f}us")
    print(f"te_fused  mean={te_mean:.3f}us median={te_median:.3f}us")


if __name__ == "__main__":
    main()
