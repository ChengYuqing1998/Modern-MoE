"""Numerical and CUDA Graph A/B for parameterized inference routing."""

from __future__ import annotations

import torch

from modern_moe.inference_router import fused_inference_route


def reference(logits, top_k, shared, dtype):
    probabilities = logits.softmax(dim=-1)
    weights, indices = probabilities.topk(top_k, dim=-1)
    weights = (weights / weights.sum(dim=-1, keepdim=True)).to(dtype)
    if shared:
        shared_ids = torch.arange(
            logits.size(1), logits.size(1) + shared,
            device=logits.device,
            dtype=indices.dtype,
        ).expand(logits.size(0), -1)
        indices = torch.cat((indices, shared_ids), dim=1)
        weights = torch.cat(
            (weights, torch.ones(logits.size(0), shared, device=logits.device, dtype=dtype)),
            dim=1,
        )
    return indices, weights


def graph_ms(fn, iterations=10000):
    for _ in range(20):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    del outputs
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main():
    torch.manual_seed(1337)
    for tokens, experts, top_k, shared in (
        (1, 12, 3, 2),
        (4, 12, 3, 2),
        (1, 16, 4, 1),
        (1, 32, 2, 0),
    ):
        logits = torch.randn(tokens, experts, device="cuda", dtype=torch.float32)
        ref_ids, ref_weights = reference(logits, top_k, shared, torch.bfloat16)
        ids, weights = fused_inference_route(
            logits, top_k, shared, torch.bfloat16
        )
        torch.cuda.synchronize()
        print(
            f"T={tokens} E={experts} K={top_k} S={shared} "
            f"ids_equal={torch.equal(ref_ids.to(ids.dtype), ids)} "
            f"weight_max_abs={(ref_weights.float()-weights.float()).abs().max().item():.9f} "
            f"reference_graph_ms={graph_ms(lambda: reference(logits, top_k, shared, torch.bfloat16)):.6f} "
            f"fused_graph_ms={graph_ms(lambda: fused_inference_route(logits, top_k, shared, torch.bfloat16)):.6f}"
        )


if __name__ == "__main__":
    main()
