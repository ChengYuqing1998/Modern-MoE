"""Strict single-token A/B for the decode-only vLLM Triton MoE adaptation."""

from __future__ import annotations

import torch

from modern_moe.vllm_fused_experts import fused_selected_experts


def timed(fn, iterations: int = 1000) -> float:
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    torch.manual_seed(1337)
    device = "cuda"
    dtype = torch.bfloat16
    slots, hidden_size, intermediate_size, experts = 5, 512, 1024, 14
    x = torch.randn(1, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(experts, 2 * intermediate_size, hidden_size, device=device, dtype=dtype) * 0.02
    w2 = torch.randn(experts, hidden_size, intermediate_size, device=device, dtype=dtype) * 0.02
    ids = torch.tensor([1, 4, 9, 12, 13], device=device, dtype=torch.int32)
    coeff = torch.tensor([0.5, 0.3, 0.2, 1.0, 1.0], device=device, dtype=dtype)

    def reference():
        selected_w1 = w1[ids.long()]
        selected_w2 = w2[ids.long()]
        inputs = x.expand(slots, -1).unsqueeze(1)
        gate_up = torch.bmm(inputs, selected_w1.transpose(1, 2)).squeeze(1)
        gate, up = gate_up.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate) * up
        output = torch.bmm(hidden.unsqueeze(1), selected_w2.transpose(1, 2)).squeeze(1)
        return (output * coeff[:, None]).sum(0, keepdim=True)

    def fused():
        return fused_selected_experts(x, w1, w2, ids, coeff)

    ref = reference()
    out = fused()
    torch.cuda.synchronize()
    print("max_abs=", (ref.float() - out.float()).abs().max().item())
    print("cosine=", torch.nn.functional.cosine_similarity(ref.float(), out.float()).item())
    print(f"reference_ms={timed(reference):.6f}")
    print(f"vllm_triton_ms={timed(fused):.6f}")

    # Capture both independently to compare kernel work without Python launch cost.
    def graph_time(fn):
        for _ in range(10):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = fn()
        del graph_out
        return timed(graph.replay, 5000)

    print(f"reference_graph_ms={graph_time(reference):.6f}")
    print(f"vllm_triton_graph_ms={graph_time(fused):.6f}")


if __name__ == "__main__":
    main()
