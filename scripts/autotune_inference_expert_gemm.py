"""Offline autotuner for the fixed batch-one decode expert GEMMs."""

from __future__ import annotations

from itertools import product
from statistics import median

import torch
import triton

from modern_moe.vllm_fused_experts import (
    _selected_expert_gemm,
    _sum_experts_kernel,
    _swiglu_kernel,
)


CONFIGS = [
    (block_n, block_k, warps, stages)
    for block_n, block_k, warps, stages in product(
        (32, 64, 128), (32, 64), (2, 4, 8), (2, 3, 4)
    )
]


def measure(fn, iterations: int = 2000) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def capture_time(fn, iterations: int = 5000) -> float:
    for _ in range(5):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    del output
    return measure(graph.replay, iterations)


def main() -> None:
    torch.manual_seed(1337)
    device, dtype = "cuda", torch.bfloat16
    slots, hidden_size, intermediate_size, experts = 5, 512, 1024, 14
    x = torch.randn(1, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(
        experts, 2 * intermediate_size, hidden_size, device=device, dtype=dtype
    ) * 0.02
    w2 = torch.randn(
        experts, hidden_size, intermediate_size, device=device, dtype=dtype
    ) * 0.02
    ids = torch.tensor([1, 4, 9, 12, 13], device=device, dtype=torch.int32)
    coeff = torch.tensor([0.5, 0.3, 0.2, 1.0, 1.0], device=device, dtype=dtype)
    gate_up = torch.empty((slots, 2 * intermediate_size), device=device, dtype=dtype)
    hidden = torch.empty((slots, intermediate_size), device=device, dtype=dtype)
    outputs = torch.empty((slots, hidden_size), device=device, dtype=dtype)
    result = torch.empty_like(x)

    def gemm1(config):
        bn, bk, warps, stages = config
        _selected_expert_gemm[(slots, triton.cdiv(2 * intermediate_size, bn))](
            x, w1, gate_up, ids, coeff,
            N=2 * intermediate_size, K=hidden_size, stride_ae=0,
            stride_be=w1.stride(0), stride_bk=w1.stride(2),
            stride_bn=w1.stride(1), stride_ce=gate_up.stride(0),
            APPLY_WEIGHT=False, SELECTED_A=False,
            BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages,
        )

    _swiglu_kernel[(slots,)](
        gate_up, hidden, I=intermediate_size,
        BLOCK=triton.next_power_of_2(intermediate_size), num_warps=4,
    )

    def gemm2(config):
        bn, bk, warps, stages = config
        _selected_expert_gemm[(slots, triton.cdiv(hidden_size, bn))](
            hidden, w2, outputs, ids, coeff,
            N=hidden_size, K=intermediate_size, stride_ae=hidden.stride(0),
            stride_be=w2.stride(0), stride_bk=w2.stride(2),
            stride_bn=w2.stride(1), stride_ce=outputs.stride(0),
            APPLY_WEIGHT=True, SELECTED_A=True,
            BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages,
        )

    def search(label, fn):
        results = []
        for index, config in enumerate(CONFIGS, 1):
            try:
                elapsed = measure(lambda c=config: fn(c), iterations=500)
                results.append((elapsed, config))
                print(f"{label} {index:02d}/{len(CONFIGS)} config={config} ms={elapsed:.6f}")
            except Exception as error:
                print(f"{label} {index:02d}/{len(CONFIGS)} config={config} failed={error}")
        results.sort()
        print(f"{label}_top5={results[:5]}")
        return [config for _, config in results[:5]]

    top1 = search("gemm1", gemm1)
    top2 = search("gemm2", gemm2)

    def full(first, second):
        gemm1(first)
        _swiglu_kernel[(slots,)](
            gate_up, hidden, I=intermediate_size,
            BLOCK=triton.next_power_of_2(intermediate_size), num_warps=4,
        )
        gemm2(second)
        _sum_experts_kernel[(1,)](
            outputs, result, E=slots, H=hidden_size,
            BLOCK=triton.next_power_of_2(hidden_size), num_warps=4,
        )
        return result

    combined = []
    for first in top1:
        for second in top2:
            elapsed = capture_time(lambda a=first, b=second: full(a, b))
            combined.append((elapsed, first, second))
            print(f"full first={first} second={second} graph_ms={elapsed:.6f}")
    combined.sort()
    print(f"BEST graph_ms={combined[0][0]:.6f} gemm1={combined[0][1]} gemm2={combined[0][2]}")

    baseline = ((64, 64, 4, 4), (64, 64, 4, 4))
    candidate = (combined[0][1], combined[0][2])
    full(*baseline)
    baseline_output = result.clone()
    full(*candidate)
    candidate_output = result.clone()
    torch.cuda.synchronize()
    print(
        "candidate_max_abs=",
        (baseline_output.float() - candidate_output.float()).abs().max().item(),
    )
    print(
        "candidate_cosine=",
        torch.nn.functional.cosine_similarity(
            baseline_output.float(), candidate_output.float()
        ).item(),
    )
    baseline_runs, candidate_runs = [], []
    for _ in range(7):
        baseline_runs.append(capture_time(lambda: full(*baseline), iterations=10000))
        candidate_runs.append(capture_time(lambda: full(*candidate), iterations=10000))
    print(f"baseline_graph_runs={baseline_runs}")
    print(f"candidate_graph_runs={candidate_runs}")
    print(
        f"MEDIAN baseline={median(baseline_runs):.6f} "
        f"candidate={median(candidate_runs):.6f} "
        f"speedup={median(baseline_runs) / median(candidate_runs):.4f}x"
    )


if __name__ == "__main__":
    main()
