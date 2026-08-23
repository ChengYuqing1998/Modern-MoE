"""Strict numerical and performance A/B for the isolated fused sampler."""

from __future__ import annotations

import time

import torch

from modern_moe.fused_sampling import FusedSampler
from modern_moe.generation import GenerationConfig, _filtered_probabilities, _sample


def main() -> None:
    torch.manual_seed(1337)
    vocab_size, history_length, iterations = 151_936, 256, 1000
    config = GenerationConfig(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1,
        no_repeat_ngram_size=4,
    )
    history = torch.randint(
        0, vocab_size, (1, history_length), device="cuda", dtype=torch.long
    )
    # Ensure the n-gram banning branch has a known match.
    history[0, -3:] = history[0, 10:13]
    logits = torch.randn(1, vocab_size, device="cuda", dtype=torch.bfloat16)
    sampler = FusedSampler(history, config, history_length + iterations + 1, vocab_size)
    flashinfer_sampler = FusedSampler(
        history,
        config,
        history_length + iterations + 1,
        vocab_size,
        backend="flashinfer",
    )

    reference = _filtered_probabilities(logits, history, config)
    indices, probabilities = sampler.candidate_probabilities(logits)
    reconstructed = torch.zeros_like(reference).scatter(1, indices[None], probabilities[None])
    print("prob_max_abs=", (reference - reconstructed).abs().max().item())
    print("prob_l1=", (reference - reconstructed).abs().sum().item())
    print("support_equal=", torch.equal(reference > 0, reconstructed > 0))
    reference_support = torch.nonzero(reference[0] > 0).flatten()
    fused_support = torch.nonzero(reconstructed[0] > 0).flatten()
    print("reference_support_count=", reference_support.numel())
    print("fused_support_count=", fused_support.numel())
    print(
        "reference_only=",
        reference_support[~torch.isin(reference_support, fused_support)].cpu().tolist(),
    )
    print(
        "fused_only=",
        fused_support[~torch.isin(fused_support, reference_support)].cpu().tolist(),
    )

    for _ in range(20):
        _sample(_filtered_probabilities(logits, history, config))
        sampler.candidate_probabilities(logits)
    torch.cuda.synchronize()

    def measure(fn) -> float:
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / iterations

    reference_ms = measure(lambda: _sample(_filtered_probabilities(logits, history, config)))
    fused_ms = measure(lambda: sampler.candidate_probabilities(logits))
    # Include multinomial in both timings.
    fused_sample_ms = measure(lambda: _sample_from_candidates(sampler, logits))
    fused_kernel_sample_ms = measure(lambda: sampler.sample(logits))
    flashinfer_sample_ms = measure(lambda: flashinfer_sampler.sample(logits))
    print(f"reference_filter_sample_ms={reference_ms:.6f}")
    print(f"fused_filter_ms={fused_ms:.6f}")
    print(f"fused_filter_sample_ms={fused_sample_ms:.6f}")
    print(f"fused_kernel_filter_sample_ms={fused_kernel_sample_ms:.6f}")
    print(f"flashinfer_filter_sample_ms={flashinfer_sample_ms:.6f}")


def _sample_from_candidates(sampler, logits):
    indices, probabilities = sampler.candidate_probabilities(logits)
    return indices.gather(0, torch.multinomial(probabilities, 1))


if __name__ == "__main__":
    main()
