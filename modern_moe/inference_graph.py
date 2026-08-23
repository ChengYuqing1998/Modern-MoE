"""CUDA Graph runner for fixed-shape single-token model decode."""

from __future__ import annotations

import torch

from .layers import FullAttentionCache


class CUDAGraphedDecode:
    """Capture one model-only decode step; sampling remains outside the graph."""

    def __init__(self, model, cache, max_cache_length: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphedDecode requires CUDA")
        if not cache or not isinstance(cache[0], FullAttentionCache):
            raise ValueError("prefill must create a full-attention KV cache first")
        self.model = model
        self.cache = cache
        self.max_cache_length = int(max_cache_length)
        self.batch_size = cache[0].key.size(0)
        self.token = torch.zeros(
            (self.batch_size, 1), device=cache[0].key.device, dtype=torch.long
        )
        self.position = torch.full(
            (self.batch_size,),
            cache[0].length,
            device=cache[0].key.device,
            dtype=torch.int32,
        )

        # Warm every lazy kernel with the exact static shapes before capture.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self.output = model.forward_inference(
                    self.token,
                    cache=self.cache,
                    max_cache_length=self.max_cache_length,
                    cache_position=self.position,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model.forward_inference(
                self.token,
                cache=self.cache,
                max_cache_length=self.max_cache_length,
                cache_position=self.position,
            )
        torch.cuda.synchronize()

    def replay(self, token: torch.Tensor, position: int):
        if token.shape != self.token.shape:
            raise ValueError(f"decode token must have shape {tuple(self.token.shape)}")
        if position >= self.max_cache_length:
            raise ValueError("KV cache capacity exceeded")
        self.token.copy_(token)
        self.position.fill_(position)
        self.graph.replay()
        for layer_cache in self.cache:
            if isinstance(layer_cache, FullAttentionCache):
                layer_cache.length = position + 1
        return self.output
