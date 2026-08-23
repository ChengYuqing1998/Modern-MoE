from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .model import ModernMoEForCausalLM


@dataclass
class JacobianLens:
    """Corpus-averaged residual transport matrices, one per decoder layer."""

    matrices: dict[int, torch.Tensor]
    hidden_size: int
    samples: int
    skip_first: int
    target: str = "final_rmsnorm"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "matrices": {
                    layer: matrix.detach().cpu().float()
                    for layer, matrix in self.matrices.items()
                },
                "hidden_size": self.hidden_size,
                "samples": self.samples,
                "skip_first": self.skip_first,
                "target": self.target,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "JacobianLens":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported J-Lens file format")
        return cls(
            matrices=payload["matrices"],
            hidden_size=int(payload["hidden_size"]),
            samples=int(payload["samples"]),
            skip_first=int(payload["skip_first"]),
            target=str(payload["target"]),
        )

    def read(
        self,
        model: ModernMoEForCausalLM,
        activations: dict[int, torch.Tensor],
        position: int = -1,
    ) -> dict[int, torch.Tensor]:
        """Return full-vocabulary J-Lens logits for one token position."""
        results = {}
        weight = model.lm_head.weight.float()
        for layer, matrix in sorted(self.matrices.items()):
            hidden = activations[layer][:, position].float()
            transported = hidden @ matrix.to(hidden.device).T
            results[layer] = transported @ weight.T
        return results


def capture_residuals(
    model: ModernMoEForCausalLM,
    input_ids: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Run the ordinary model and capture post-layer residual streams."""
    activations, normalized = _forward_residual_stream(model, input_ids)
    return activations, model.lm_head(normalized)


def capture_residuals_and_routes(
    model: ModernMoEForCausalLM,
    input_ids: torch.Tensor,
) -> tuple[
    dict[int, torch.Tensor],
    torch.Tensor,
    dict[int, torch.Tensor],
]:
    """Capture residuals plus Top-k expert IDs for every layer and position."""
    routes: dict[int, torch.Tensor] = {}
    handles = []
    batch, sequence = input_ids.shape
    for layer_index, layer in enumerate(model.layers):
        def record_router(_module, _inputs, output, index=layer_index):
            routes[index] = output.detach().view(
                batch,
                sequence,
                -1,
            ).topk(model.config.num_experts_per_tok, dim=-1).indices

        handles.append(layer.moe.router.register_forward_hook(record_router))
    try:
        activations, logits = capture_residuals(model, input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return activations, logits, routes


def _forward_residual_stream(
    model: ModernMoEForCausalLM,
    input_ids: torch.Tensor,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    x = model.embed_tokens(input_ids)
    activations = {}
    for layer_index, layer in enumerate(model.layers):
        x, _, _ = layer(x, compute_router_losses=False)
        activations[layer_index] = x
    return activations, model.norm(x)


def jacobian_for_tokens(
    model: ModernMoEForCausalLM,
    input_ids: torch.Tensor,
    source_layers: Sequence[int],
    dim_batch: int = 1,
    skip_first: int = 16,
) -> dict[int, torch.Tensor]:
    """Exact paper-style Jacobian estimator for one tokenized prompt.

    A one-hot cotangent for an output channel is placed at every valid target
    position simultaneously. Its source-layer gradients are then averaged over
    valid source positions. Replicated batch elements compute ``dim_batch``
    Jacobian rows per reverse pass.
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must contain exactly one prompt")
    if dim_batch < 1:
        raise ValueError("dim_batch must be positive")
    seq_len = input_ids.size(1)
    if seq_len <= skip_first + 1:
        raise ValueError(
            f"Prompt has {seq_len} tokens; need more than {skip_first + 1}"
        )
    hidden_size = model.config.hidden_size
    invalid = [index for index in source_layers if not 0 <= index < len(model.layers)]
    if invalid:
        raise ValueError(f"Invalid source layers: {invalid}")

    replicas = input_ids.expand(dim_batch, -1)
    all_activations, target = _forward_residual_stream(model, replicas)
    source_activations = {
        layer: all_activations[layer] for layer in source_layers
    }
    valid = torch.arange(seq_len, device=input_ids.device)
    valid = (valid >= skip_first) & (valid < seq_len - 1)
    matrices = {
        layer: torch.empty(
            hidden_size,
            hidden_size,
            dtype=torch.float32,
            device=input_ids.device,
        )
        for layer in source_layers
    }
    sources = [source_activations[layer] for layer in source_layers]

    for start in range(0, hidden_size, dim_batch):
        count = min(dim_batch, hidden_size - start)
        cotangent = torch.zeros_like(target)
        for batch_index in range(count):
            cotangent[batch_index, valid, start + batch_index] = 1
        gradients = torch.autograd.grad(
            target,
            sources,
            grad_outputs=cotangent,
            retain_graph=start + dim_batch < hidden_size,
            allow_unused=False,
        )
        for layer, gradient in zip(source_layers, gradients):
            for batch_index in range(count):
                matrices[layer][start + batch_index] = (
                    gradient[batch_index, valid].float().mean(dim=0)
                )
    return matrices


def fit_jacobian_lens(
    model: ModernMoEForCausalLM,
    prompts: Iterable[torch.Tensor],
    source_layers: Sequence[int] | None = None,
    dim_batch: int = 1,
    skip_first: int = 16,
) -> JacobianLens:
    layers = (
        list(range(len(model.layers)))
        if source_layers is None
        else list(source_layers)
    )
    sums = {
        layer: torch.zeros(
            model.config.hidden_size,
            model.config.hidden_size,
            dtype=torch.float64,
        )
        for layer in layers
    }
    samples = 0
    for input_ids in prompts:
        matrices = jacobian_for_tokens(
            model,
            input_ids,
            layers,
            dim_batch=dim_batch,
            skip_first=skip_first,
        )
        for layer, matrix in matrices.items():
            sums[layer].add_(matrix.detach().cpu().double())
        samples += 1
    if samples == 0:
        raise ValueError("No prompts were provided")
    return JacobianLens(
        matrices={
            layer: (matrix / samples).float()
            for layer, matrix in sums.items()
        },
        hidden_size=model.config.hidden_size,
        samples=samples,
        skip_first=skip_first,
    )
