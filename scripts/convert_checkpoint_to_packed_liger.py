"""Convert legacy or packed-ScatterMoE checkpoints to permanent Liger layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from modern_moe.config import ModernMoEConfig
from scripts.convert_checkpoint_to_packed_scattermoe import (
    convert_model_state,
    convert_optimizer_state,
    model_parameter_names,
    model_state_names,
)


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    first, second = tensor.chunk(2, dim=1)
    return torch.cat((second, first), dim=1)


def _routed_gate_up_names(names: list[str]) -> list[str]:
    return [name for name in names if name.endswith("routed.experts.weight")]


def _swap_model_routed_gate_up(state: dict, names: list[str]) -> None:
    for name in _routed_gate_up_names(names):
        state[name] = _swap_halves(state[name])


def _swap_optimizer_routed_gate_up(
    optimizer_state: dict, parameter_names: list[str]
) -> None:
    parameter_ids = [
        parameter_id
        for group in optimizer_state["param_groups"]
        for parameter_id in group["params"]
    ]
    if len(parameter_ids) != len(parameter_names):
        raise ValueError("optimizer parameter IDs do not match model parameters")
    id_by_name = dict(zip(parameter_names, parameter_ids, strict=True))
    for name in _routed_gate_up_names(parameter_names):
        state = optimizer_state["state"].get(id_by_name[name], {})
        for field, value in tuple(state.items()):
            if torch.is_tensor(value) and value.ndim > 0:
                state[field] = _swap_halves(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Drop optimizer/training state for a compact initialization checkpoint.",
    )
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    values = dict(checkpoint["model_config"])
    old_layout = values.get("moe_parameter_layout", "legacy")
    if old_layout not in {"legacy", "packed_scattermoe"}:
        raise ValueError(
            f"expected legacy or packed_scattermoe checkpoint, found {old_layout!r}"
        )
    values["attention_pattern"] = tuple(values["attention_pattern"])
    config = ModernMoEConfig(**values)
    packed_parameter_names = model_parameter_names(values, "packed_liger")
    packed_state_names = model_state_names(values, "packed_liger")

    if old_layout == "legacy":
        legacy_names = model_parameter_names(values, "legacy")
        checkpoint["model"] = convert_model_state(
            checkpoint["model"], packed_state_names, config
        )
        if "optimizer" in checkpoint and not args.model_only:
            checkpoint["optimizer"] = convert_optimizer_state(
                checkpoint["optimizer"], legacy_names, packed_parameter_names, config
            )

    # The existing packed converter produces [up, gate]. Liger permanently
    # stores routed experts as [gate, up]. Shared experts stay in ScatterMoE's
    # [up, gate] format because they continue to use its cached parallel path.
    _swap_model_routed_gate_up(checkpoint["model"], packed_state_names)
    if "optimizer" in checkpoint and not args.model_only:
        _swap_optimizer_routed_gate_up(
            checkpoint["optimizer"], packed_parameter_names
        )

    checkpoint["model_config"] = dict(checkpoint["model_config"])
    checkpoint["model_config"]["moe_parameter_layout"] = "packed_liger"
    checkpoint["checkpoint_format_version"] = 3
    if args.model_only:
        checkpoint = {
            "model": checkpoint["model"],
            "model_config": checkpoint["model_config"],
            "checkpoint_format_version": checkpoint["checkpoint_format_version"],
            "source_checkpoint": str(args.input),
            "initialization_only": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"converted {args.input} ({old_layout}) -> {args.output} (packed_liger)")


if __name__ == "__main__":
    main()
