"""Convert a legacy Modern-MoE model and AdamW state to packed ScatterMoE."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from modern_moe.config import ModernMoEConfig
from modern_moe.model import ModernMoEForCausalLM


def model_parameter_names(config_values: dict, layout: str) -> list[str]:
    values = dict(config_values)
    values["attention_pattern"] = tuple(values.get("attention_pattern", ("full",)))
    values["moe_parameter_layout"] = layout
    with torch.device("meta"):
        model = ModernMoEForCausalLM(ModernMoEConfig(**values))
    return [name for name, _ in model.named_parameters()]


def model_state_names(config_values: dict, layout: str) -> list[str]:
    values = dict(config_values)
    values["attention_pattern"] = tuple(values.get("attention_pattern", ("full",)))
    values["moe_parameter_layout"] = layout
    with torch.device("meta"):
        model = ModernMoEForCausalLM(ModernMoEConfig(**values))
    return list(model.state_dict())


def source_names_for_packed(name: str, config: ModernMoEConfig) -> list[str]:
    mappings = (
        ("routed.experts.weight", "experts", "gate_up"),
        ("routed.output_experts.weight", "experts", "down"),
        ("shared.experts.weight", "shared_experts", "gate_up"),
        ("shared.output_experts.weight", "shared_experts", "down"),
    )
    for suffix, legacy_group, kind in mappings:
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            count = config.num_experts if legacy_group == "experts" else config.num_shared_experts
            if kind == "gate_up":
                result = []
                for index in range(count):
                    result.extend(
                        (
                            f"{prefix}{legacy_group}.{index}.up_proj.weight",
                            f"{prefix}{legacy_group}.{index}.gate_proj.weight",
                        )
                    )
                return result
            return [
                f"{prefix}{legacy_group}.{index}.down_proj.weight"
                for index in range(count)
            ]
    return [name]


def pack_values(name: str, values: list[torch.Tensor], config: ModernMoEConfig):
    if name.endswith(("routed.experts.weight", "shared.experts.weight")):
        return torch.stack(
            [torch.cat(values[index : index + 2], dim=0) for index in range(0, len(values), 2)]
        )
    if name.endswith(
        ("routed.output_experts.weight", "shared.output_experts.weight")
    ):
        return torch.stack(values)
    if len(values) == 1:
        return values[0]
    return torch.stack(values)


def convert_model_state(state: dict, packed_names: list[str], config: ModernMoEConfig):
    converted = {}
    for name in packed_names:
        sources = source_names_for_packed(name, config)
        converted[name] = pack_values(name, [state[source] for source in sources], config)
    return converted


def convert_optimizer_state(
    optimizer_state: dict,
    legacy_names: list[str],
    packed_names: list[str],
    config: ModernMoEConfig,
):
    groups = optimizer_state["param_groups"]
    old_ids = [parameter_id for group in groups for parameter_id in group["params"]]
    if len(groups) != 1 or len(old_ids) != len(legacy_names):
        raise ValueError("converter currently requires the project's single AdamW parameter group")
    old_id_by_name = dict(zip(legacy_names, old_ids, strict=True))
    old_states = optimizer_state["state"]
    new_states = {}
    for new_id, name in enumerate(packed_names):
        sources = source_names_for_packed(name, config)
        source_states = [old_states.get(old_id_by_name[source], {}) for source in sources]
        if not source_states or not source_states[0]:
            continue
        fields = {}
        for field in source_states[0]:
            field_values = [state[field] for state in source_states]
            first = field_values[0]
            if torch.is_tensor(first) and first.ndim > 0:
                fields[field] = pack_values(name, field_values, config)
            else:
                if any(torch.is_tensor(value) and not torch.equal(value, first) for value in field_values[1:]):
                    raise ValueError(f"inconsistent scalar optimizer state for {name}.{field}")
                fields[field] = copy.deepcopy(first)
        new_states[new_id] = fields
    new_group = copy.deepcopy(groups[0])
    new_group["params"] = list(range(len(packed_names)))
    return {"state": new_states, "param_groups": [new_group]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input and output must be different files")
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    values = dict(checkpoint["model_config"])
    old_layout = values.get("moe_parameter_layout", "legacy")
    if old_layout != "legacy":
        raise ValueError(f"expected legacy checkpoint, found {old_layout!r}")
    values["attention_pattern"] = tuple(values["attention_pattern"])
    config = ModernMoEConfig(**values)
    legacy_names = model_parameter_names(values, "legacy")
    packed_names = model_parameter_names(values, "packed_scattermoe")
    packed_state_names = model_state_names(values, "packed_scattermoe")
    checkpoint["model"] = convert_model_state(
        checkpoint["model"], packed_state_names, config
    )
    if "optimizer" in checkpoint:
        checkpoint["optimizer"] = convert_optimizer_state(
            checkpoint["optimizer"], legacy_names, packed_names, config
        )
    checkpoint["model_config"] = dict(checkpoint["model_config"])
    checkpoint["model_config"]["moe_parameter_layout"] = "packed_scattermoe"
    checkpoint["checkpoint_format_version"] = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"converted {args.input} -> {args.output}")
    print(f"parameters: legacy={len(legacy_names)} packed={len(packed_names)}")


if __name__ == "__main__":
    main()
