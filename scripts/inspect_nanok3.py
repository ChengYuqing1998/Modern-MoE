"""Instantiate a nanoK3 YAML config and report exact parameter counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from nanok3 import NanoK3Config, NanoK3ForCausalLM, parameter_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/nanok3_300m.yaml")
    )
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw["attention_pattern"] = tuple(raw["attention_pattern"])
    config = NanoK3Config(**raw)
    model = NanoK3ForCausalLM(config)
    report = {
        "config": str(args.config),
        "attention_layers": {
            kind: sum(
                config.attention_type(index) == kind
                for index in range(config.num_hidden_layers)
            )
            for kind in ("kda", "mla")
        },
        **parameter_report(model),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
