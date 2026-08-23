import argparse
from pathlib import Path

import yaml

from modern_moe import ModernMoEConfig, load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/modern_moe_pilot.yaml"))
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw["attention_pattern"] = tuple(raw["attention_pattern"])
    config = ModernMoEConfig(**raw)
    tokenizer = load_tokenizer(config)
    print(f"class: {type(tokenizer).__name__}")
    print(f"base vocabulary: {tokenizer.vocab_size:,}")
    print(f"tokenizer IDs: {len(tokenizer):,}")
    print(f"model embedding rows: {config.vocab_size:,}")
    print(f"padding/reserved rows: {config.vocab_size - len(tokenizer):,}")
    print(f"eos: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})")
    print(f"pad: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")


if __name__ == "__main__":
    main()
