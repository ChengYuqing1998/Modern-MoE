from pathlib import Path
from typing import Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .config import ModernMoEConfig


def load_tokenizer(
    config: ModernMoEConfig,
    path: Union[str, Path, None] = None,
) -> PreTrainedTokenizerBase:
    """Load the vendored Qwen3 MoE tokenizer and validate embedding capacity."""
    tokenizer_path = str(path or config.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if len(tokenizer) > config.vocab_size:
        raise ValueError(
            f"Tokenizer has {len(tokenizer):,} IDs but model vocab_size is only "
            f"{config.vocab_size:,}"
        )
    return tokenizer
