# SFT ChatML / Thinking Example

This directory contains three small, clean SFT format examples following the
same schema as the local SFT mixture. The records use the Qwen message schema:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"category":"..."}
```

The assistant content may contain an explicit Qwen-style reasoning block:

```text
<think>
internal reasoning...
</think>

final answer...
```

The `messages` structure is the logical ChatML input. The tokenizer's Qwen3
chat template renders it into `<|im_start|>...<|im_end|>` tokens during SFT
preparation. The small file here is for format inspection and smoke tests, not
for meaningful SFT quality.

## Preparing SFT binaries

The full local SFT preparation pipeline is intentionally kept outside the
public release because it depends on the complete private/local mixture. The
raw example can be passed to the local `scripts/prepare_sft_data.py` tool when
that tool and its corpus-building dependencies are available:

```bash
python -u -m scripts.prepare_sft_data \
  --input assets/data-example/sft-chatml/raw/sft_sample.jsonl \
  --output-dir assets/data-example/sft-chatml/tokenized \
  --tokenizer tokenizer/qwen3_moe \
  --all-data \
  --sequence-length 2048
```

The generated files are `train.input_ids.bin`, `train.labels.bin`,
`validation.input_ids.bin`, and `validation.labels.bin`. They are local
artifacts and are ignored by Git.

## Inference options

`scripts/generate.py` accepts a plain user prompt. Add `--chat-template` to
render that prompt with the Qwen3 ChatML template and stop at `<|im_end|>`:

```bash
python -u -m scripts.generate \
  --checkpoint /path/to/checkpoint.pt \
  --prompt "请解释什么是混合专家模型" \
  --chat-template \
  --mode cache \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --stream
```

Useful choices:

- `--chat-template`: enable Qwen3 ChatML rendering and assistant-only output;
- omit `--chat-template`: run raw continuation mode;
- `--mode cache`: normal incremental generation;
- `--mode no_cache`: correctness baseline;
- `--temperature`, `--top-p`, `--top-k`: sampling controls;
- `--repetition-penalty`, `--no-repeat-ngram-size`: repetition controls;
- `--stream`: stream decoded text as it is generated.

Performance-related optional flags are `--cuda-graph-decode`,
`--vllm-fused-experts`, `--fused-inference-router`, `--fused-sampling`, and
`--flashinfer-sampling`. These are environment- and backend-dependent; use
them only after the ordinary `cache` path is working.

For the recommended accelerated single-GPU path, use:

```bash
python -u -m scripts.generate \
  --checkpoint /path/to/checkpoint.pt \
  --prompt "请解释混合专家模型的工作原理" \
  --chat-template \
  --mode cache \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-k 50 \
  --top-p 0.9 \
  --repetition-penalty 1.1 \
  --no-repeat-ngram-size 4 \
  --cuda-graph-decode \
  --vllm-fused-experts \
  --fused-inference-router \
  --flashinfer-sampling \
  --stream
```

Use `--fused-sampling` instead of `--flashinfer-sampling` if the custom fused
sampling path is the one being tested; do not enable both at once.

After warm-up deployment, the current RTX 4090/CUDA 13.0 setup can reach about
720 generated tokens/s in steady-state decode. The first request includes
one-time graph and kernel setup overhead.

The current command-line interface does not expose a separate
`--enable-thinking/--disable-thinking` switch. With `--chat-template`, the
Qwen3 tokenizer template is used; if explicit `<think>...</think>` content is
present in an SFT assistant example, it is part of that assistant target.
