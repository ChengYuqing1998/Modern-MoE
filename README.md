[English](README.md) | [中文](README.zh-CN.md)

# Modern-MoE


Modern-MoE is a research-oriented pure-PyTorch decoder-only language-model project focused on architecture, training workflows, inference, and representation analysis.

The public repository contains code, configuration templates, a small self-contained phase-1 dictionary example, and general documentation. Full training datasets, tokenized binaries, model weights, W&B runs, local artifacts, runbooks, and TOPD/OPD workflows are intentionally excluded.

### Layout

```text
modern_moe/       model, data, and generation components
scripts/          training, inference, tokenization, checkpoint, and analysis tools
configs/          model and training configuration templates
tests/            unit tests and smoke tests
docs/             general technical notes
docker/           container/environment files
tokenizer/        local tokenizer files; check the applicable license
```

### Installation and checks

```bash
conda env create -f environment.yml
conda activate moe-env
python -m unittest discover -s tests -v
```

Use Python 3.12 and a PyTorch/CUDA build compatible with the target GPU. Fused-kernel paths may require additional CUDA dependencies.

### Data interface

The repository includes a small dictionary-style phase-1 example under
`assets/data-example/raw/`. The training entry points consume
already-tokenized binaries directly; the raw example must be tokenized first.

Pretraining requires:

```text
train.bin
train.sample_idx.npy
validation.bin
validation.sample_idx.npy
```

```yaml
data_dir: assets/data-example/tokenized_qwen3_ctx2048
dataset_format: pretraining
sequence_length: 2048
```

Tokenize the included example with the repository tokenizer:

```bash
python -u -m scripts.tokenize_corpus \
  --config configs/nanogptmoe_v2_500m_liger.yaml \
  --input-dir assets/data-example/raw \
  --output-dir assets/data-example/tokenized_qwen3_ctx2048 \
  --context-length 2048
```

The command creates `train.bin`, `validation.bin`, the corresponding
`*.sample_idx.npy` files, and metadata in the output directory. The generated
tokenized files are local artifacts and are not committed.

SFT requires:

```text
train.input_ids.bin
train.labels.bin
validation.input_ids.bin
validation.labels.bin
```

```yaml
data_dir: /path/to/tokenized_sft_data
dataset_format: sft
training_entrypoint: scripts/train_sft.py
```

A small raw Qwen ChatML/Thinking format example is available at
[`assets/data-example/sft-chatml/`](assets/data-example/sft-chatml/).

DPO receives external preference pairs through `dataset`. The native loader
expects the fields `question`, `chosen`, and `rejected`:

```yaml
dataset: CultriX/dpo-merged
split: train
max_length: 1024
```

The dataset and its concrete source are not committed. DPO applies ChatML rendering, separate tokenization, policy/reference log-probability computation, and the DPO loss at runtime.

### Training order and checkpoint flow

The intended order is:

```text
phase-1 pretraining → pretrained checkpoint → SFT → SFT checkpoint
                                           → DPO policy + frozen SFT reference
                                           → DPO checkpoint
```

SFT is initialized from the pretrained checkpoint. DPO must be initialized
from the SFT checkpoint; that same SFT checkpoint is copied as the frozen
reference model. A fresh DPO run must not use an earlier DPO checkpoint as its
reference. When resuming DPO, only the policy resumes from the DPO checkpoint;
the reference remains fixed at the original SFT weights.

### Reference training performance

Recorded on a single NVIDIA RTX 4090 with CUDA 13.0, the 501.7M-parameter
packed-Liger model, BF16, and a fixed `2 × 2048` training microbatch:

- the optimized pretraining path is approximately **51.5 ms per microbatch**;
- the isolated CCE forward/backward CUDA-graph benchmark reported about
  **3.61 GiB reserved** for the microbatch kernel;
- a complete training run including gradient accumulation and fused AdamW
  reached about **15.81 GiB peak** in the recorded test.

These are reference measurements rather than hardware guarantees. Do not
interpret the isolated kernel benchmark as the total training memory footprint:
optimizer state, gradient accumulation, allocator behavior, CUDA Graphs, and
the selected model implementation all affect the final peak.

### Training

Pretraining:

```bash
python -u -m scripts.train \
  --config configs/examples/train_pretraining.yaml
```

SFT:

```bash
python -u -m scripts.train_sft \
  --config configs/examples/train_sft.yaml
```

DPO:

```bash
python -u -m scripts.train_dpo \
  --config configs/examples/train_dpo.yaml
```

The public example uses
[`CultriX/dpo-merged`](https://huggingface.co/datasets/CultriX/dpo-merged),
which provides about 60k English math, science, and knowledge preference pairs
with the exact `question`/`chosen`/`rejected` columns expected by
`scripts/train_dpo.py`. No column adapter is required. The chosen and rejected
answers are ordinary technical responses with a visible quality difference,
which makes the DPO behavior easy to inspect without using offensive data. The
example configuration uses `max_samples: 64`; set it to `0` only when
intentionally processing the full dataset.

For a quick phase-1 smoke test, tokenize the example first and then run the
pretraining command shown below. Replace checkpoint and output paths before a
real experiment. Keep full datasets, weights, and caches outside the repository.

### Inference

```bash
python -u -m scripts.generate \
  --checkpoint /path/to/checkpoint.pt \
  --prompt "Explain how a mixture-of-experts model works" \
  --chat-template \
  --mode cache \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --repetition-penalty 1.05 \
  --stream
```

Inference options:

- `--chat-template`: render the prompt with the Qwen3 ChatML template and stop at `<|im_end|>`;
- omit `--chat-template`: use raw continuation mode;
- `--mode cache`: normal incremental generation;
- `--mode no_cache`: correctness baseline;
- `--max-new-tokens`: maximum number of generated tokens;
- `--temperature`, `--top-p`, `--top-k`: sampling controls;
- `--repetition-penalty`, `--no-repeat-ngram-size`: repetition controls;
- `--stream`: print generated tokens incrementally;
- optional backend flags: `--cuda-graph-decode`, `--vllm-fused-experts`,
  `--fused-inference-router`, `--fused-sampling`, and `--flashinfer-sampling`.

Recommended accelerated inference command (single GPU, cache mode):

```bash
python -u -m scripts.generate \
  --checkpoint /path/to/checkpoint.pt \
  --prompt "Please explain how a mixture-of-experts model works" \
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

This is the recorded fast-path combination: inference fast path, selected
experts, vLLM-style fused expert GEMMs, fused router, model-only CUDA Graph,
and FlashInfer sampling. `--fused-sampling` is an alternative sampling path;
use it instead of `--flashinfer-sampling`, not together with it. Remove
`--chat-template` when raw continuation rather than ChatML dialogue is wanted.

After the CUDA Graph and fused-kernel paths have been warmed up and deployed,
the current RTX 4090/CUDA 13.0 setup can reach approximately **720 generated
tokens/s** in steady-state decode. The first request may be slower because it
includes graph and kernel setup.

### Checkpoint policy

`max_checkpoints` is a strict limit on physical `step_*.pt` files. The newest checkpoint is `latest`; the best validation checkpoint is `best`. When over the limit, the oldest checkpoint that is neither protected role is deleted. A step that is both a new best and a periodic save is written only once. Roles are recorded in the manifest rather than duplicated as weight files.

### License and data

Modern-MoE code is released under the Apache License 2.0; see the root
`LICENSE` and `NOTICE` files.

The tokenizer files under `tokenizer/qwen3_moe/` are based on the Qwen3
tokenizer released by Alibaba Cloud / Qwen and are redistributed under the
Apache License 2.0. Their attribution is recorded in that directory's
`LICENSE` and `NOTICE` files. If the tokenizer files are modified, the
modifications should be clearly identified while preserving upstream
attribution.

Third-party code and datasets remain subject to their respective licenses and
terms. This repository does not grant redistribution rights for external data
or model weights. The Hugging Face DPO dataset is downloaded at runtime and is
not mirrored into this repository; review its dataset card and Apache-2.0 terms
before use.
