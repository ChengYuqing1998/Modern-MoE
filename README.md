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

DPO receives external chosen/rejected preference pairs through `dataset`:

```yaml
dataset: <external-preference-dataset>
split: train
max_length: 1024
```

The dataset and its concrete source are not committed. DPO applies ChatML rendering, separate tokenization, policy/reference log-probability computation, and the DPO loss at runtime.

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

For a quick phase-1 smoke test, tokenize the example first and then run the
pretraining command shown below. Replace checkpoint and output paths before a
real experiment. Keep full datasets, weights, and caches outside the repository.

### Inference

```bash
python -u -m scripts.generate \
  --checkpoint /path/to/checkpoint.pt \
  --prompt "Explain how a mixture-of-experts model works" \
  --mode cache \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9
```

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
or model weights.
