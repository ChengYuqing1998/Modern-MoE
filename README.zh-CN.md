[English](README.md) | [中文](README.zh-CN.md)

# Modern-MoE


Modern-MoE 是一个面向模型结构、训练流程、推理和表示分析的纯 PyTorch 语言模型项目。

主要能力：

- Decoder-only Transformer/MoE；
- GQA、RoPE、RMSNorm 和稀疏 MoE；
- 预训练、监督微调（SFT）和直接偏好优化（DPO）；
- packed token binary 数据管线；
- KV cache、增量生成和推理性能分析；
- checkpoint 续训、best/latest 管理和模型行为探查。

本仓库发布代码、配置模板、一个小型自包含的 phase-1 字典示例数据和通用文档。完整训练数据、tokenized binary、模型权重、W&B 记录、实验产物以及 TOPD/OPD 流程不发布。

## 目录结构

```text
modern_moe/       模型配置、网络结构、数据和生成基础组件
scripts/          训练、推理、tokenize、checkpoint 和分析脚本
configs/          模型配置及预训练/SFT/DPO 配置模板
tests/            单元测试和 smoke tests
docs/             通用技术说明
docker/           容器和环境相关文件
tokenizer/        本地 tokenizer 文件（使用前请确认许可证）
```

以下内容不随仓库提供：

- 完整数据集及其来源文件；仓库只提供 `assets/data-example/` 下的小型示例；
- `checkpoints/` 下的预训练、SFT、DPO 和 RL 权重；
- 本地日志、profiling 文件、W&B 运行目录和缓存；
- 训练 runbook、个人实验记录和机器绝对路径；
- TOPD/OPD 的训练说明和运行流程。

需要复现实验时，请在本地准备这些资源，并在 YAML 中填写路径。

## 安装

```bash
conda env create -f environment.yml
conda activate moe-env
```

或者：

```bash
python -m pip install -r requirements.txt
```

建议使用 Python 3.12，并根据目标 GPU 安装匹配的 PyTorch/CUDA 版本。部分 FlashAttention、Liger 或 fused-kernel 路径需要额外的 CUDA 依赖。

## 快速检查

```bash
python -m scripts.inspect_model --config configs/nanogptmoe_v2_500m_liger.yaml
python -m unittest discover -s tests -v
python -m scripts.inspect_tokenizer --tokenizer tokenizer/qwen3_moe
```

## 数据接口

训练入口默认直接读取已经 tokenize 好的二进制文件，不会在训练过程中读取原始语料或重新 tokenize。仓库提供一个小型字典示例，使用前需要先执行 tokenize。

### 预训练

`data_dir` 应指向：

```text
<data_dir>/
├── train.bin
├── train.sample_idx.npy
├── validation.bin
└── validation.sample_idx.npy
```

```yaml
data_dir: assets/data-example/tokenized_qwen3_ctx2048
dataset_format: pretraining
sequence_length: 2048
```

对仓库内的 phase-1 字典示例执行 tokenize：

```bash
python -u -m scripts.tokenize_corpus \
  --config configs/nanogptmoe_v2_500m_liger.yaml \
  --input-dir assets/data-example/raw \
  --output-dir assets/data-example/tokenized_qwen3_ctx2048 \
  --context-length 2048
```

命令会生成 `train.bin`、`validation.bin`、对应的 `*.sample_idx.npy` 和
`metadata.json`。生成的 tokenized 文件属于本地产物，不会提交到仓库。

`.bin` 保存 packed token IDs，`*.sample_idx.npy` 保存固定长度样本的起始位置。长度为 `T` 时：

```text
input_ids = tokens[:T]
labels    = tokens[1:T+1]
```

### SFT

SFT 使用已经完成对话模板渲染和 tokenize 的输入/标签文件：

```text
<data_dir>/
├── train.input_ids.bin
├── train.labels.bin
├── validation.input_ids.bin
└── validation.labels.bin
```

```yaml
data_dir: /path/to/tokenized_sft_data
dataset_format: sft
training_entrypoint: scripts/train_sft.py
sequence_length: 2048
```

labels 文件负责指定监督位置；实际对话内容不放入仓库。

仓库提供少量 Qwen ChatML/Thinking 原始格式示例，见
[`assets/data-example/sft-chatml/`](assets/data-example/sft-chatml/)。

### DPO

DPO 使用偏好对，通过 YAML 或命令行的 `dataset` 字段加载外部数据。原生
读取逻辑要求字段为 `question`、`chosen`、`rejected`：

```yaml
dataset: CultriX/dpo-merged
split: train
max_length: 1024
```

DPO 数据集不提交到仓库。运行时由使用者提供可访问的数据集标识。DPO 会将
`question`、`chosen` 和 `rejected` 渲染为 ChatML，分别 tokenize，计算
policy/reference completion log-prob，再计算 DPO loss。

## 训练顺序与 checkpoint 关系

项目推荐的顺序是：

```text
phase-1 预训练 → 预训练 checkpoint → SFT → SFT checkpoint
                                      → DPO policy + 固定的 SFT reference
                                      → DPO checkpoint
```

SFT 使用预训练 checkpoint 初始化。DPO 必须使用 SFT checkpoint 初始化
policy，同时把同一个 SFT checkpoint 复制为冻结的 reference。新的 DPO 训练
不能把之前的 DPO checkpoint 当作 reference。DPO 续训时，只有 policy 从
DPO checkpoint 恢复，reference 始终保持最初的 SFT 权重不变。

## 预训练性能参考

在单张 NVIDIA RTX 4090、501.7M 参数 packed-Liger 模型、BF16、固定
`2 × 2048` 训练 microbatch 条件下，已有记录显示：

- 优化后的预训练路径约为 **51.5 ms / microbatch**；
- 单独的 CCE forward/backward CUDA Graph benchmark 的 reserved 显存约为
  **3.61 GiB**；
- 包含梯度累积和 fused AdamW 的完整训练测试，峰值显存约为 **15.81 GiB**。

这些是参考测量，不是所有机器上的保证值。单独 kernel benchmark 的显存不能
代表完整训练显存；optimizer 状态、梯度累积、CUDA allocator、CUDA Graph 和
具体模型实现都会影响最终峰值。因此不建议简单把该配置标成固定的“10GB
显存占用”。

## 预训练

准备 tokenized binary 后，修改配置中的数据路径；仓库示例配置已经默认指向：

```yaml
model_config: configs/nanogptmoe_v2_500m_liger.yaml
data_dir: assets/data-example/tokenized_qwen3_ctx2048
dataset_format: pretraining
checkpoint_root: /path/to/checkpoints
```

启动：

```bash
python -u -m scripts.train \
  --config configs/examples/train_pretraining.yaml
```

这是一个用于验证数据管线和训练入口的极小示例，不代表正式预训练规模。

从已有模型权重初始化时使用：

```yaml
init_checkpoint: /path/to/base_checkpoint.pt
resume_training: false
```

`init_checkpoint` 只加载模型权重；如果要恢复 optimizer、scheduler、epoch 和 batch 位置，应使用 resume checkpoint。

## SFT

```yaml
model_config: configs/nanogptmoe_v2_500m_liger.yaml
data_dir: /path/to/tokenized_sft_data
dataset_format: sft
training_entrypoint: scripts/train_sft.py
init_checkpoint: /path/to/pretrained_checkpoint.pt
checkpoint_root: /path/to/checkpoints
```

启动示例：

```bash
python -u -m scripts.train_sft \
  --config configs/examples/train_sft.yaml
```

运行正式实验前，请替换数据、初始化 checkpoint 和输出目录。

## DPO

DPO 从 SFT checkpoint 初始化 policy，并固定 reference 模型。主干和 router 可以使用不同学习率：

```yaml
checkpoint: /path/to/sft_checkpoint.pt
dataset: CultriX/dpo-merged
split: train
lr: 1.0e-4
router_lr: 2.0e-5
beta: 0.1
max_checkpoints: 4
```

启动：

```bash
python -u -m scripts.train_dpo \
  --config configs/examples/train_dpo.yaml
```

公开示例使用 Hugging Face 数据集
[`CultriX/dpo-merged`](https://huggingface.co/datasets/CultriX/dpo-merged)。
该数据集约有 6 万条英文数学、科学和知识类偏好对，字段正好是
`question`、`chosen`、`rejected`，与 `scripts/train_dpo.py` 的读取逻辑一致，
不需要额外改列名或适配器。chosen/rejected 都是正常的技术回答，但质量差异
比较明显，适合公开演示 DPO，不需要使用脏话数据。示例配置默认只取
`max_samples: 64`；确认资源充足并希望跑完整数据集时再改为 `0`。

DPO checkpoint、reference cache、数据集 cache 都应放在仓库之外。

## 推理

```bash
python -u -m scripts.generate \
  --checkpoint /path/to/checkpoint.pt \
  --prompt "请解释混合专家模型的工作原理" \
  --mode cache \
  --max-new-tokens 128 \
  --temperature 0.7 \
  --top-p 0.9
```

`no_cache` 用于正确性基线，`cache` 用于增量生成，`mtp` 用于 MTP 提议/校正实验，`all` 用于多路径对比。

## Checkpoint 管理

训练 checkpoint 包含模型、optimizer、scheduler、训练位置和随机数状态。`max_checkpoints` 是物理 `step_*.pt` 文件的严格上限。

当前策略：

1. 新 checkpoint 先写临时文件，再原子替换正式文件；
2. 最新文件标记为 `latest`；
3. 验证集最优文件标记为 `best`；
4. 超过上限时删除最老的、既不是 `latest` 也不是 `best` 的文件；
5. 同一步同时刷新 best 和命中周期保存时，只进行一次物理写盘。

`best` 和 `latest` 是 manifest 中的角色，不会额外复制权重。

## 分析工具

- `scripts/compare_dpo.py`：比较 SFT 与 DPO 输出；
- `scripts/probe_dpo_trigger.py`：分析 token、logit 和层间差异；
- `scripts/inspect_model.py`：检查模型配置和参数规模；
- `scripts/inspect_tokenizer.py`：检查 tokenizer 和特殊 token；
- `scripts/fit_jlens.py`、`scripts/inspect_jlens.py`：分析层表示与输出映射；
- `scripts/benchmark_*.py`：测量训练或推理性能。

这些工具需要使用者自行提供 checkpoint 和数据路径。

## 路径约定

公开配置应使用相对路径、环境变量或占位路径：

```yaml
data_dir: ${DATA_DIR}/tokenized/train
checkpoint_root: ${CHECKPOINT_ROOT}
init_checkpoint: ${BASE_CHECKPOINT}
```

不要提交个人 home 目录、服务器挂载目录、私有 W&B URL 或本地 cache 路径。

## 许可证和数据

Modern-MoE 代码使用 Apache License 2.0，详见根目录的 `LICENSE` 和 `NOTICE`。

`tokenizer/qwen3_moe/` 下的 tokenizer 文件基于 Alibaba Cloud / Qwen 发布的
Qwen3 tokenizer，并按照 Apache License 2.0 再分发；其归属说明见该目录下的
`LICENSE` 和 `NOTICE`。如果替换或修改 tokenizer 文件，应在相应文件或说明中
明确标注修改内容，并保留上游归属信息。

第三方代码和数据集仍分别受各自许可证或使用条款约束。本仓库不授予任何外部
数据集或模型权重的再分发权。
