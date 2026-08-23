# Modern-MoE 性能优化总结（截至 2026-08-14）

本文记录本轮对 `nanoGPTMoE-v2` 的有效训练和推理优化，以及已经验证无收益并回退的实验。

## 1. 固定测试对象

- GPU：NVIDIA RTX 4090
- 模型参数量：501,721,600
- hidden size：512
- 层数：16（第 1 层 Dense，后 15 层 MoE）
- Attention：GQA，8 个 Query heads / 4 个 KV heads
- MoE：12 routed experts，Top-3，2 shared experts
- Expert intermediate size：1024
- 词表：151,936
- 训练 dtype：BF16
- 训练 microbatch：`2 × 2048 tokens`
- 梯度累积：12
- 固定推理 checkpoint：

```text
checkpoints/exp_20260807_172527_0bc1a144/phase-3/step_0151000.pt
```

不同表格中的数据来自不同阶段的严格 A/B，GPU 温度、序列长度和代码基线可能不同，因此各项收益不能直接相加。表中只有明确保存了独立 A/B 的项目才给出精确增量。

## 2. 训练优化

### 2.1 有效改进汇总

| 改进 | 基线 | 优化后 | 实测收益 | 状态 |
|---|---:|---:|---:|---|
| 固定形状 forward+backward CUDA Graph | 65.321 ms/microbatch | 60.380 ms/microbatch | **-4.941 ms（7.6%）** | 正式启用 |
| Liger fused RoPE + 非持久 cos/sin cache | 61.115 ms | 58.963 ms | **-2.152 ms（3.5%）** | 正式启用 |
| Apple CCE Triton kernel 替代 Apple torch-compile CE | 57.650 ms | 51.473 ms | **-6.177 ms（10.7%，1.12×）** | 正式启用 |
| Liger MoE autotune + shared experts Dense/Fused SwiGLU | 约 63.75 ms | 约 61.75 ms | **约 -2.0 ms（组合收益）** | 正式启用 |
| 最终真实训练日志 | — | 约 50.1–51.1 ms | **稳定约 51.5 ms/microbatch** | 最终状态 |

### 2.2 固定形状 microbatch CUDA Graph

训练 Graph 只捕获固定 `2×2048` 的 forward+backward：

```text
Eager：       783.852 ms/update ÷ 12 = 65.321 ms/microbatch
CUDA Graph：  724.563 ms/update ÷ 12 = 60.380 ms/microbatch
```

有效原因：减少 Python/ATen dispatch、张量分配和大量小 kernel launch。梯度裁剪、AdamW、scheduler、日志、验证和 checkpoint 仍在图外。

Graph 与 eager 总 loss 一致，代表性梯度 cosine 大于 `0.99995`。

### 2.3 Liger fused RoPE

训练专用 fused RoPE 将每层重复的 outer/cos/sin/cat/neg/mul/add 合并，并使用按 sequence/device/dtype 自动重建的非持久缓存：

```text
关闭：61.115 ms/microbatch
开启：58.963 ms/microbatch
收益：2.152 ms/microbatch
```

输出和梯度 cosine 均大于 `0.99998`。缓存不写入 checkpoint，也不与权重绑定。

### 2.4 Apple Cut Cross Entropy（CCE）

CCE 避免训练时显式构造完整 `[tokens, vocab]` logits，并将 LM head 与 cross entropy 计算融合。固定完整模型 CUDA Graph A/B：

```text
Apple torch_compile CE：57.650 ms，reserved 4.316 GiB
Apple Triton CCE：      51.473 ms，reserved 3.609 GiB
收益：                   6.177 ms/microbatch
```

LM loss 和总 loss 一致；embedding/classifier 梯度 cosine 分别为 `0.9999914` 和 `0.9999973`。

配置：

```yaml
linear_cross_entropy_impl: cce
```

### 2.5 Packed Liger MoE、永久 packed 参数和 fused router

最终训练路径使用：

```yaml
moe_training_impl: liger
moe_parameter_layout: packed_liger
fused_router: true
```

有效变化包括：

- routed expert 权重永久保存为 Liger 所需 packed layout；
- checkpoint 转换时一次性完成 Gate/Up 排列转换；
- 不再在每个 forward 中 `cat/stack` expert 权重；
- AdamW 直接维护 packed 参数和对应 moments；
- softmax、Top-k、renormalization 和训练 routing metadata 使用 fused router；
- routed experts 使用训练专用 Liger fused MoE；
- shared experts 不再经过动态 dispatch，而走 Dense/Fused SwiGLU。

这几项在最终代码中是组合启用的。保存下来的明确组合 A/B 是 Liger autotune + shared dense 路径约 `63.75 → 61.75 ms`，约节省 2 ms；其余子项没有可靠、可独立相加的完整模型数字，因此不虚构单项收益。

永久 packed checkpoint 转换脚本：

```text
scripts/convert_checkpoint_to_packed_liger.py
```

### 2.6 最终训练状态

正式 Phase-1 配置：

```text
configs/train_nanogptmoe_v2_advanced_kernels_phase1.yaml
```

关键开关：

```yaml
linear_cross_entropy_impl: cce
moe_training_impl: liger
cuda_graph: true
cuda_graph_full_update: false
```

真实日志：

```text
avg_micro=50.12 ms
avg_micro=50.69 ms
avg_micro=51.07 ms
avg_micro=50.55 ms
```

因此最终应以约 **51.5 ms/microbatch（2×2048）** 作为稳定训练速度，而不是旧日志中容易误解的区间累计时间。

## 3. 推理优化

推理指标均为 batch 1、KV cache、单 token autoregressive decode；prefill 单独报告，不计入纯 decode token/s。

### 3.1 有效改进汇总

| 改进 | 基线 | 优化后 | 实测收益 | 状态 |
|---|---:|---:|---:|---|
| 独立 inference fast path（padded prefill + selected experts + native RMSNorm） | 67.94 tok/s | 148.86 tok/s | **+80.92 tok/s（2.19×）** | 正式启用 |
| 单-token model forward CUDA Graph | 166.64 tok/s | 547.50 tok/s | **+380.86 tok/s（3.29×）** | 正式启用 |
| CUDA Graph 在完整采样生成中 | 148.50 tok/s | 426.32 tok/s | **+277.82 tok/s**；扣 setup 后约 446.5 | 正式启用 |
| Fused sampling：PyTorch → 自定义 Triton | 0.408 ms/sample | 0.0821 ms/sample | **-0.326 ms（约 4.97×）** | 可用 |
| Fused sampling：自定义 Triton → FlashInfer | 0.0821 ms/sample | 0.0794 ms/sample | **-0.0027 ms（约 3.3%）** | 正式推荐 |
| Fused inference router | 0.011–0.015 ms/router | 约 0.0017 ms/router | **约 6–9×** | 正式启用 |
| 最终完整推理栈 | 早期约 68–75 tok/s | 约 620–625 tok/s | **约 8–9×** | 最终状态 |

### 3.2 推理专用 MoE 路径

训练与推理完全隔离：

- 训练继续使用训练专用 packed Liger 路径；
- eval prefill 使用 padded expert-major batched BMM；
- 单 token decode 将 Top-3 routed + 2 shared experts 合为 5 个 selected slots；
- eval RMSNorm 使用原生 `F.rms_norm`；
- expert-major 权重使用非持久缓存，不进入 checkpoint；
- `load_state_dict()` 后缓存会失效，避免切换 checkpoint 后继续使用旧权重快照。

严格 64-token A/B：

```text
原路径：prefill 133.61 ms，decode 67.94 tok/s
快路径：prefill  30.03 ms，decode 148.86 tok/s
tokens_equal=True
```

代价是推理峰值 allocated 从约 0.95 GiB 增至约 1.58 GiB，主要来自非持久 expert 权重缓存。

统一开关：

```text
MODERN_MOE_USE_INFERENCE_FAST_PATH=1
```

### 3.3 单-token Decode CUDA Graph

只捕获固定 `[batch,1]` 模型 forward；prefill 和 sampling 保持在图外，动态 KV 位置通过 GPU int32 tensor 输入。

模型本体三轮 A/B：

```text
Eager：166.64 tok/s（6.001 ms/token）
Graph：547.50 tok/s（1.826 ms/token）
一次性 setup：约 26 ms/cache session
```

完整采样路径：

```text
Eager：148.50 tok/s
Graph：426.32 tok/s（包含约 27.05 ms setup）
扣除 setup：约 446.5 tok/s
```

开关：

```text
--cuda-graph-decode
```

### 3.4 vLLM 风格 selected-expert Triton kernels

单 token 的 3 个 routed experts 和 2 个 shared experts 共用同一条 expert 路径：

```text
Gate+Up GEMM
→ SwiGLU
→ Down GEMM + routing coefficient
→ expert sum
```

实现位于：

```text
modern_moe/vllm_fused_experts.py
```

开关：

```text
--vllm-fused-experts
```

它是最终 620+ tok/s 组合的重要组成部分，但本轮保存记录中没有一组可将它从其他同期优化完全剥离的可靠完整模型 token/s，因此不填写虚假的独立增量。

### 3.5 Fused inference router

Router Linear 保持独立；后续一次完成：

```text
FP32 softmax / Top-k / renormalization
→ routed expert IDs
→ shared expert IDs
→ routed coefficients + shared coefficients 1
```

典型输出布局：

```text
IDs:          [routed_1, routed_2, routed_3, 12, 13]
coefficients: [w1,       w2,       w3,       1,  1]
```

Router Graph microbenchmark：

```text
原路径：约 0.011–0.015 ms
融合：  约 0.0017 ms
```

支持 token count `1..4`、experts `<=128`、Top-k `<=8`、shared `<=8`；不支持的形状自动回退 PyTorch。

开关：

```text
--fused-inference-router
```

### 3.6 FlashInfer fused sampling

采样路径在 GPU 上完成 repetition penalty、temperature、no-repeat n-gram，以及 FlashInfer sorting-free Top-k/Top-p sampling。

单独 sampling benchmark：

```text
PyTorch：       0.4080 ms
自定义 Triton：0.0821 ms
FlashInfer：    0.0794 ms
```

FlashInfer 与 `torch.multinomial` 的随机数消费方式不同，相同 seed 下生成 token 序列不要求逐 token 相等；验证重点是候选分布和统计等价。

开关：

```text
--flashinfer-sampling
```

### 3.7 最终推理速度

最终组合：

```text
inference fast path
+ native eval RMSNorm
+ selected Top-3 routed + 2 shared experts
+ vLLM-style Triton expert GEMMs
+ fused inference router
+ model-only CUDA Graph
+ FlashInfer sampling
```

保存下来的完整生成结果：

```text
自定义 Triton sampling：首次 605.09 tok/s，热态 619.96 tok/s
FlashInfer sampling：    首次 597.33 tok/s，热态 625.27 tok/s
扣除一次性 Graph setup 后的稳态：约 665 tok/s
```

推荐命令：

```bash
conda run --no-capture-output -n moe-env \
  python -u -m scripts.generate \
  --checkpoint checkpoints/exp_20260807_172527_0bc1a144/phase-3/step_0151000.pt \
  --prompt "梯度下降法（英语：Gradient descent）是一种" \
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

## 4. 已验证无收益并回退

这些实验不属于最终有效配置，但记录下来可以避免重复投入。

### 4.1 训练侧

| 实验 | 结果 | 结论 |
|---|---:|---|
| 12 microbatch + clip + AdamW 完整 update Graph | 61.563 vs 58.755 ms/microbatch | 更慢，reserved 6.21→8.73 GiB |
| foreach AdamW 替代 fused AdamW | AdamW 23.869 vs 19.215 ms/update | 慢 4.654 ms，且多约 1.13 GiB reserved |
| optimizer-only Graph | tail 23.273 vs 23.137 ms/update | 无收益 |
| Fused Add+RMSNorm | 完整模型没有稳定收益/部分测试变慢 | 关闭 |
| grouped_mm 训练路径 | 约 0.14–0.16 vs 0.11–0.13 s/microbatch | 动态路由下约慢 15% |
| padded dispatch 的 vectorized `index_put_` | reserved 升至约 21.75 GiB | 无速度收益，回退 |
| packed gate/up 临时组装实验 | backward 出现大量 nonzero/index_put | 无速度和显存收益，回退 |

### 4.2 推理侧

| 实验 | 基线 | 候选 | 结论 |
|---|---:|---:|---|
| SwiGLU 融入 Down GEMM | 592.88 tok/s | 538.20 tok/s | 慢约 9.2%，回退 |
| LM head + 精确分块 Top-k | 0.2054 ms | 0.2586 ms | 慢 25.9%，回退 |
| Expert GEMM 离线 autotune | 1.811934 ms/model Graph | 1.810686 ms | 完整模型仅快 0.069%，回退 |
| 完整 decode-step CUDA Graph | 519.50 tok/s steady | 515.10 tok/s steady | 慢约 0.85%，回退 |

SwiGLU+Down 融合变慢的原因是 Down GEMM 有多个输出 tile，每个 tile 都重复计算 SwiGLU。完整 decode Graph 变慢的原因是 FlashInfer 已经高度融合，而固定 Graph 为动态 history/ngram 状态增加了额外工作。

## 5. 最终结论

当前推荐状态：

```text
训练：约 51.5 ms/microbatch（2×2048，BF16）
推理：约 620–625 tok/s；扣一次性建图后稳态约 665 tok/s
```

训练主要收益来自 CCE、固定 microbatch CUDA Graph、Liger RoPE 和 packed Liger MoE。推理主要收益来自推理专用 selected-expert 路径、model-only CUDA Graph、fused router 和 FlashInfer sampling。

在不量化、不改变模型数学结构、batch=1 的前提下，继续优化已经进入低回报区间；后续若要获得明显跃迁，应优先考虑量化、连续批处理或 speculative decoding，而不是继续融合已经很小的单个 kernel。
