# nanoK3 架构与缩放方案

## 实现基线

nanoK3 是 Kimi K3 的文本模型缩小版，依据 Moonshot AI 发布的 K3
Hugging Face 参考代码和技术报告实现。当前保留：

- 3 层 KDA + 1 层 NoPE Gated MLA 的周期，最后一层强制为 MLA；
- KDA 的短卷积、full-rank decay gate、safe gate 和 full-rank output gate；
- NoPE MLA 的低秩 Q/KV 投影、共享位置通道和 sigmoid 输出门；
- Stable LatentMoE：路由专家在 latent space 中运算，RMSNorm 后升回主干维度，
  并行保留 full-width shared expert；
- sigmoid Top-k 无辅助损失路由，路由偏置只参与专家选择，实际混合权重使用
  未加偏置的分数；
- Quantile Balancing、SiTU-GLU 和 Block Attention Residual（AttnRes）；
- pre-norm、无注意力位置旋转。K3 本身是 NoPE，不使用 RoPE。

与官方超大模型相比，nanoK3 只调整宽度、深度、专家数、Top-k 和 AttnRes
block 大小。视觉编码器、多 token prediction 和分布式 MoonEP 不在这个文本模型
首版中。

## 训练与反向传播

技术报告给出了 KDA 的 recurrent/chunkwise 公式以及 FlashKDA
训练/预填充 kernel，但没有给出一套要求模型作者手写的 CUTLASS backward
公式。官方开源路径将这部分放在 FLA 中。

本实现调用 `fla.ops.kda.chunk_kda`。它带有 PyTorch autograd 支持，所以
`loss.backward()` 会进入 FLA 的 KDA backward；无需在本仓库再维护一套
CUDA/CUTLASS 扩展。MLA、MoE、SiTU 和 AttnRes 使用可微 PyTorch 运算。

这和“自己写 CUDA backward”在模型数学上没有差别，区别只在算子工程层。
若以后要做性能复现，再单独锁定 GPU 架构、CUDA、Triton/CUTLASS 和 FLA
版本实现专用 kernel 更合适。

## 参数方案

精确计数如下（active 表示单 token 路径中启用两个 routed experts）：

| 配置 | 总参数 | 激活参数/token | 说明 |
|---|---:|---:|---|
| `nanok3_300m.yaml` | 299,923,300 | 189,626,212 | 默认；untied embedding/head，最忠实 |
| `nanok3_240m_tied.yaml` | 241,630,052 | 111,868,772 | tied embedding/head，42 experts |

Qwen3 tokenizer 有 151,936 个 token。在 hidden size 512 下，untied
embedding 和 LM head 合计约 1.556 亿参数，而且它们始终激活。因此 300M
默认版即使使用 MoE，激活参数仍约 1.89 亿。这不是路由失效，而是大词表成本。

如果重点是忠实复现 K3，使用默认 300M 版。如果重点是单卡训练吞吐和显存，
使用 240M tied 版更合理；代价是偏离官方 K3 的 untied embedding 设计。

## 当前边界

- KDA 训练要求 CUDA 和兼容的 `fla-core`；模型导入、配置和参数统计可在 CPU。
- 缩放模型保留 K3 的 128 维 KDA head，以及 MLA 的 128+64 Q/K head
  和 128 维 V head，只减少 head 数。这样也满足 FLA FlashKDA CUTLASS
  推理前向对 K/V head dimension 128 的约束。
- 两套正式配置的 MLA 使用 FlashAttention 3。实现沿用官方 FlashAttention
  路径的 V-dimension padding；`sdpa` 和 `eager` 后端保留作调试和回退。
- 当前训练路径面向已经打包好的等长 2048 token 样本，不依赖 padding。
  KDA 遇到真实 padding 会显式报错，避免 ShortConv 跨 padding 产生与官方
  unpadding + `cu_seqlens` 路径不同的静默结果。
- Quantile Balancing 当前按单进程的每个 micro-batch 更新。扩展到多卡时，应对
  quantile/histogram 做跨 rank 聚合，再将新偏置用于下一步。
- 官方 Hugging Face MoE 参考代码偏重推理；本实现的专家 dispatch 是可反传的
  PyTorch 版本，数学结构相同，但不是最终高吞吐 fused dispatch。
