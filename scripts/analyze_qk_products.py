#!/usr/bin/env python3
"""Measure unscaled post-RoPE QK products from a Modern-MoE checkpoint.

The script intentionally avoids the LM head: it runs the decoder layer by
layer, records the actual post-RoPE Q/K tensors used by attention, and samples
causal query/key pairs.  Scores are *not* divided by sqrt(head_dim).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from modern_moe.config import ModernMoEConfig
from modern_moe.layers import FullCausalAttention, apply_rope
from modern_moe.model import ModernMoEForCausalLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--validation-bin",
        type=Path,
        default=Path("data/pretrain/phase-3/with-legacy-1gb/tokenized_qwen3_ctx2048/validation.bin"),
    )
    p.add_argument(
        "--validation-sample-idx",
        type=Path,
        default=Path("data/pretrain/phase-3/with-legacy-1gb/tokenized_qwen3_ctx2048/validation.sample_idx.npy"),
    )
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--pairs-per-layer", type=int, default=100_000)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/qk_analysis_step28000"))
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_checkpoint(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = dict(ckpt["model_config"])
    raw["attention_pattern"] = tuple(raw["attention_pattern"])
    if device.startswith("cpu"):
        # FlashAttention is CUDA-only; eager attention is equivalent for the
        # Q/K instrumentation and keeps the notebook runnable on CPU.
        raw["full_attention_backend"] = "eager"
    config = ModernMoEConfig(**raw)
    model = ModernMoEForCausalLM(config)
    model.load_state_dict(ckpt["model"], strict=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model.eval().to(device=device, dtype=dtype)
    return model, config, ckpt


def load_samples(bin_path: Path, sample_idx_path: Path, count: int, length: int):
    tokens = np.memmap(bin_path, dtype="<u4", mode="r")
    offsets = np.load(sample_idx_path, allow_pickle=False)
    samples = []
    for offset in offsets[:count]:
        start = int(offset)
        arr = np.asarray(tokens[start : start + length], dtype=np.int64)
        if arr.size == length:
            samples.append(torch.from_numpy(arr))
    if not samples:
        raise RuntimeError("No complete validation samples found")
    return torch.stack(samples)


@torch.inference_mode()
def measure(model, config, input_ids, pairs_per_layer: int, device: str):
    generator = torch.Generator(device=device).manual_seed(1337)
    by_layer: list[torch.Tensor] = []
    vector_stats: list[dict] = []
    x = model.embed_tokens(input_ids.to(device))
    seq_len = x.size(1)
    positions = torch.arange(seq_len, device=device)
    for layer_idx, layer in enumerate(model.layers):
        normalized = layer.input_norm(x)
        attention = layer.attention
        if not isinstance(attention, FullCausalAttention):
            x, _, _ = layer(x, compute_router_losses=False)
            continue
        q = attention.q_proj(normalized).view(
            input_ids.size(0), seq_len, attention.num_heads, attention.head_dim
        ).transpose(1, 2)
        k = attention.k_proj(normalized).view(
            input_ids.size(0), seq_len, attention.num_kv_heads, attention.head_dim
        ).transpose(1, 2)
        cos, sin = attention.rope(positions, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(attention.kv_groups, dim=1)

        # Uniformly sample valid causal (query, key) pairs.  Include all
        # diagonal pairs, which are useful for spotting scale outliers.
        n_random = max(0, pairs_per_layer - seq_len * input_ids.size(0))
        q_idx = torch.randint(seq_len, (input_ids.size(0), n_random), generator=generator, device=device)
        k_idx = torch.rand((input_ids.size(0), n_random), generator=generator, device=device)
        k_idx = (k_idx * (q_idx + 1)).long()
        batch_idx = torch.arange(input_ids.size(0), device=device)[:, None]
        head_scores = []
        q_vectors = []
        k_vectors = []
        for head in range(attention.num_heads):
            q_random = q[batch_idx, head, q_idx]
            k_random = k[batch_idx, head, k_idx]
            random_scores = (q_random * k_random).sum(dim=-1)
            diag = (q[:, head] * k[:, head]).sum(dim=-1)
            head_scores.append(torch.cat((random_scores.flatten(), diag.flatten())))
            q_vectors.append(q_random.reshape(-1, attention.head_dim))
            k_vectors.append(k_random.reshape(-1, attention.head_dim))
        scores = torch.cat(head_scores).float().cpu()
        by_layer.append(scores)
        qv = torch.cat(q_vectors).float()
        kv = torch.cat(k_vectors).float()
        q_means, q_stds = qv.mean(dim=-1), qv.std(dim=-1, unbiased=False)
        k_means, k_stds = kv.mean(dim=-1), kv.std(dim=-1, unbiased=False)
        vector_stats.append({
            "layer": layer_idx,
            "vectors": int(qv.size(0)),
            "q_vector_mean_avg": float(q_means.mean()),
            "q_vector_mean_std": float(q_means.std(unbiased=False)),
            "q_vector_mean_q01": float(torch.quantile(q_means, 0.01)),
            "q_vector_mean_q99": float(torch.quantile(q_means, 0.99)),
            "q_vector_std_avg": float(q_stds.mean()),
            "q_vector_std_std": float(q_stds.std(unbiased=False)),
            "q_vector_std_q01": float(torch.quantile(q_stds, 0.01)),
            "q_vector_std_q99": float(torch.quantile(q_stds, 0.99)),
            "k_vector_mean_avg": float(k_means.mean()),
            "k_vector_mean_std": float(k_means.std(unbiased=False)),
            "k_vector_mean_q01": float(torch.quantile(k_means, 0.01)),
            "k_vector_mean_q99": float(torch.quantile(k_means, 0.99)),
            "k_vector_std_avg": float(k_stds.mean()),
            "k_vector_std_std": float(k_stds.std(unbiased=False)),
            "k_vector_std_q01": float(torch.quantile(k_stds, 0.01)),
            "k_vector_std_q99": float(torch.quantile(k_stds, 0.99)),
        })
        # Advance the actual model state once; this keeps later-layer Q/K
        # statistics faithful to the checkpoint rather than reusing x.
        attn_out = attention(normalized)
        x = x + layer.residual_dropout(attn_out)
        moe_out, _, _ = layer.moe(layer.post_attention_norm(x), compute_router_losses=False)
        x = x + layer.residual_dropout(moe_out)
        vs = vector_stats[-1]
        print(
            f"layer={layer_idx:02d} n={scores.numel():,} "
            f"Q(mean={vs['q_vector_mean_avg']:.4f}, std={vs['q_vector_std_avg']:.4f}) "
            f"K(mean={vs['k_vector_mean_avg']:.4f}, std={vs['k_vector_std_avg']:.4f}) "
            f"QK[min={scores.min():.3f}, max={scores.max():.3f}]",
            flush=True,
        )
    return by_layer, vector_stats


def summarize(by_layer):
    rows = []
    all_scores = torch.cat(by_layer)
    qs = [0.001, 0.01, 0.05, 0.50, 0.95, 0.99, 0.999]
    for i, scores in enumerate(by_layer):
        quantiles = torch.quantile(scores, torch.tensor(qs))
        rows.append({
            "layer": i,
            "count": int(scores.numel()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            **{f"q{int(q*1000):03d}": float(v) for q, v in zip(qs, quantiles)},
        })
    overall_q = torch.quantile(all_scores, torch.tensor(qs))
    overall = {
        "count": int(all_scores.numel()),
        "min": float(all_scores.min()),
        "max": float(all_scores.max()),
        "mean": float(all_scores.mean()),
        "std": float(all_scores.std()),
        **{f"q{int(q*1000):03d}": float(v) for q, v in zip(qs, overall_q)},
    }
    return rows, overall


def plot(by_layer, rows, overall, output: Path):
    """Write a dependency-free SVG with histogram and per-layer quantiles."""
    flat = torch.cat(by_layer).numpy()
    lo, hi = float(np.quantile(flat, 0.001)), float(np.quantile(flat, 0.999))
    hist, edges = np.histogram(np.clip(flat, lo, hi), bins=100, range=(lo, hi))
    width, height = 1200, 620
    left, top, plot_w, plot_h = 70, 70, 510, 430
    max_h = max(int(hist.max()), 1)
    def sx(v): return left + (v - lo) / max(hi - lo, 1e-9) * plot_w
    bars = []
    bar_w = plot_w / len(hist)
    for i, count in enumerate(hist):
        h = count / max_h * plot_h
        bars.append(f'<rect x="{left+i*bar_w:.2f}" y="{top+plot_h-h:.2f}" width="{bar_w+0.2:.2f}" height="{h:.2f}" fill="#3182ce"/>')
    # Per-layer min/max and 1%/99% interval; each row is a horizontal range.
    x0, y0, w2, h2 = 680, 95, 440, 430
    global_lo, global_hi = overall['min'], overall['max']
    def sx2(v): return x0 + (v-global_lo)/max(global_hi-global_lo,1e-9)*w2
    lines = []
    for row in rows:
        y = y0 + row['layer'] * (h2 / max(len(rows)-1, 1))
        lines.append(f'<line x1="{sx2(row["min"]):.1f}" y1="{y:.1f}" x2="{sx2(row["max"]):.1f}" y2="{y:.1f}" stroke="#718096" stroke-width="2"/>')
        lines.append(f'<line x1="{sx2(row["q010"]):.1f}" y1="{y:.1f}" x2="{sx2(row["q990"]):.1f}" y2="{y:.1f}" stroke="#2b6cb0" stroke-width="7" stroke-linecap="round"/>')
        lines.append(f'<circle cx="{sx2(row["q500"]):.1f}" cy="{y:.1f}" r="4" fill="#c53030"/>')
        lines.append(f'<text x="{x0-18}" y="{y+4:.1f}" text-anchor="end" font-size="12">L{row["layer"]}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<style>text{{font-family:Arial,sans-serif;fill:#1a202c}} .axis{{stroke:#4a5568;stroke-width:1}} .grid{{stroke:#e2e8f0;stroke-width:1}}</style>
<text x="600" y="28" text-anchor="middle" font-size="18" font-weight="bold">Unscaled post-RoPE QKᵀ | n={overall['count']:,} | full range [{overall['min']:.2f}, {overall['max']:.2f}]</text>
<text x="325" y="55" text-anchor="middle" font-size="14">Histogram (clipped to 0.1%–99.9%)</text>
<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/><line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>
{''.join(bars)}
<text x="{left}" y="{top+plot_h+25}" font-size="12">{lo:.2f}</text><text x="{left+plot_w}" y="{top+plot_h+25}" text-anchor="end" font-size="12">{hi:.2f}</text>
<text x="325" y="{top+plot_h+50}" text-anchor="middle" font-size="13">QK product (no 1/sqrt(dₖ))</text>
<text x="900" y="55" text-anchor="middle" font-size="14">Per-layer: gray=min/max, blue=P1–P99, red=median</text>
<line class="axis" x1="{x0}" y1="{y0+h2}" x2="{x0+w2}" y2="{y0+h2}"/>{''.join(lines)}
<text x="{x0}" y="{y0+h2+28}" font-size="12">{global_lo:.2f}</text><text x="{x0+w2}" y="{y0+h2+28}" text-anchor="end" font-size="12">{global_hi:.2f}</text>
</svg>'''
    output.write_text(svg, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, config, ckpt = load_checkpoint(args.checkpoint, args.device)
    input_ids = load_samples(args.validation_bin, args.validation_sample_idx, args.num_samples, config.max_position_embeddings)
    by_layer, vector_stats = measure(model, config, input_ids, args.pairs_per_layer, args.device)
    rows, overall = summarize(by_layer)
    result = {
        "checkpoint": str(args.checkpoint),
        "optimizer_step": ckpt.get("optimizer_step"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "model_config": config.to_dict(),
        "measurement": {
            "description": "post-RoPE QK^T, before division by sqrt(head_dim); causal random pairs plus all diagonal pairs",
            "num_samples": int(input_ids.size(0)),
            "sequence_length": int(input_ids.size(1)),
            "pairs_per_layer_requested": args.pairs_per_layer,
            "head_dim": config.hidden_size // config.num_attention_heads,
        },
        "overall": overall,
        "layers": rows,
        "qk_vector_stats": vector_stats,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "qk_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(by_layer, rows, overall, args.output_dir / "qk_product_distribution.svg")
    print(json.dumps({"overall": overall, "output_dir": str(args.output_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
