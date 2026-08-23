"""DPO for the native Modern-MoE (ModernMoEForCausalLM).

Train-on-the-same-SFT-weight preference tuning using the DPO objective
(Rafailov et al. 2023) implemented directly on the native PyTorch model:
no Unsloth / Megatron / HF ``AutoModel`` involved.

Pipeline (per sample, done per-stream to keep absolute RoPE positions exact):
    dataset (prompt / chosen / rejected)
      -> Qwen3 ChatML render, assistant completion keeps its ground truth
      -> tokenize chosen-stream and rejected-stream separately
      -> policy forward (grad)  and  reference forward (bf16 no-grad)
      -> per-completion-token log-prob, summed -> DPO loss
      -> backward on the policy only; reference is frozen.

Usage (from the project root, in the moe-env conda env):
    python -u -m scripts.train_dpo --help
    python -u -m scripts.train_dpo \
        --checkpoint checkpoints/posttrain/sft/sft_nanogptmoe_v2_gqa_advanced_kernel_120m_lr3e4_v2/step_0002442.pt \
        --dataset Karsh-CAI/btfChinese-DPO-small \
        --split train \
        --out-dir checkpoints/posttrain/dpo/dpo_smoke \
        --max-samples 64

A W&B run is created under --wandb-project when provided.  Logging mirror the
SFT scripts (best/token scalars, a periodic loss/acc line) so the output reads
familiar to the rest of this project.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import secrets
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from modern_moe.config import ModernMoEConfig
from modern_moe.model import ModernMoEForCausalLM
from modern_moe.tokenizer import load_tokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def prune_checkpoints(out_dir: Path, max_checkpoints: int, losses: dict[int, float], latest_step: int):
    """Delete step checkpoints beyond ``max_checkpoints``.

    Retention mirrors the SFT behaviour: always keep the latest step and the
    best (lowest) DPO-loss step, then fill remaining slots from the most recent
    steps, unlinking the rest.  ``losses`` is an in-memory ``{step: loss}`` map
    maintained during training (avoids loading multi-GB files).  A small sidecar
    ``manifest.json`` is written so the best step survives restarts.
    """
    if max_checkpoints < 1:
        return
    existing = sorted({int(p.stem.split("_")[-1]) for p in out_dir.glob("step_*.pt")})
    if not existing:
        return
    best_step = min(existing, key=lambda s: losses.get(s, float("inf")))
    keep = {latest_step, best_step}
    for step in sorted(existing, reverse=True):
        if len(keep) >= max_checkpoints:
            break
        keep.add(step)
    for step in existing:
        if step in keep:
            continue
        fp = out_dir / f"step_{step:07d}.pt"
        if fp.is_file():
            fp.unlink()
    manifest = {
        "latest_step": latest_step,
        "best_step": best_step,
        "best_dpo_loss": losses.get(best_step),
        "max_checkpoints": max_checkpoints,
        "steps_kept": sorted(keep),
        "checkpoint_losses": {str(s): float(l) for s, l in losses.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def resolve_experiment(
    checkpoint_root: Path,
    experiment_id: str | None,
    resume_override: Path | None,
    resume_latest: bool,
) -> tuple[str, Path, Path | None]:
    """Return ``(experiment_id, output_dir, resume_path)`` mirroring SFT.

    Fresh run without an id: auto-generate ``exp_<ts>_<hex>`` so repeated launches
    never collide and each experiment keeps its own directory.  ``--resume`` uses
    the explicit file; ``--resume-latest`` picks the newest checkpoint under the
    experiment dir.  ``--experiment-id`` pins the output dir for stable resume.
    """
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def outdir(eid: str) -> Path:
        d = checkpoint_root / eid
        d.mkdir(parents=True, exist_ok=True)
        return d

    if resume_override is not None:
        rp = resume_override
        if not rp.is_file():
            raise FileNotFoundError(rp)
        eid = experiment_id or rp.parent.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", eid):
            raise ValueError(f"invalid experiment_id {eid!r}")
        return eid, outdir(eid), rp

    if resume_latest:
        if not experiment_id:
            raise SystemExit("--resume-latest requires --experiment-id")
        od = outdir(experiment_id)
        candidates: list[Path] = []
        final_path = od / "dpo_final.pt"
        if final_path.is_file():
            candidates.append(final_path)
        step_cands: list[tuple[int, Path]] = []
        for p in od.glob("step_*.pt"):
            m = re.fullmatch(r"step_(\d+)\.pt", p.name)
            if m:
                step_cands.append((int(m.group(1)), p))
        if step_cands:
            candidates.append(max(step_cands)[1])
        if not candidates:
            raise FileNotFoundError(f"no DPO checkpoint under {od}")
        return experiment_id, od, max(candidates, key=lambda p: p.stat().st_mtime_ns)

    # fresh run
    if experiment_id:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id):
            raise ValueError(f"invalid experiment_id {experiment_id!r}")
        return experiment_id, outdir(experiment_id), None
    while True:
        eid = f"exp_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(4)}"
        od = checkpoint_root / eid
        try:
            od.mkdir(parents=True)
            return eid, od, None
        except FileExistsError:
            continue


def load_resume_state(resume_path: Path):
    """Load a DPO checkpoint's model/optimizer/step/loss for continuation."""
    ck = torch.load(resume_path, map_location="cpu", weights_only=False)
    return ck


def build_model_from_checkpoint(
    checkpoint_path: Path, force_backend: str | None = None
) -> tuple[ModernMoEForCausalLM, dict]:
    """Rebuild the native model from a saved training checkpoint.

    Returns ``(policy, model_config_dict)``.  The reference model is a second
    instantiation of the same architecture sharing the same weights, made
    non-trainable; DPO keeps the reference frozen at the SFT weights.

    ``force_backend`` optionally overrides ``full_attention_backend`` so DPO can
    build every copy with ``sdpa`` (mask-friendly) regardless of what the
    checkpoint was trained with.  Weights are identical either way.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "model_config" not in ckpt:
        raise ValueError(f"Not a native Modern-MoE training checkpoint: {checkpoint_path}")

    raw_config = ckpt["model_config"]
    config_dict = raw_config.to_dict() if hasattr(raw_config, "to_dict") else dict(raw_config)
    if force_backend is not None:
        config_dict["full_attention_backend"] = force_backend
    model_config = ModernMoEConfig(**config_dict)
    del config_dict["architecture_name"]  # informational only

    model = ModernMoEForCausalLM(model_config)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys while loading checkpoint: {unexpected[:10]}")
    if missing:
        raise ValueError(f"Missing keys while loading checkpoint: {missing[:10]}")

    model_config_dict = model_config.to_dict()
    return model, model_config_dict


# ---------------------------------------------------------------------------
# Data: HuggingFace dataset -> {prompt, chosen, rejected}
# ---------------------------------------------------------------------------
def build_conversations(
    tokenizer,
    question: str,
    chosen: str,
    rejected: str,
):
    """Render the user prompt plus each answer as a Qwen3 ChatML message list."""
    user_turn = [{"role": "user", "content": question}]
    chosen_messages = user_turn + [{"role": "assistant", "content": chosen}]
    rejected_messages = user_turn + [{"role": "assistant", "content": rejected}]
    return chosen_messages, rejected_messages


def tokenize_stream(tokenizer, messages, max_length: int):
    """Encode one chat message list, giving (ids, response_start).

    ``response_start`` is the index (into ``ids``) of the first completion
    token, i.e. the first position whose predicted-next-token belongs to the
    assistant answer and therefore participates in the DPO log-prob.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    ids = tokenizer(text, add_special_tokens=False).input_ids
    # Locate the first assistant turn's content so the loss-mask only covers the
    # completion, never the (shared) prompt.
    cursor = 0
    response_start = None
    while True:
        start = text.find(f"{IM_START}assistant", cursor)
        if start < 0:
            break
        if response_start is None:
            # tokenize the prefix through the assistant marker to get its length
            prefix_text = text[: start + len(f"{IM_START}assistant")]
            prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
            response_start = len(prefix_ids)
        cursor = start + 1

    if response_start is None:
        raise ValueError("no assistant turn found in rendered template")
    if len(ids) > max_length:
        # keep the head/premble and truncate the tail (completion) to budget
        ids = ids[:max_length]
        response_start = min(response_start, max_length)
    return ids, response_start


# ---------------------------------------------------------------------------
# DPO loss — concatenated forward (one batched policy pass per bucket)
# ---------------------------------------------------------------------------
def collate_bucket(stream_ids, stream_starts, device, max_length=-1):
    """Right-pad a list of variable-length token streams into one ``[B, L]`` batch.

    Returns ``(ids[B, L], mask[B, L] bool, starts[B])``.  ``starts[i]`` is the
    index of the first completion token in row ``i``.  All rows are padded to
    the longest sequence in this bucket, so one ``model(ids, mask)`` SDPA pass
    serves the whole bucket with independent per-row causal attention (no
    cross-sequence leakage — verified against the per-sample path).
    """
    lengths = [len(s) for s in stream_ids]
    max_len = max(lengths)
    if max_length > 0:
        max_len = min(max_len, max_length)
        lengths = [min(l, max_length) for l in lengths]
    B = len(stream_ids)
    ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
    starts = torch.zeros(B, dtype=torch.long, device=device)
    for i, (seq, sl, st) in enumerate(zip(stream_ids, lengths, stream_starts)):
        ids[i, :sl] = torch.tensor(seq[:sl], dtype=torch.long, device=device)
        mask[i, :sl] = True
        starts[i] = min(st, sl)
    return ids, mask, starts


def compute_batch_completion_logprob(model, ids, mask, starts):
    """Per-row completion log-prob (summed over the completion tokens only).

    One forward on the whole ``[B, L]`` padded batch.  To avoid materializing
    the full ``[B, L, V]`` logits (OOB on `data`), we ask the model for the
    final hidden states + lm-head weight and run **Apple CCE with
    ``reduction="none"``** to get per-token NLL directly, exactly the DPO path
    their docs describe.  We keep only source positions ``>= starts[row] - 1``
    (first completion token ``ids[start]`` is predicted just before it) within
    the valid mask.  Returns ``logp[B]``.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, attention_mask=mask, return_loss_hidden_states=True)
    B, L = ids.shape
    # targets must align 1:1 with embedding rows (B*L): position p predicts
    # np.random.next token ids[p+1].  Right-shift ids and put a dummy in the last
    # column (we mask it out via `keep` below, before any aggregation).
    targets = ids.clone()                     # [B, L]
    targets[:, :-1] = ids[:, 1:]
    targets[:, -1] = ids[:, -1].clone()       # placeholder; never used
    emb = out.loss_hidden_states.contiguous().flatten(0, -2)  # [B*L, H]
    w = out.classifier_weight                                # [V, H]
    from cut_cross_entropy.cce import CCEParams, _handle_eps, linear_cross_entropy_apply

    cce_dtype = emb.dtype
    params = CCEParams(
        targets=targets.contiguous().flatten(),
        valids=None,
        softcap=None,
        reduction="none",
        filter_eps=_handle_eps("auto", cce_dtype),
        shift=0,
        batch_shape=targets.size(),      # (B, L)
        accum_e_fp32=False,
        accum_c_fp32=False,
        filter_e_grad=True,
        filter_c_grad=True,
        vocab_parallel_options=None,
        return_lse=False,
    )
    nll, _ = linear_cross_entropy_apply(emb, w, None, params)   # [B, L]
    # nll[p] = -log P(ids[p+1] | prefix).  Completion token ids[start] is
    # predicted by source position start-1.  Keep p in [starts[i]-1, L-2] and
    # p must be a real (non-pad) source position.
    pos = torch.arange(L, device=ids.device).unsqueeze(0)        # [1, L]
    keep = (pos >= starts.unsqueeze(1) - 1) & (pos <= L - 2) & mask
    return -(nll * keep).sum(dim=1)                            # [B] = +sum logp over completion


# ---------------------------------------------------------------------------
# Varlen (FA3) concatenated path — optional stage-2 accelerator.
# ---------------------------------------------------------------------------
def collate_varlen(stream_ids, stream_starts, device):
    """Pack variable-length streams into one padding-free flat batch.

    Returns ``(flat_ids[N], cu_seqlens[n+1], positions[N], seq_starts[n])``.
    ``positions`` resets to 0 at each sequence boundary (native RoPE requirement
    for a concatenated stream).  ``seq_starts`` are the completion start logged
    per sequence (input-relative), used for the completion log-prob mask.
    """
    flat = []
    cu = [0]
    pos = []
    seq_starts = []
    for s, st in zip(stream_ids, stream_starts):
        flat += list(s)
        pos += list(range(len(s)))          # reset position at each sequence
        cu.append(cu[-1] + len(s))
        seq_starts.append(int(st))
    N = len(flat)
    return (
        torch.tensor(flat, dtype=torch.long, device=device),
        torch.tensor(cu, dtype=torch.long, device=device),
        torch.tensor(pos, dtype=torch.long, device=device),
        seq_starts,
    )


def compute_completion_logprob_varlen(model, flat_ids, cu_seqlens, positions, seq_starts):
    """Per-sequence completion log-prob under the padding-free FA3 path.

    Mirrors ``compute_batch_completion_logprob`` but on a flat ``[N]`` stream
    with ``packed_seq_params``; the completion-mask uses the per-sequence
    cumulative offsets ``cu_seqlens`` and logged ``seq_starts``.
    """
    from modern_moe.layers import PackedSeqParams

    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=flat_ids,
            packed_seq_params=PackedSeqParams(cu_seqlens, max_seqlen, positions),
            return_loss_hidden_states=True,
        )
    N = flat_ids.shape[0]
    targets = flat_ids.clone()
    targets[:-1] = flat_ids[1:]
    targets[-1] = flat_ids[-1]
    emb = out.loss_hidden_states.contiguous().flatten(0, -2)   # [N, H]
    w = out.classifier_weight
    from cut_cross_entropy.cce import CCEParams, _handle_eps, linear_cross_entropy_apply

    params = CCEParams(
        targets=targets.contiguous().flatten(),
        valids=None,
        softcap=None,
        reduction="none",
        filter_eps=_handle_eps("auto", emb.dtype),
        shift=0,
        batch_shape=(N,),
        accum_e_fp32=False,
        accum_c_fp32=False,
        filter_e_grad=True,
        filter_c_grad=True,
        vocab_parallel_options=None,
        return_lse=False,
    )
    nll, _ = linear_cross_entropy_apply(emb, w, None, params)   # [N]
    nll = nll.float()
    n_seqs = len(seq_starts)
    cu = cu_seqlens.tolist()
    logp = torch.zeros(n_seqs, device=flat_ids.device, dtype=torch.float32)
    for i in range(n_seqs):
        src_lo = cu[i] + seq_starts[i] - 1     # completion token cu[i]+starts[i] predicted here
        src_hi = cu[i + 1] - 1                 # last prediction of this sequence
        idx = torch.arange(src_lo, src_hi, device=flat_ids.device)
        keep = (idx >= 0) & (idx <= N - 2)
        logp[i] = -(nll[idx[keep]]).sum()
    return logp


def dpo_batch_loss_varlen(model, flat_ids, cu_seqlens, positions, seq_chosen_starts, ref_logp, beta, label_smoothing=0.0):
    """DPO loss over one padding-free bucket (``B`` chosen first, then ``B`` rejected).

    Mirrors ``dpo_batch_loss`` for the FA3 varlen path.
    """
    B = len(seq_chosen_starts) // 2
    logp = compute_completion_logprob_varlen(model, flat_ids, cu_seqlens, positions,
                                             [s for s in seq_chosen_starts])
    logp = logp - torch.as_tensor(ref_logp, device=logp.device)
    chosen = logp[:B]
    rejected = logp[B:]
    ratio = chosen - rejected
    loss = -F.logsigmoid(beta * ratio)
    if label_smoothing > 0:
        loss = (1 - label_smoothing) * loss - label_smoothing * F.logsigmoid(-beta * ratio)
    acc = (ratio > 0).float().mean().item()
    return loss.mean(), acc


def dpo_batch_loss(model, bucket_ids, bucket_mask, bucket_starts, ref_logp, beta, label_smoothing=0.0):
    """DPO loss over one concatenated bucket.

    ``bucket_ids``/``bucket_mask`` hold the **chosen+rejected** streams of ``B``
    pairs in a single ``[2B, L]`` padded batch (chosen rows first, then
    rejected); ``bucket_starts`` is ``[2B]`` completion starts and ``ref_logp``
    is the parallel ``[2B]`` array of precomputed reference log-probs.  One
    policy forward serves the whole bucket.  Returns ``(loss, acc)``.
    """
    B = bucket_ids.shape[0] // 2
    logp = compute_batch_completion_logprob(model, bucket_ids, bucket_mask, bucket_starts)
    logp = logp - torch.as_tensor(ref_logp, device=logp.device)
    chosen_logp = logp[:B]
    rejected_logp = logp[B:]
    ratio = chosen_logp - rejected_logp          # [B]
    loss = -F.logsigmoid(beta * ratio)            # [B]
    if label_smoothing > 0:
        loss = (1 - label_smoothing) * loss - label_smoothing * F.logsigmoid(-beta * ratio)
    acc = (ratio > 0).float().mean().item()
    return loss.mean(), acc


def precompute_ref_logprobs(reference, stream, device, backend="varlen", **bucket_kwargs):
    """Cache reference (chosen_logp, rejected_logp) for every sample once.

    The reference is frozen, so its log-probs never change during training.
    Uses the same concatenated forward style as training (varlen FA3 or padded
    SDPA) so precompute is as fast as a training pass, leaving no reference
    forward in the loop.
    """
    chosen_ids = []
    chosen_starts = []
    rejected_ids = []
    rejected_starts = []
    for c_ids, c_start, r_ids, r_start in stream:
        chosen_ids.append(list(c_ids))
        chosen_starts.append(int(c_start))
        rejected_ids.append(list(r_ids))
        rejected_starts.append(int(r_start))

    ref_pairs = [None] * len(stream)
    reference.eval()
    chunk_pairs = 64  # process reference in chunks like the training bucket size
    with torch.no_grad():
        if backend == "varlen":
            n = len(stream)
            for b in range(0, n, chunk_pairs):
                e = min(b + chunk_pairs, n)
                chunk_ci = chosen_ids[b:e] + rejected_ids[b:e]
                chunk_cs = chosen_starts[b:e] + rejected_starts[b:e]
                flat_ids, cu, positions, starts = collate_varlen(chunk_ci, chunk_cs, device)
                logp = compute_completion_logprob_varlen(
                    reference, flat_ids, cu, positions, starts
                )
                k = e - b
                for i in range(b, e):
                    ref_pairs[i] = (logp[i - b].item(), logp[i - b + k].item())
        else:
            ids_c, mask_c, starts_c = collate_bucket(chosen_ids, chosen_starts, device, **bucket_kwargs)
            logp_c = compute_batch_completion_logprob(reference, ids_c, mask_c, starts_c)
            ids_r, mask_r, starts_r = collate_bucket(rejected_ids, rejected_starts, device, **bucket_kwargs)
            logp_r = compute_batch_completion_logprob(reference, ids_r, mask_r, starts_r)
            for i in range(len(stream)):
                ref_pairs[i] = (logp_c[i].item(), logp_r[i].item())
    return ref_pairs


# ---------------------------------------------------------------------------
# CLI / logging
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO for native Modern-MoE.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Native training checkpoint whose weights seed policy + reference.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Karsh-CAI/btfChinese-DPO-small",
        help="HuggingFace dataset id containing prompted chosen/rejected columns.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints/posttrain/dpo/dpo_smoke"),
        help="DEPRECATED: kept for back-compat. Use checkpoint-root + experiment-id; "
        "the effective output dir = checkpoint_root/<experiment_id>.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoints/posttrain/dpo"),
        help="Root under which each experiment writes its own dir. "
        "Effective out dir = checkpoint_root/<experiment_id>.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = whole dataset")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for non-router policy parameters.")
    parser.add_argument("--router-lr", type=float, default=2e-5,
                        help="Learning rate for router parameters (default: 0.2 * --lr).")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--save-every-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--wandb-project", type=str, default="nanogpt-moe-dpo")
    parser.add_argument("--run-name", type=str, default="dpo-native-moe")
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=4,
        help="Keep at most this many step checkpoint files in out_dir: always "
        "the latest plus the best-DPO-loss, then the most recent, dropping the "
        "rest by unlink (mirrors the SFT retention behaviour).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["varlen", "sdpa"],
        default="varlen",
        help="'varlen' = padding-free FA3 packed_seq_paths; 'sdpa' = padded SDPA "
        "baseline.  Default varlen.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=Path("data/cache/hf"),
        help="Writable HF hub cache for dataset downloads (defaults to a "
        "project-local dir; the default ~/.cache/huggingface is read-only here).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load models + tokenize one sample + one forward/backward, then exit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config overriding defaults (mirrors the SFT config layout).",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Stable experiment id used for the output dir (checkpoint_root/<id>). "
        "If omitted a fresh timestamped id is generated on a fresh run, so repeated "
        "launches of the same command never collide.  Required to --resume-latest.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a specific DPO checkpoint file (model/optimizer/step).",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the most recently written checkpoint under the experiment "
        "directory (latest step_*.pt or dpo_final.pt, by mtime).",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def load_yaml_overrides(path: Path) -> dict:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    args = parse_args()
    # CLI explicitly-provided options win over YAML defaults; YAML wins over the
    # argparse built-in defaults.  Detect explicit CLI flags by scanning argv.
    explicit = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            name = token.split("=", 1)[0]
            explicit.add(name.replace("-", "_"))
            arg = name.split(".")[0]
            explicit.add(arg.replace("-", "_"))
    if args.config is not None and args.config.exists():
        overrides = load_yaml_overrides(args.config)
        path_keys = {"checkpoint", "out_dir", "config", "hf_cache_dir", "checkpoint_root"}
        # Coerce numeric params whose YAML value came back as a str (older PyYAML
        # parses `1e-5` as a string on some versions).  Use the argparse `type`
        # so lr/beta/etc. land as the right Python type.
        numeric_type = {}
        for action in _build_parser()._actions:
            if action.type in (int, float) and action.dest != "help":
                numeric_type[action.dest] = action.type
        for key, value in overrides.items():
            if not hasattr(args, key):
                raise ValueError(f"--config has unknown key {key!r}")
            cli_flag = f"--{key.replace('_', '-')}"
            if cli_flag in sys.argv or key in explicit:
                continue  # explicit CLI value takes precedence
            if key in path_keys and isinstance(value, str):
                value = Path(value)
            elif key in numeric_type and isinstance(value, str):
                try:
                    value = numeric_type[key](value)
                except (TypeError, ValueError):
                    pass  # leave as-is if uncoercible
            setattr(args, key, value)
    if args.checkpoint is None:
        raise SystemExit(
            "missing --checkpoint (or 'checkpoint:' in the --config yaml)"
        )

    # Resolve experiment output dir + resume target (mirrors SFT).
    experiment_id, output_dir, resume_path = resolve_experiment(
        args.checkpoint_root, args.experiment_id, args.resume, args.resume_latest
    )
    args.out_dir = output_dir
    args.experiment_id = experiment_id
    print(f"[dpo] experiment_id={experiment_id}")
    print(f"[dpo] output_dir={output_dir}")
    if resume_path is not None:
        print(f"[dpo] RESUME from {resume_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dpo] device={device}")
    if device == "cuda":
        torch.cuda.set_device(0)

    # --- model & reference --------------------------------------------------
    # Two concatenated-forward backends:
    #   * varlen : padding-free FA3 packed_seq_params (default). Build with
    #              full_attention_backend=flash_attn (the _forward_varlen path).
    #   * sdpa   : padded [2B, L] batch + attention_mask (mask-aware), the
    #              correctness baseline. Build with full_attention_backend=sdpa.
    # Weights: fresh runs seed policy + reference from the SFT --checkpoint.
    # On resume, only policy comes from the DPO checkpoint; the reference must
    # remain the original frozen SFT model for the whole DPO experiment.
    force_backend = "flash_attn" if args.backend == "varlen" else "sdpa"
    resume_ckpt = load_resume_state(resume_path) if resume_path is not None else None
    base_ckpt = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if "model" not in base_ckpt:
        raise ValueError(f"Not a native Modern-MoE checkpoint: {args.checkpoint}")
    seed_model = resume_ckpt["model"] if resume_ckpt is not None else base_ckpt["model"]
    policy, config_dict = build_model_from_checkpoint(args.checkpoint, force_backend=force_backend)
    policy.load_state_dict(seed_model, strict=True)
    reference = ModernMoEForCausalLM(ModernMoEConfig(**config_dict))
    reference.load_state_dict(base_ckpt["model"], strict=True)
    for p in reference.parameters():
        p.requires_grad_(False)
    policy = policy.to(device)
    reference = reference.to(device)
    print(
        f"[dpo] model params={policy.num_parameters():,} "
        f"lsize={config_dict.get('hidden_size')} layers={config_dict.get('num_hidden_layers')}"
    )

    tokenizer = load_tokenizer(ModernMoEConfig(**config_dict), Path("tokenizer/qwen3_moe"))
    print(f"[dpo] tokenizer vocab={len(tokenizer)}")

    # --- data ---------------------------------------------------------------
    # Use a project-local, writable HF cache (the default ~/.cache is read-only
    # in this environment/sandbox).
    hf_cache = args.hf_cache_dir
    hf_cache.mkdir(parents=True, exist_ok=True)
    (hf_cache / "hub").mkdir(parents=True, exist_ok=True)
    (hf_cache / "datasets").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    from datasets import load_dataset

    try:
        ds = load_dataset(args.dataset, split=args.split, cache_dir=str(hf_cache))
    except Exception as e:
        # help surface the common "cache dir read-only" failure clearly
        raise SystemExit(f"failed to load dataset {args.dataset}: {e}")
    rows = list(ds)
    if args.max_samples and args.max_samples > 0:
        random.shuffle(rows)
        rows = rows[: args.max_samples]
    print(f"[dpo] samples={len(rows)}")

    # warm up tokenize once so a failure surfaces before training
    stream = []
    for r in rows:
        q, c, rc = r["question"], r["chosen"], r["rejected"]
        cm, rm = build_conversations(tokenizer, q, c, rc)
        c_ids, c_start = tokenize_stream(tokenizer, cm, args.max_length)
        r_ids, r_start = tokenize_stream(tokenizer, rm, args.max_length)
        stream.append((c_ids, c_start, r_ids, r_start))
    print("[dpo] tokenize warm-up OK")

    # Cache reference log-probs once (reference is frozen across the run).
    print(f"[dpo] precomputing reference log-probs (backend={args.backend})...")
    ref_pairs = precompute_ref_logprobs(reference, stream, device, backend=args.backend)
    print(f"[dpo] reference cache done ({len(ref_pairs)} pairs)")

    if args.dry_run:
        # One real policy forward + backward (chosen+rejected in one bucket) to
        # catch load/rope/moe CUDA issues before committing to a full run.
        policy.train()
        chosen_ids = [c_ids]
        chosen_st = [c_start]
        rej_ids = [r_ids]
        rej_st = [r_start]
        refs = [ref_pairs[0][0], ref_pairs[0][1]]
        if args.backend == "varlen":
            flat_ids, cu, positions, starts = collate_varlen(
                chosen_ids + rej_ids, chosen_st + rej_st, device
            )
            loss, acc = dpo_batch_loss_varlen(
                policy, flat_ids, cu, positions, starts,
                refs, args.beta, args.label_smoothing,
            )
        else:
            bucket_ids, bucket_mask, bucket_starts = collate_bucket(
                chosen_ids + rej_ids, chosen_st + rej_st, device, max_length=args.max_length
            )
            loss, acc = dpo_batch_loss(
                policy, bucket_ids, bucket_mask, bucket_starts,
                refs, args.beta, args.label_smoothing,
            )
        loss.backward()
        print(f"[dpo] DRY-RUN OK backend={args.backend} loss={loss.item():.4f} acc={acc:.3f}")
        print(
            f"[dpo] loaded model + tokenizer + dataset, concatenated forward/backward "
            f"({args.backend}) path works. Exiting (--dry-run)."
        )
        return

    trainable = [p for p in policy.parameters() if p.requires_grad]
    router_parameters = []
    other_parameters = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        if "router" in name:
            router_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    if not other_parameters or not router_parameters:
        raise RuntimeError(
            "DPO optimizer parameter split failed: "
            f"other={len(other_parameters)} router={len(router_parameters)}"
        )
    optim = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": args.lr, "router_group": False},
            {"params": router_parameters, "lr": args.router_lr, "router_group": True},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    print(
        f"[dpo] optimizer groups: other={sum(p.numel() for p in other_parameters):,} "
        f"lr={args.lr:.2e}, router={sum(p.numel() for p in router_parameters):,} "
        f"lr={args.router_lr:.2e}"
    )
    global_step = 0
    tokens_seen = 0
    checkpoint_losses: dict[int, float] = {}
    if resume_ckpt is not None:
        if "optimizer" in resume_ckpt:
            try:
                optim.load_state_dict(resume_ckpt["optimizer"])
            except ValueError as exc:
                # Older DPO checkpoints had one parameter group.  Keep the
                # model weights and restart AdamW moments with the new split.
                print(f"[dpo] warning: optimizer state not compatible; restarting moments ({exc})")
        global_step = int(resume_ckpt.get("global_step", 0))
        tokens_seen = int(resume_ckpt.get("tokens_seen", 0))
        # Recover per-step losses from the existing manifest if present.
        manifest_p = args.out_dir / "manifest.json"
        if manifest_p.is_file():
            try:
                prev = json.loads(manifest_p.read_text(encoding="utf-8"))
                checkpoint_losses = {
                    int(s): float(l) for s, l in (prev.get("checkpoint_losses") or {}).items()
                }
            except (json.JSONDecodeError, TypeError, ValueError):
                checkpoint_losses = {}
        print(f"[dpo] resumed at global_step={global_step}, tokens_seen={tokens_seen}")

    # --- training loop ------------------------------------------------------
    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup_steps and args.warmup_steps > 0:
        warmup = args.warmup_steps
        t_max = max(1, args.max_samples // (args.micro_batch_size * args.gradient_accumulation_steps) * args.epochs)

        def lr_at(step, peak_lr):
            if step < warmup:
                return peak_lr * (step + 1) / warmup
            return peak_lr

    else:
        lr_at = lambda step, peak_lr: peak_lr  # noqa: E731

    n_samples = len(stream)
    bucket_size = args.micro_batch_size                # PAIRS per bucket
    pairs_per_step = bucket_size * args.gradient_accumulation_steps
    policy.train()
    print(f"[dpo] begin training (backend={args.backend}, bucket_size={bucket_size}, "
          f"pairs_per_step={pairs_per_step})")
    total_steps_per_epoch = max(1, n_samples // pairs_per_step)
    print(f"[dpo] samples={n_samples}, pairs/step={pairs_per_step} "
          f"=> ~{total_steps_per_epoch} steps/epoch, ~{total_steps_per_epoch * args.epochs} total steps")
    start_wall = time.time()
    running_loss = 0.0
    running_acc = 0.0
    running_steps = 0
    epoch_perm = list(range(n_samples))

    for epoch in range(args.epochs):
        random.shuffle(epoch_perm)
        for start_idx in range(0, n_samples, pairs_per_step):
            step_idx = epoch_perm[start_idx : start_idx + pairs_per_step]
            if len(step_idx) < bucket_size:
                break
            step_t0 = time.time()
            optim.zero_grad(set_to_none=True)
            n_buckets = len(step_idx) // bucket_size
            micro_losses = []
            for w in range(n_buckets):
                pair_idx = step_idx[w * bucket_size : (w + 1) * bucket_size]
                chosen_ids = [stream[i][0] for i in pair_idx]
                chosen_st = [stream[i][1] for i in pair_idx]
                rej_ids = [stream[i][2] for i in pair_idx]
                rej_st = [stream[i][3] for i in pair_idx]
                refs = []
                for i in pair_idx:
                    refs.append(ref_pairs[i][0])
                for i in pair_idx:
                    refs.append(ref_pairs[i][1])
                if args.backend == "varlen":
                    flat_ids, cu, positions, starts = collate_varlen(
                        chosen_ids + rej_ids, chosen_st + rej_st, device
                    )
                    loss, acc = dpo_batch_loss_varlen(
                        policy, flat_ids, cu, positions, starts,
                        refs, args.beta, args.label_smoothing,
                    )
                else:
                    bucket_ids, bucket_mask, bucket_starts = collate_bucket(
                        chosen_ids + rej_ids, chosen_st + rej_st, device,
                        max_length=args.max_length,
                    )
                    loss, acc = dpo_batch_loss(
                        policy, bucket_ids, bucket_mask, bucket_starts,
                        refs, args.beta, args.label_smoothing,
                    )
                # Average the micro-batch losses before accumulating.  Without
                # this division, gradient_accumulation_steps also multiplies
                # the effective learning rate.
                (loss / n_buckets).backward()
                micro_losses.append((loss.item(), acc))
            # average micro losses for reporting
            mean_loss = sum(l for l, _ in micro_losses) / len(micro_losses)
            mean_acc = sum(a for _, a in micro_losses) / len(micro_losses)

            for g in optim.param_groups:
                peak_lr = args.router_lr if g.get("router_group", False) else args.lr
                g["lr"] = lr_at(global_step, peak_lr)
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optim.step()

            tokens_seen += sum(len(stream[i][0]) + len(stream[i][2]) for i in step_idx)
            running_loss += mean_loss
            running_acc += mean_acc
            running_steps += 1
            step_dt = time.time() - step_t0

            if (global_step + 1) % args.log_every_steps == 0 or global_step == 0:
                elapsed = time.time() - start_wall
                print(
                    f"[dpo] step {global_step + 1} loss {running_loss / running_steps:.4f} "
                    f"acc {running_acc / running_steps:.3f} lr {optim.param_groups[0]['lr']:.2e} "
                    f"step {step_dt:.1f}s  wall {elapsed:.1f}s"
                )
                if run is not None:
                    run.log(
                        {
                            "dpo_loss": running_loss / running_steps,
                            "dpo_acc": running_acc / running_steps,
                            "lr": optim.param_groups[0]["lr"],
                            "step": global_step,
                        }
                    )
                running_loss = 0.0
                running_acc = 0.0
                running_steps = 0

            global_step += 1
            # Save periodically (like SFT). Track per-step loss so retention can
            # keep max_checkpoints files by best-loss; then prune.
            step_loss = mean_loss
            if (global_step % args.save_every_steps) == 0 or global_step == 1:
                save_path = out_dir / f"step_{global_step:07d}.pt"
                torch.save(
                    {
                        "model": policy.state_dict(),
                        "model_config": ModernMoEConfig(**config_dict).to_dict(),
                        "train_config": vars(args),
                        "optimizer": optim.state_dict(),
                        "global_step": global_step,
                        "tokens_seen": tokens_seen,
                        "dpo_loss": float(step_loss),
                    },
                    save_path,
                )
            checkpoint_losses[global_step] = float(step_loss)
            prune_checkpoints(out_dir, args.max_checkpoints, checkpoint_losses, global_step)
    # final durable save (kept separately; does not count toward the 4-file cap)
    save_path = out_dir / "dpo_final.pt"
    torch.save(
        {
            "model": policy.state_dict(),
            "model_config": ModernMoEConfig(**config_dict).to_dict(),
            "train_config": vars(args),
            "optimizer": optim.state_dict(),
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "dpo_loss": float(mean_loss),
        },
        save_path,
    )
    print(f"[dpo] saved {save_path}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
