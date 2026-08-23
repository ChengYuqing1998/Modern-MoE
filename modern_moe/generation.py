from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch

from .model import ModernMoEForCausalLM

try:
    from nanok3.model import NanoK3ForCausalLM
except ImportError:
    NanoK3ForCausalLM = None


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    mode: Literal["no_cache", "cache", "mtp"] = "cache"
    max_cache_length: Optional[int] = None
    cuda_graph_decode: bool = False
    fused_sampling: bool = False
    flashinfer_sampling: bool = False


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    new_tokens: int
    elapsed_seconds: float
    tokens_per_second: float
    prefill_seconds: float = 0.0
    time_to_first_token_seconds: float = 0.0
    decode_seconds: float = 0.0
    decode_tokens_per_second: float = 0.0
    decode_graph_setup_seconds: float = 0.0
    mtp_proposed: int = 0
    mtp_accepted: int = 0

    @property
    def mtp_acceptance_rate(self) -> float:
        return self.mtp_accepted / max(1, self.mtp_proposed)


def _filtered_probabilities(
    logits: torch.Tensor,
    history: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    logits = logits.float().clone()
    if config.repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if config.repetition_penalty != 1.0:
        for batch_index in range(logits.size(0)):
            seen = torch.unique(history[batch_index])
            selected = logits[batch_index, seen]
            selected = torch.where(
                selected < 0,
                selected * config.repetition_penalty,
                selected / config.repetition_penalty,
            )
            logits[batch_index, seen] = selected
    ngram_size = int(config.no_repeat_ngram_size)
    if ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be non-negative")
    if ngram_size >= 2 and history.size(1) >= ngram_size - 1:
        # Ban a token if appending it would repeat an n-gram already present
        # in the prompt/generated history.  The short Python loop is only
        # over the generated sequence length and runs on the inference path,
        # not the training MoE dispatch path.
        for batch_index in range(logits.size(0)):
            tokens = history[batch_index].tolist()
            prefix = tuple(tokens[-(ngram_size - 1):])
            banned = {
                tokens[index + ngram_size - 1]
                for index in range(len(tokens) - ngram_size + 1)
                if tuple(tokens[index:index + ngram_size - 1]) == prefix
            }
            if banned:
                logits[batch_index, list(banned)] = float("-inf")
    if config.temperature <= 0:
        return torch.nn.functional.one_hot(
            logits.argmax(dim=-1),
            num_classes=logits.size(-1),
        ).float()
    logits /= config.temperature
    if config.top_k > 0:
        threshold = logits.topk(min(config.top_k, logits.size(-1))).values[:, -1:]
        logits.masked_fill_(logits < threshold, float("-inf"))
    if 0 < config.top_p < 1:
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > config.top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits.masked_fill_(remove, float("-inf"))
        logits.fill_(float("-inf")).scatter_(1, sorted_indices, sorted_logits)
    return logits.softmax(dim=-1)


def _sample(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probabilities, num_samples=1)


def _speculative_sample(
    target: torch.Tensor,
    draft: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate = _sample(draft)
    p = target.gather(1, candidate)
    q = draft.gather(1, candidate).clamp_min(1e-12)
    accepted = torch.rand_like(p) <= torch.minimum(
        torch.ones_like(p),
        p / q,
    )
    residual = (target - draft).clamp_min(0)
    residual_sum = residual.sum(dim=-1, keepdim=True)
    fallback = torch.where(
        residual_sum > 0,
        residual / residual_sum.clamp_min(1e-12),
        target,
    )
    replacement = _sample(fallback)
    return torch.where(accepted, candidate, replacement), accepted


@torch.inference_mode()
def generate(
    model: ModernMoEForCausalLM | "NanoK3ForCausalLM",
    input_ids: torch.Tensor,
    config: GenerationConfig,
    eos_token_id: Optional[int] = None,
    stream_callback: Optional[Callable[[torch.Tensor], None]] = None,
) -> GenerationResult:
    if model.training:
        raise RuntimeError("Call model.eval() before generate()")
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError("Generation currently supports one prompt at a time")
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if NanoK3ForCausalLM is not None and isinstance(model, NanoK3ForCausalLM):
        if config.mode == "mtp":
            raise ValueError("nanoK3 has no trained MTP draft layer")
    max_cache_length = config.max_cache_length or (
        input_ids.size(1) + config.max_new_tokens
    )
    if max_cache_length > model.config.max_position_embeddings:
        raise ValueError("Requested generation exceeds max_position_embeddings")

    if input_ids.is_cuda:
        torch.cuda.synchronize(input_ids.device)
    start = time.perf_counter()
    generated = input_ids
    proposed = accepted_count = 0
    prefill_seconds = 0.0
    time_to_first_token = 0.0

    if config.mode == "no_cache":
        for token_index in range(config.max_new_tokens):
            output = model(generated)
            if token_index == 0:
                if input_ids.is_cuda:
                    torch.cuda.synchronize(input_ids.device)
                prefill_seconds = time.perf_counter() - start
            probabilities = _filtered_probabilities(
                output.logits[:, -1],
                generated,
                config,
            )
            token = _sample(probabilities)
            generated = torch.cat((generated, token), dim=1)
            if token_index == 0:
                if input_ids.is_cuda:
                    torch.cuda.synchronize(input_ids.device)
                time_to_first_token = time.perf_counter() - start
            if stream_callback is not None:
                stream_callback(token)
            if eos_token_id is not None and token.item() == eos_token_id:
                break
    else:
        main = model.forward_inference(
            generated,
            max_cache_length=max_cache_length,
        )
        mtp_cache = None
        draft_logits = None
        decode_graph = None
        decode_graph_setup_seconds = 0.0
        if config.mode == "mtp":
            if input_ids.size(1) > 1:
                warm = model.mtp_draft(
                    main.hidden_states[:, :-1],
                    input_ids[:, 1:],
                    max_cache_length=max_cache_length,
                )
                mtp_cache = warm.cache
                draft_logits = warm.logits[:, -1]
        if input_ids.is_cuda:
            torch.cuda.synchronize(input_ids.device)
        prefill_seconds = time.perf_counter() - start

        fused_sampler = None
        if config.fused_sampling or config.flashinfer_sampling:
            if config.mode == "mtp":
                raise ValueError("fused sampling does not support MTP mode")
            from .fused_sampling import FusedSampler

            fused_sampler = FusedSampler(
                generated,
                config,
                max_cache_length + 1,
                main.logits.size(-1),
                backend="flashinfer" if config.flashinfer_sampling else "triton",
            )

        for token_index in range(config.max_new_tokens):
            if fused_sampler is not None:
                token = fused_sampler.sample(main.logits[:, -1])
            else:
                target = _filtered_probabilities(
                    main.logits[:, -1],
                    generated,
                    config,
                )
            if fused_sampler is None and config.mode == "mtp" and draft_logits is not None:
                draft = _filtered_probabilities(
                    draft_logits,
                    generated,
                    config,
                )
                token, accepted = _speculative_sample(target, draft)
                proposed += 1
                accepted_count += int(accepted.item())
            elif fused_sampler is None:
                token = _sample(target)

            previous_hidden = main.hidden_states[:, -1:]
            generated = torch.cat((generated, token), dim=1)
            if token_index == 0:
                if input_ids.is_cuda:
                    torch.cuda.synchronize(input_ids.device)
                time_to_first_token = time.perf_counter() - start
            if stream_callback is not None:
                stream_callback(token)
            if eos_token_id is not None and token.item() == eos_token_id:
                break
            if config.cuda_graph_decode:
                if config.mode == "mtp":
                    raise ValueError("CUDA Graph decode does not support MTP mode")
                if decode_graph is None:
                    from .inference_graph import CUDAGraphedDecode

                    if input_ids.is_cuda:
                        torch.cuda.synchronize(input_ids.device)
                    graph_setup_started = time.perf_counter()
                    decode_graph = CUDAGraphedDecode(
                        model, main.cache, max_cache_length
                    )
                    if input_ids.is_cuda:
                        torch.cuda.synchronize(input_ids.device)
                    decode_graph_setup_seconds = (
                        time.perf_counter() - graph_setup_started
                    )
                position = main.cache[0].length
                main = decode_graph.replay(token, position)
            else:
                main = model.forward_inference(
                    token,
                    cache=main.cache,
                    max_cache_length=max_cache_length,
                )
            if config.mode == "mtp":
                draft_output = model.mtp_draft(
                    previous_hidden,
                    token,
                    cache=mtp_cache,
                    max_cache_length=max_cache_length,
                )
                mtp_cache = draft_output.cache
                draft_logits = draft_output.logits[:, -1]

    if input_ids.is_cuda:
        torch.cuda.synchronize(input_ids.device)
    elapsed = time.perf_counter() - start
    new_tokens = generated.size(1) - input_ids.size(1)
    decode_seconds = max(0.0, elapsed - time_to_first_token)
    decoded_after_first = max(0, new_tokens - 1)
    return GenerationResult(
        token_ids=generated,
        new_tokens=new_tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=new_tokens / max(elapsed, 1e-9),
        prefill_seconds=prefill_seconds,
        time_to_first_token_seconds=time_to_first_token,
        decode_seconds=decode_seconds,
        decode_tokens_per_second=(
            decoded_after_first / max(decode_seconds, 1e-9)
            if decoded_after_first
            else 0.0
        ),
        decode_graph_setup_seconds=(
            decode_graph_setup_seconds if config.mode != "no_cache" else 0.0
        ),
        mtp_proposed=proposed,
        mtp_accepted=accepted_count,
    )
