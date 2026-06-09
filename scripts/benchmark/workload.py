#!/usr/bin/env python3
"""Workload generators for LLM benchmarking."""

from typing import Optional
from transformers import AutoTokenizer


_CONVERSATION = (
    "You are a helpful assistant. Answer the following questions concisely.\n\n"
    "Q: What is machine learning?\n"
    "A: Machine learning is a subset of artificial intelligence that enables "
    "systems to learn and improve from experience without being explicitly programmed. "
    "It uses algorithms to identify patterns in data and make predictions.\n\n"
    "Q: Explain transformers.\n"
    "A: Transformers are a neural network architecture that uses self-attention "
    "mechanisms to process sequential data. They process all input tokens in parallel, "
    "making them more efficient than recurrent networks. The architecture consists of "
    "encoder and decoder layers with multi-head attention and feed-forward networks.\n\n"
    "Q: What is GPU acceleration?\n"
    "A:"
)


_SINGLE_TOKEN_MARKERS = tuple(f" {c}" for c in "abcdefghijklmnopqrstuvwxyz")


def generate_prompt_exact(target_tokens: int, tokenizer: AutoTokenizer, variant: int = 0) -> str:
    """Generate a prompt that encodes to exactly `target_tokens` tokens."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    # Qwen tokenizes " a" as one stable token.  Keeping the prompt plain ASCII
    # also avoids server-side tokenizer expansion that can prematurely hit the
    # HTTP max_context_len during long-prompt benchmark cases.
    filler = _SINGLE_TOKEN_MARKERS[0]
    markers = ""
    remaining_tokens = target_tokens
    if target_tokens >= 2:
        hi = (variant // len(_SINGLE_TOKEN_MARKERS)) % len(_SINGLE_TOKEN_MARKERS)
        lo = variant % len(_SINGLE_TOKEN_MARKERS)
        markers = _SINGLE_TOKEN_MARKERS[hi] + _SINGLE_TOKEN_MARKERS[lo]
        remaining_tokens -= 2
    repeated = markers + filler * remaining_tokens
    ids = tokenizer.encode(repeated, add_special_tokens=False)
    if len(ids) < target_tokens:
        raise RuntimeError(f"prompt filler produced {len(ids)} tokens, expected at least {target_tokens}")
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def generate_prompt(target_tokens: int, tokenizer: Optional[AutoTokenizer] = None,
                    variant: int = 0) -> str:
    """Generate a prompt with exact token count when a tokenizer is available."""
    if tokenizer is None:
        return _CONVERSATION
    return generate_prompt_exact(target_tokens, tokenizer, variant=variant)


def full_workload_matrix() -> list[dict]:
    return [
        {"prompt_len": prompt_len, "output_len": output_len, "concurrency": concurrency}
        for prompt_len in (32, 128, 512)
        for output_len in (32, 128, 256)
        for concurrency in (1, 2, 4, 8, 16, 32)
    ]


def quick_workload_matrix() -> list[dict]:
    return [{"prompt_len": 32, "output_len": 16, "concurrency": 1}]


def get_tokenizer(model_path: str):
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
