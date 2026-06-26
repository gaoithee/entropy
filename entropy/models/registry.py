"""Model registry: thinking token delimiters per model family.

Ported from neurohike shared/data_utils.py get_thinking_tokens().
"""
from __future__ import annotations
from typing import Any


def get_thinking_tokens(model_name: str) -> dict[str, Any]:
    """Return start/end thinking token strings and explicit token ID lists.

    When token ID lists are None they are resolved by encoding the string
    tokens via the tokenizer at runtime.

    Returns dict with keys:
        start_token       : str
        end_token         : str
        start_token_ids   : list[int] | None
        end_token_ids     : list[int] | None
    """
    m = model_name.lower()

    if "gpt-oss" in m:
        return {
            "start_token": "<|channel|>analysis<|message|>",
            "end_token": "<|channel|>final<|message|>",
            "start_token_ids": [200005, 35644, 200008],
            "end_token_ids": [200007, 200006, 173781, 200005, 17196, 200008],
        }

    if "deepseek" in m:
        return {
            "start_token": "<think>",
            "end_token": "</think>",
            "start_token_ids": None,
            "end_token_ids": None,
        }

    if "gemma-4" in m or "gemma4" in m:
        return {
            "start_token": "<|channel>thought",
            "end_token": "<channel|>",
            "start_token_ids": None,
            "end_token_ids": None,
        }

    if "ministral" in m or "mistral" in m:
        return {
            "start_token": "[THINK]",
            "end_token": "[/THINK]",
            "start_token_ids": [34],
            "end_token_ids": [35],
        }

    if "qwen" in m:
        return {
            "start_token": "<think>",
            "end_token": "</think>",
            "start_token_ids": None,
            "end_token_ids": None,
        }

    if "phi" in m:
        return {
            "start_token": "<think>",
            "end_token": "</think>",
            "start_token_ids": None,
            "end_token_ids": None,
        }

    # Fallback
    return {
        "start_token": "<think>",
        "end_token": "</think>",
        "start_token_ids": None,
        "end_token_ids": None,
    }