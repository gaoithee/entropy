"""Reasoning trace loader.

Loads pre-generated vLLM traces from HuggingFace datasets.
Ported from neurohike.core.get_reasoning_traces.

Each element in the returned list has:
    input_text      : str
    prompt_tokens   : list[int]
    GT_answer       : str
    traces          : list[str]          — raw trace texts
    traces_tokens   : list[list[int]]    — token IDs per trace
    traces_entropy  : list[list[float]]  — vLLM per-token entropy per trace
"""
from __future__ import annotations
from typing import Any

from datasets import load_dataset


# HuggingFace repo → dataset name mapping for pre-generated traces
_TRACE_REPOS: dict[str, dict[str, str]] = {
    "openai/gpt-oss-20b": {
        "opencompass/AIME2024": "DanielSc4/neurohike-traces",   # AIME_2024 subdir
        "opencompass/AIME2025": "gsarti/aime25_trace_activations_entropy",
        "WildEval/ZebraLogic": "DanielSc4/neurohike-traces",
        "TIGER-Lab/MMLU-Pro":  "DanielSc4/neurohike-traces",
    },
    # Other models: add HF dataset repo when traces are uploaded
}


def get_reasoning_traces(model_name: str, data_name: str) -> list[dict[str, Any]]:
    """Load reasoning traces for (model_name, data_name) from HuggingFace.

    Falls back to a local path convention if no HF repo is registered:
        ~/traces/{model_name}/{data_name}/traces.jsonl

    Returns list of question dicts (see module docstring for schema).
    """
    repo = _TRACE_REPOS.get(model_name, {}).get(data_name)
    if repo:
        ds = load_dataset(repo, split="train", trust_remote_code=True)
        return list(ds)

    # Local fallback
    import json, os
    local = os.path.expanduser(
        f"~/traces/{model_name.replace('/', '_')}/{data_name.replace('/', '_')}/traces.jsonl"
    )
    if not os.path.exists(local):
        raise FileNotFoundError(
            f"No HF repo registered for ({model_name}, {data_name}) "
            f"and local fallback not found: {local}"
        )
    with open(local) as f:
        return [json.loads(line) for line in f]
