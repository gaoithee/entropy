"""Reasoning trace loader.

Loads pre-generated vLLM traces from local files.
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

import json
from pathlib import Path
from typing import Any


def get_reasoning_traces(model_name: str, data_name: str) -> list[dict[str, Any]]:
    """Load reasoning traces for (model_name, data_name) from local files.

    Looks for:
        data/{data_name_short}/{model_name_short}_teacher_traces.json

    Returns list of question dicts (see module docstring for schema).
    """
    data_short = data_name.split("/")[-1]
    model_short = model_name.split("/")[-1]

    local = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / data_short
        / f"{model_short}_teacher_traces.json"
    )

    if not local.exists():
        raise FileNotFoundError(
            f"Traces not found for ({model_name}, {data_name}). "
            f"Expected: {local}"
        )

    with open(local) as f:
        return json.load(f)