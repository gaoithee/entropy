"""Tests for trace generation via collect_traces.py (vLLM).

Requires GPU, network access, and .venv-traces environment.

Run with:
    .venv-traces/bin/python -m pytest tests/test_trace_collection.py -v -s
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODELS = [
    "openai/gpt-oss-20b",
    # "openai/gpt-oss-120b",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-4B",
    # "google/gemma-4-26B-A4B-it",
    "google/gemma-4-E4B-it",
    "microsoft/Phi-4-reasoning-plus",
]

DATASET_NAMES = [
    "aime2025",
    "aime_2024",
    "aime_2026",
    "zebralogic",
    "math-500",
    "non-math-mmlu-pro",
    "gpqa",
]

ROOT = Path(__file__).parent.parent


class TestTraceCollection:
    """Test collect_traces.py for each model × dataset combination."""

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("data_name", DATASET_NAMES)
    def test_trace_generated(self, model_name, data_name):
        from entropy.core.data_utils import get_data
        question, _ = get_data(data_name)[0]

        result = subprocess.run(
            [sys.executable, "scripts/collect_traces.py",
             "--model_name", model_name,
             "--data_name", data_name,
             "--num_out", "1",
             "--batch_size", "1",
             "--resume", "False",
             "--max_tokens", "128",
             "--num_questions", "1",
            ],
            capture_output=True, text=True, cwd=str(ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        print(result.stdout[-2000:] if result.stdout else "")
        print(result.stderr[-2000:] if result.stderr else "")
        assert result.returncode == 0, f"collect_traces.py failed:\n{result.stderr[-1000:]}"

        dataset_short = data_name.split("/")[-1]
        model_short = model_name.split("/")[-1]
        json_path = ROOT / "data" / dataset_short / f"{model_short}_teacher_traces.json"
        assert json_path.exists(), f"Output not found: {json_path}"

        with open(json_path) as f:
            traces = json.load(f)
        assert len(traces) >= 1

        # Find the record for our question
        record = next((r for r in traces if r["input_text"] == question), traces[0])

        assert len(record["traces"]) >= 1
        assert len(record["traces_tokens"]) == len(record["traces"])
        assert len(record["traces_tokens"][0]) > 0
        assert len(record["prompt_tokens"]) > 0

        print(f"\n[{model_name}] [{data_name}] ✓ "
              f"prompt_len={len(record['prompt_tokens'])} "
              f"trace_len={len(record['traces_tokens'][0])}"
              )
        print(f"  trace: {record['traces'][0]!r}")