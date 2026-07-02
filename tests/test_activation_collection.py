"""Tests for activation and entropy collection via collect_activations.py (HF).

Requires GPU, pre-generated traces (from collect_traces.py), and .venv environment.

Run with:
    .venv/bin/python -m pytest tests/test_activation_collection.py -v -s
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

MODELS = [
    # "openai/gpt-oss-20b",       # ~14GB, needs A100 80GB for long sequences
    # "openai/gpt-oss-120b",
    "Qwen/Qwen3-4B",              # ~8GB
    # "Qwen/Qwen3-14B",           # ~28GB
    # "google/gemma-4-26B-A4B-it",
    "google/gemma-4-E4B-it",      # ~16GB, limit ~13k tokens on A100 40GB
    # "microsoft/Phi-4-reasoning-plus",
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


def _traces_json_path(model_name: str, data_name: str) -> Path:
    dataset_short = data_name.split("/")[-1]
    model_short = model_name.split("/")[-1]
    return ROOT / "data" / dataset_short / f"{model_short}_teacher_traces.json"


class TestActivationCollection:
    """Test collect_activations.py for each model × dataset combination.

    Requires pre-generated traces from collect_traces.py.
    Skips if the traces JSON doesn't exist yet.
    """

    @pytest.mark.parametrize("model_name", MODELS)
    @pytest.mark.parametrize("data_name", DATASET_NAMES)
    def test_activations_collected(self, model_name, data_name):
        json_path = _traces_json_path(model_name, data_name)
        if not json_path.exists():
            pytest.skip(f"Traces not found: {json_path}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [sys.executable, "scripts/collect_activations.py",
                 "--model_name", model_name,
                 "--data_name", data_name,
                 "--output_dir", tmp_dir,
                 "--max_traces", "1",
                 "--max_seq_length", "8000",
                 "--max_questions", "1",
                ],
                capture_output=True, text=True, cwd=str(ROOT),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            )
            print(result.stdout[-2000:] if result.stdout else "")
            print(result.stderr[-2000:] if result.stderr else "")
            assert result.returncode == 0, \
                f"collect_activations.py failed:\n{result.stderr[-1000:]}"

            # Find output .pth files
            pth_files = sorted(Path(tmp_dir).glob("question_*.pth"))
            assert len(pth_files) >= 1, f"No .pth files produced in {tmp_dir}"

            # Validate first .pth
            data = torch.load(pth_files[0], weights_only=False)

            # Top-level fields
            assert "input_text" in data
            assert "prompt_tokens" in data
            assert "GT_answer" in data
            assert "traces" in data
            assert "num_layers" in data
            assert "model_num_layers" in data
            assert "layers_collected" in data
            assert len(data["traces"]) >= 1

            # Trace-level fields
            trace = data["traces"][0]
            assert "text" in trace
            assert "tokens" in trace
            assert "activations" in trace
            assert "entropies_hf" in trace
            assert "entropies_vllm" in trace
            assert "answer" in trace

            # Shape consistency
            acts = trace["activations"]
            n_tokens = len(trace["tokens"])
            n_layers = data["num_layers"]

            assert acts.ndim == 3, f"Expected 3D tensor, got {acts.ndim}D"
            assert acts.shape[0] == n_tokens, \
                f"activations dim 0 ({acts.shape[0]}) != n_tokens ({n_tokens})"
            assert acts.shape[1] == n_layers, \
                f"activations dim 1 ({acts.shape[1]}) != n_layers ({n_layers})"
            assert len(trace["entropies_hf"]) == n_tokens, \
                f"entropies_hf len ({len(trace['entropies_hf'])}) != n_tokens ({n_tokens})"
            assert len(trace["entropies_vllm"]) == n_tokens, \
                f"entropies_vllm len ({len(trace['entropies_vllm'])}) != n_tokens ({n_tokens})"

            # Sanity checks on values
            assert torch.isfinite(acts).all(), "activations contain NaN/Inf"
            assert acts.abs().sum() > 0, "activations are all zero"

            print(f"\n[{model_name}] [{data_name}] ✓ "
                  f"shape={list(acts.shape)} "
                  f"answer={trace['answer']!r}")