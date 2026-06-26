"""Dataset registry mapping dataset names to HuggingFace identifiers and configs.

Covers all datasets tracked in the entropy project spreadsheet.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    hf_name: str          # HuggingFace dataset identifier
    hf_split: str = "test"
    hf_config: str | None = None
    answer_field: str = "answer"
    question_field: str = "problem"


DATASETS: dict[str, DatasetConfig] = {
    "opencompass/AIME2024": DatasetConfig(
        hf_name="opencompass/AIME2024",
    ),
    "opencompass/AIME2025": DatasetConfig(
        hf_name="opencompass/AIME2025",
    ),
    "opencompass/AIME2026": DatasetConfig(
        hf_name="opencompass/AIME2026",
    ),
    "WildEval/ZebraLogic": DatasetConfig(
        hf_name="WildEval/ZebraLogic",
        answer_field="solution",
        question_field="puzzle",
    ),
    "lighteval/MATH-500": DatasetConfig(
        hf_name="lighteval/MATH-500",
        answer_field="answer",
        question_field="problem",
    ),
    "TIGER-Lab/MMLU-Pro": DatasetConfig(
        hf_name="TIGER-Lab/MMLU-Pro",
        hf_split="test",
        answer_field="answer",
        question_field="question",
    ),
    "Idavidrein/gpqa": DatasetConfig(
        hf_name="Idavidrein/gpqa",
        hf_config="gpqa_diamond",
        answer_field="Correct Answer",
        question_field="Question",
    ),
    "openai/gsm8k": DatasetConfig(
        hf_name="openai/gsm8k",
        hf_config="main",
        answer_field="answer",
        question_field="question",
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")
    return DATASETS[name]
