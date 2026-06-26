"""Dataset loading and answer extraction utilities.

Ported from neurohike/shared/data_utils.py.
Kept: get_data, get_answer_suffix, get_answer_domain, extract_boxed_answer.
Dropped: code benchmark helpers (evalplus, live_code_bench) — not needed here.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from datasets import load_dataset


def get_data(data_name: str) -> list[tuple[str, int | str]]:
    """Load a reasoning dataset. Returns list of (question, answer) tuples."""

    if "gsm8k" in data_name.casefold():
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        def _extract(a): return int(a.split("#### ")[-1].strip().replace(",", ""))
        return [(item["question"], _extract(item["answer"])) for item in dataset]

    if "aime2025" in data_name.casefold():
        ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        instr = "\nAnswer by placing your final answer in a \\boxed{} environment."
        out = [(item["question"] + instr, int(item["answer"])) for item in ds1]
        out += [(item["question"] + instr, int(item["answer"].replace(r"^\circ", ""))) for item in ds2]
        return out

    if "aime_2024" in data_name.casefold():
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        out = []
        for item in dataset:
            try:
                ans = str(item["answer"]).strip().replace(r"\\boxed{", "").replace("}", "")
                out.append((item["problem"], int(ans)))
            except (ValueError, KeyError):
                continue
        return out

    if "aime_2026" in data_name.casefold():
        dataset = load_dataset("MathArena/aime_2026", split="train")
        instr = "\nAnswer by placing your final answer in a \\boxed{} environment."
        out = []
        for item in dataset:
            try:
                out.append((item["problem"] + instr, int(item["answer"])))
            except (ValueError, KeyError):
                continue
        return out

    if "non-math-mmlu-pro" in data_name.casefold():
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
        filtered = [e for e in dataset if e["category"] != "math"]
        out = []
        instr = "\nAnswer with the letter corresponding to the correct option."
        for item in filtered:
            try:
                opts = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(item["options"]))
                q = item["question"] + "\n" + opts + instr
                out.append((q, item["answer"]))
            except (ValueError, KeyError, IndexError):
                continue
        return out

    if "zebralogic" in data_name.casefold():
        dataset = load_dataset("WildEval/ZebraLogic", "mc_mode", split="test")
        np.random.seed(12)
        indices = np.random.choice(len(dataset), size=50, replace=False)
        dataset = dataset.select(indices)
        out = []
        for item in dataset:
            try:
                instr = "\nAnswer with one of the following options in a \\boxed{} environment:"
                q = item["puzzle"] + "\n" + item["question"] + "\n" + instr + "\n".join(item["choices"])
                out.append((q, item["answer"]))
            except (ValueError, KeyError, IndexError):
                continue
        return out

    if "gpqa" in data_name.casefold():
        dataset = load_dataset("fingertap/GPQA-Diamond", split="test")
        instr = "\nAnswer with the letter corresponding to the correct option in a \\boxed{} environment."
        return [(item["question"] + instr, item["answer"]) for item in dataset]

    if "math-500" in data_name.casefold():
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        instr = "\nAnswer by placing your final answer in a \\boxed{} environment."
        return [(item["problem"] + instr, str(item["answer"])) for item in dataset]

    raise ValueError(f"Unknown dataset: {data_name}")


def get_answer_suffix(data_name: str) -> str:
    """Return the answer-forcing suffix for the dataset."""
    return r"Therefore, the final answer is \boxed{"


def get_answer_domain(data_name: str) -> str:
    """Return answer domain: 'math' or 'mcq'."""
    name = data_name.casefold()
    if any(k in name for k in ("mmlu", "zebralogic", "gpqa")):
        return "mcq"
    return "math"
