#!/usr/bin/env python3
"""
Smoke test driver for evaluate_entropy_splice.py.

For each (model, dataset) pair, finds the first non-pathological trace
(not truncated, has a non-empty extracted answer — via the same logic as
find_good_trace.py) directly from the teacher_traces.json file, then calls
evaluate_entropy_splice.py's main() in-process with
--question_offset/--trace_offset/--max_questions 1/--max_traces 1 pointed
at that trace.

No subprocess spawning: imports evaluate_entropy_splice and calls its
main() directly, so model loading etc. all happens through the same code
path you'd get from the CLI. One model reload per (model, dataset) pair
(unavoidable — each dataset's traces_file needs a fresh main() call with
different traces_file/offsets, and main() itself reloads the model).

Usage:
    # Default: gpt-oss-20b across aime2025 / zebralogic / gpqa
    python smoke_test_entropy_splice.py

    # Custom models / datasets
    python smoke_test_entropy_splice.py \
        --models '["openai/gpt-oss-20b","Qwen/Qwen3-4B"]' \
        --datasets '["aime2025","gpqa"]'

    # Different selector / retention_rate
    python smoke_test_entropy_splice.py --selector numbers --retention_rate 0.2

NOTE: since main() reloads the model for every (model, dataset) pair, this
is NOT efficient for a full sweep — it's meant purely as a correctness
smoke test (one clean trace per combination), not a benchmark driver.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from entropy.core.utils import check_correct
except ImportError as exc:
    print(f"ERROR importing entropy.core.utils.check_correct: {exc}")
    sys.exit(1)

try:
    import evaluate_entropy_splice as ees
except ImportError as exc:
    print(f"ERROR importing evaluate_entropy_splice.py: {exc}")
    print("Run this script from the same directory as evaluate_entropy_splice.py.")
    sys.exit(1)


_DEFAULT_DOMAIN_BY_DATASET = {
    "non-math-mmlu-pro": "mcq",
    "gpqa": "mcq",
    "zebralogic": "mcq",
}

_DEFAULT_MODELS = [
    "openai/gpt-oss-20b",
]

_DEFAULT_DATASETS = [
    "aime2025",
    "zebralogic",
    "gpqa",
]


def _model_short_name(model: str) -> str:
    return model.split("/")[-1]


def find_good_trace(
    traces_file: Path,
    max_tokens: int = 16384,
    truncation_margin: int = 8,
    require_correct: bool = False,
) -> Optional[dict]:
    """Same logic as find_good_trace.py, returned as a dict instead of printed."""
    domain = _DEFAULT_DOMAIN_BY_DATASET.get(traces_file.parent.name, "math")

    with open(traces_file) as f:
        data = json.load(f)

    for q_idx, q in enumerate(data):
        gt = str(q.get("GT_answer", ""))
        traces_tokens = q.get("traces_tokens", [])
        extracted_answers = q.get("extracted_answers", [])

        for t_idx, (tt, ans) in enumerate(zip(traces_tokens, extracted_answers)):
            n_tokens = len(tt) if tt is not None else 0
            if n_tokens >= max_tokens - truncation_margin:
                continue
            ans_str = str(ans).strip()
            if not ans_str:
                continue

            is_correct = check_correct(ans_str, gt, already_extracted=True, answer_domain=domain)
            if require_correct and not is_correct:
                continue

            return {
                "question_offset": q_idx,
                "trace_offset": t_idx,
                "n_tokens": n_tokens,
                "extracted": ans_str,
                "gt": gt,
                "correct": is_correct,
            }

    return None


def main(
    models: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    data_dir: str = "data",
    selector: str = "low_entropy",
    retention_rate: float = 0.1,
    max_new_tokens: int = 20,
    skip_patched: bool = True,
    require_correct: bool = False,
    max_tokens: int = 16384,
    truncation_margin: int = 8,
    device: str = "cuda",
):
    models = models or _DEFAULT_MODELS
    datasets = datasets or _DEFAULT_DATASETS

    results = []

    for dataset in datasets:
        for model in models:
            model_short = _model_short_name(model)
            traces_file = Path(data_dir) / dataset / f"{model_short}_teacher_traces.json"

            print(f"\n{'=' * 70}")
            print(f"=== {model} / {dataset} ===")
            print(f"{'=' * 70}")

            if not traces_file.exists():
                print(f"  SKIP: {traces_file} not found")
                results.append((model, dataset, "skip_missing_file", None))
                continue

            trace = find_good_trace(
                traces_file,
                max_tokens=max_tokens,
                truncation_margin=truncation_margin,
                require_correct=require_correct,
            )
            if trace is None:
                print(f"  SKIP: no non-pathological trace found in {traces_file}")
                results.append((model, dataset, "skip_no_good_trace", None))
                continue

            print(
                f"  Using question={trace['question_offset']} trace={trace['trace_offset']} "
                f"(n_tokens={trace['n_tokens']}, extracted={trace['extracted']!r}, "
                f"gt={trace['gt']!r}, correct={trace['correct']})"
            )

            try:
                ees.main(
                    model=model,
                    traces_file=str(traces_file),
                    device=device,
                    max_new_tokens=max_new_tokens,
                    retention_rate=retention_rate,
                    selector=selector,
                    question_offset=trace["question_offset"],
                    trace_offset=trace["trace_offset"],
                    max_questions=1,
                    max_traces=1,
                    skip_patched=skip_patched,
                )
                results.append((model, dataset, "ok", trace))
            except Exception as exc:
                print(f"  ERROR running evaluate_entropy_splice.main(): {type(exc).__name__}: {exc}")
                traceback.print_exc()
                results.append((model, dataset, f"error: {exc}", trace))

    print(f"\n{'=' * 70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'=' * 70}")
    header = f"{'model':30s} {'dataset':16s} {'status':30s}"
    print(header)
    print("-" * len(header))
    for model, dataset, status, _ in results:
        print(f"{model:30s} {dataset:16s} {status:30s}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
