#!/usr/bin/env python3
"""
Gold vs recomputed comparison.

For the first N questions in a teacher_traces.json file, this:
  1. Calls evaluate_entropy_splice.main() IN-PROCESS (one model load) with
     --retention_rate 1.0 --trace_offset 0 --max_traces 1 — i.e. reconstruct
     the FULL thinking region (rate=1.0 means reached_only == full_sequence
     content-wise) for the first usable (non-pathological) trace of each
     question.
  2. Reads back the JSON file evaluate_entropy_splice.main() writes.
  3. For each question, looks up the GOLD correctness (from the original
     teacher_traces.json: trace_correct field if present, else recomputed
     via check_correct on extracted_answers) for the EXACT SAME trace index
     that evaluate_entropy_splice actually used (it may have skipped
     trace 0 if pathological — trace_index in its output tells you which
     one it landed on).
  4. Prints everything: GT, gold extracted answer + correctness, and the
     full generated text + correctness for full_sequence / reached_only /
     random_only / no_cot from the recomputed run.

CAVEAT: gold used sampled (temperature>0) continuation to EOS in one shot
at generation time; evaluate_entropy_splice.py always re-generates the
post-thinking continuation GREEDILY from scratch. Expect gold and
recomputed accuracy to be correlated, not identical — a large mismatch
(e.g. gold 70% vs recomputed 10%) signals a real bug, a modest gap is
normal decoding-method noise.

Usage:
    python gold_check.py \
        --model openai/gpt-oss-20b \
        --traces_file data/aime2025/gpt-oss-20b_teacher_traces.json \
        --n_questions 10 \
        --max_new_tokens 400
"""

import json
import sys
from pathlib import Path
from typing import Optional

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


def _gold_for_trace(q: dict, trace_idx: int, domain: str) -> dict:
    gt = str(q.get("GT_answer", ""))
    extracted_answers = q.get("extracted_answers", [])
    trace_correct_field = q.get("trace_correct")

    extracted = str(extracted_answers[trace_idx]) if trace_idx < len(extracted_answers) else ""

    if trace_correct_field is not None and trace_idx < len(trace_correct_field):
        correct = bool(trace_correct_field[trace_idx])
        source = "field"
    else:
        correct = check_correct(extracted, gt, already_extracted=True, answer_domain=domain)
        source = "recomputed"

    return {"gt": gt, "extracted": extracted, "correct": correct, "source": source}


def main(
    model: str,
    traces_file: str,
    n_questions: int = 10,
    selector: str = "low_entropy",
    max_new_tokens: int = 400,
    device: str = "cuda",
    show_text: bool = True,
    text_chars: int = 300,
):
    path = Path(traces_file)
    domain = _DEFAULT_DOMAIN_BY_DATASET.get(path.parent.name, "math")

    with open(path) as f:
        raw_data = json.load(f)
    subset = raw_data[:n_questions]

    print(f"Running evaluate_entropy_splice at retention_rate=1.0 on first {n_questions} questions "
          f"(one usable trace each, model loaded once)...\n")

    ees.main(
        model=model,
        traces_file=traces_file,
        device=device,
        max_new_tokens=max_new_tokens,
        retention_rate=1.0,
        selector=selector,
        question_offset=0,
        max_questions=n_questions,
        trace_offset=0,
        max_traces=1,
        skip_patched=True,
    )

    eval_output_file = path.with_name(path.stem + f"_eval_{selector}_r1.0_nopatch.json")
    if not eval_output_file.exists():
        print(f"ERROR: expected output file not found: {eval_output_file}")
        sys.exit(1)

    with open(eval_output_file) as f:
        eval_data = json.load(f)

    results_by_qid = {r["question_id"]: r for r in eval_data["results"]}

    print(f"\n{'=' * 100}")
    print("GOLD vs RECOMPUTED (retention_rate=1.0) COMPARISON")
    print(f"{'=' * 100}\n")

    n_gold_correct = 0
    n_full_correct = 0
    n_reached_correct = 0
    n_compared = 0

    for q_idx, q in enumerate(subset):
        r = results_by_qid.get(q_idx)
        if r is None:
            print(f"Q{q_idx}: no usable trace found by evaluate_entropy_splice, skipping\n")
            continue

        trace = r["trace"]
        trace_idx = trace["trace_index"]
        gold = _gold_for_trace(q, trace_idx, domain)

        n_compared += 1
        if gold["correct"]:
            n_gold_correct += 1
        if trace["full_sequence"]["correct"]:
            n_full_correct += 1
        if trace["reached_only"]["correct"]:
            n_reached_correct += 1

        print(f"--- Q{q_idx} (trace_index={trace_idx}, GT={gold['gt']!r}) ---")
        print(f"  GOLD       : extracted={gold['extracted']!r:20s} correct={gold['correct']!s:5s} "
              f"(source={gold['source']})")
        print(f"  full_seq   : correct={trace['full_sequence']['correct']!s:5s}"
              + (f"  text={trace['full_sequence']['generated_answer'][-text_chars:]!r}" if show_text else ""))
        print(f"  reached    : correct={trace['reached_only']['correct']!s:5s}"
              + (f"  text={trace['reached_only']['generated_answer'][-text_chars:]!r}" if show_text else ""))
        print(f"  random     : correct={trace['random_only']['correct']!s:5s}"
              + (f"  text={trace['random_only']['generated_answer'][-text_chars:]!r}" if show_text else ""))
        print(f"  no_cot     : correct={trace['no_cot']['correct']!s:5s}"
              + (f"  text={trace['no_cot']['generated_answer'][-text_chars:]!r}" if show_text else ""))
        print()

    print(f"{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    if n_compared:
        print(f"n_compared:         {n_compared}")
        print(f"gold accuracy:      {n_gold_correct}/{n_compared} = {n_gold_correct/n_compared:.1%}")
        print(f"full_seq accuracy:  {n_full_correct}/{n_compared} = {n_full_correct/n_compared:.1%}")
        print(f"reached accuracy:   {n_reached_correct}/{n_compared} = {n_reached_correct/n_compared:.1%}  "
              f"(should be close to full_seq at rate=1.0)")
        gap = abs(n_gold_correct - n_full_correct) / n_compared
        if gap > 0.3:
            print(f"\nWARNING: gold vs full_seq gap is {gap:.1%} — large enough to suggest a real bug, "
                  f"not just greedy-vs-sampled decoding noise. Investigate individual mismatches above.")
    else:
        print("No comparable traces found.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)