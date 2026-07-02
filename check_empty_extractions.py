#!/usr/bin/env python3
"""
Quick, GPU-free check: for every teacher_traces.json file, count how many
extracted_answers entries are empty (vs total), across all questions.

Usage:
    python check_empty_extractions.py
    python check_empty_extractions.py --data_glob "data/*/*teacher_traces.json"
"""

import glob
import json
from pathlib import Path


def check_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    total = 0
    empty = 0
    n_questions_all_empty = 0

    for q in data:
        answers = q.get("extracted_answers", [])
        q_empty = 0
        for a in answers:
            total += 1
            if not str(a).strip():
                empty += 1
                q_empty += 1
        if answers and q_empty == len(answers):
            n_questions_all_empty += 1

    return {
        "n_questions": len(data),
        "n_questions_all_empty": n_questions_all_empty,
        "total": total,
        "empty": empty,
    }


def main(data_glob: str = "data/*/*teacher_traces.json"):
    paths = sorted(Path(p) for p in glob.glob(data_glob))
    if not paths:
        print(f"No files matched: {data_glob!r}")
        return

    header = f"{'file':60s} {'quest':>6s} {'q_all_empty':>12s} {'traces':>7s} {'empty':>7s} {'empty_%':>8s}"
    print(header)
    print("-" * len(header))

    flagged = []
    for path in paths:
        try:
            s = check_file(path)
        except Exception as e:
            print(f"{str(path):60s}  ERROR: {type(e).__name__}: {e}")
            continue

        pct = (s["empty"] / s["total"]) if s["total"] else 0.0
        print(
            f"{str(path):60s} {s['n_questions']:>6d} {s['n_questions_all_empty']:>12d} "
            f"{s['total']:>7d} {s['empty']:>7d} {pct:>7.1%}"
        )
        if pct > 0.05:
            flagged.append((str(path), pct))

    if flagged:
        print(f"\n{len(flagged)} file(s) with >5% empty extractions:")
        for f, pct in flagged:
            print(f"  {f}: {pct:.1%}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
