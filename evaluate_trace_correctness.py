#!/usr/bin/env python3
"""
Re-evaluate and categorize every trace in every teacher_traces.json file.

For each trace, assigns exactly one category:
    truncated  — generation hit the token budget without finishing
                 (len(trace_tokens) >= max_tokens - truncation_margin);
                 we can't know if it would've been correct, it just never
                 got the chance to produce a boxed answer.
    no_boxed   — generation finished (not truncated) but extract_boxed_answer
                 found nothing (extracted_answers[i] is empty).
    correct    — finished, has a non-empty extracted answer, and
                 entropy.core.utils.check_correct (math-verify) says it
                 matches GT_answer.
    incorrect  — finished, has a non-empty extracted answer, but it does
                 NOT match GT_answer.

Also (optionally, via --write True) rewrites trace_correct = [bool, ...] in
each file — True only for the "correct" category, False for all others
(incorrect / truncated / no_boxed) — consistent with check_correct's own
"empty extraction => False" behavior. A timestamped backup is made first.

By default this is a DRY RUN: it reports the breakdown but writes nothing.

Usage:
    # Report only, no files touched
    python reevaluate_trace_correct.py --data_glob "data/*/*teacher_traces.json"

    # Also rewrite trace_correct (with backups)
    python reevaluate_trace_correct.py --data_glob "data/*/*teacher_traces.json" --write True

    # If some datasets were generated with a different token budget
    python reevaluate_trace_correct.py --max_tokens 16384 --truncation_margin 8
"""

import glob
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from entropy.core.utils import check_correct
except ImportError as exc:
    print(
        "ERROR: could not import entropy.core.utils.check_correct.\n"
        "Run this script from the repo root (where the `entropy` package is "
        "importable), or adjust PYTHONPATH.\n"
        f"Original error: {exc}"
    )
    sys.exit(1)


_DEFAULT_DOMAIN_BY_DATASET = {
    # dataset dir name (Path(f).parent.name) -> answer_domain
    "non-math-mmlu-pro": "mcq",
    "gpqa": "mcq",
}


def _resolve_domain(path: Path, cli_default: str) -> str:
    return _DEFAULT_DOMAIN_BY_DATASET.get(path.parent.name, cli_default)


def _backup_path(path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_suffix(path.suffix + f".bak_{ts}")


def categorize_trace(
    trace_tokens: list,
    extracted: str,
    gt: str,
    answer_domain: str,
    max_tokens: int,
    truncation_margin: int,
) -> str:
    """Return one of: 'truncated', 'no_boxed', 'correct', 'incorrect'."""
    n_tokens = len(trace_tokens) if trace_tokens is not None else 0
    if n_tokens >= max_tokens - truncation_margin:
        return "truncated"

    extracted_str = str(extracted).strip()
    if not extracted_str:
        return "no_boxed"

    if check_correct(extracted_str, gt, already_extracted=True, answer_domain=answer_domain):
        return "correct"
    return "incorrect"


def reevaluate_file(
    path: Path,
    answer_domain: str,
    max_tokens: int,
    truncation_margin: int,
    write: bool,
) -> dict:
    with open(path) as f:
        data = json.load(f)

    n_questions = len(data)
    counts = {"correct": 0, "incorrect": 0, "truncated": 0, "no_boxed": 0}
    n_missing_fields = 0

    for q in data:
        gt = str(q.get("GT_answer", ""))
        extracted_answers = q.get("extracted_answers")
        traces_tokens = q.get("traces_tokens")

        if extracted_answers is None or traces_tokens is None:
            n_missing_fields += 1
            continue

        new_tc = []
        for i, ans in enumerate(extracted_answers):
            tt = traces_tokens[i] if i < len(traces_tokens) else []
            category = categorize_trace(
                tt, ans, gt, answer_domain, max_tokens, truncation_margin
            )
            counts[category] += 1
            new_tc.append(category == "correct")

        q["trace_correct"] = new_tc

    total = sum(counts.values())
    summary = {
        "file": str(path),
        "n_questions": n_questions,
        "n_missing_fields": n_missing_fields,
        "total": total,
        **counts,
    }

    if write:
        backup = _backup_path(path)
        shutil.copy2(path, backup)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        summary["backup"] = str(backup)
        summary["written"] = True
    else:
        summary["written"] = False

    return summary


def main(
    data_glob: str = "data/*/*teacher_traces.json",
    answer_domain: str = "math",
    max_tokens: int = 16384,
    truncation_margin: int = 8,
    write: bool = False,
):
    """
    answer_domain: default domain for datasets not listed in
        _DEFAULT_DOMAIN_BY_DATASET (edit that dict at the top of this file
        to add/override per-dataset domains).
    max_tokens / truncation_margin: a trace is classified 'truncated' if
        len(trace_tokens) >= max_tokens - truncation_margin. Adjust
        max_tokens if some files were generated with a different budget
        than the 16384 default used elsewhere in this repo.
    """
    paths = sorted(Path(p) for p in glob.glob(data_glob))
    if not paths:
        print(f"No files matched glob: {data_glob!r}")
        return

    print(f"Found {len(paths)} file(s) matching {data_glob!r}")
    print(f"answer_domain(default)={answer_domain!r}  max_tokens={max_tokens}  "
          f"truncation_margin={truncation_margin}  write={write}")
    if not write:
        print("*** DRY RUN — no files will be modified. Pass --write True to apply. ***")
    print()

    header = (
        f"{'file':55s} {'domain':>6s} {'quest':>6s} {'total':>6s} "
        f"{'correct':>8s} {'incorrect':>9s} {'truncated':>9s} {'no_boxed':>8s} {'miss':>5s}"
    )
    print(header)
    print("-" * len(header))

    summaries = []
    for path in paths:
        domain = _resolve_domain(path, answer_domain)
        try:
            s = reevaluate_file(path, domain, max_tokens, truncation_margin, write)
        except Exception as exc:
            print(f"{str(path):55s}  ERROR: {type(exc).__name__}: {exc}")
            continue
        summaries.append((s, domain))

        total = s["total"]

        def _pct(k):
            return f"{s[k] / total:.1%}" if total else "  n/a"

        print(
            f"{str(path):55s} {domain:>6s} {s['n_questions']:>6d} {total:>6d} "
            f"{_pct('correct'):>8s} {_pct('incorrect'):>9s} "
            f"{_pct('truncated'):>9s} {_pct('no_boxed'):>8s} "
            f"{s['n_missing_fields']:>5d}"
        )

    print()
    if write:
        n_written = sum(1 for s, _ in summaries if s["written"])
        print(f"Wrote trace_correct to {n_written}/{len(summaries)} file(s). Backups created alongside each.")
    else:
        print("Dry run complete. Re-run with --write True to persist trace_correct.")

    n_missing_total = sum(s["n_missing_fields"] for s, _ in summaries)
    if n_missing_total:
        print(
            f"\nNOTE: {n_missing_total} question(s) across all files were missing "
            "'extracted_answers' or 'traces_tokens' and were left untouched."
        )


if __name__ == "__main__":
    import fire
    fire.Fire(main)