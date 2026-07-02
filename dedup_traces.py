#!/usr/bin/env python3
"""
Targeted fix for files where prefix-based dedup (dedup_traces.py) misses
some duplicate pairs because their input_text differs even in the first N
characters (e.g. whitespace/encoding drift between pre-fix and post-fix
generations).

This does a two-signal dedup:
  1. Primary key: input_text prefix (first `key_prefix_len` chars) — same
     as dedup_traces.py.
  2. For entries that DON'T share a prefix key with anything else, do a
     secondary pass grouping by GT_answer alone, but only merge two
     prefix-groups under the same GT_answer if their prefixes are also
     "close enough" (share a long common prefix, configurable via
     --min_common_prefix) — this guards against GT_answer collisions
     between genuinely different questions (a real risk on MCQ datasets
     like gpqa/mmlu-pro/zebralogic, where GT_answer is just a letter).

Within any merged group, keeps the entry with the most non-empty
extracted_answers (ties broken by keeping the last one).

Usage:
    # Dry run
    python fix_cross_dedup.py --path data/aime_2024/gemma-4-E4B-it_teacher_traces.json

    # Apply (backup created first)
    python fix_cross_dedup.py --path data/aime_2024/gemma-4-E4B-it_teacher_traces.json --write True

    # Stricter/looser common-prefix requirement for the GT_answer pass
    python fix_cross_dedup.py --path ... --min_common_prefix 60
"""

import glob
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _backup_path(path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return path.with_suffix(path.suffix + f".crossdedup_bak_{ts}")


def _n_nonempty(q: dict) -> int:
    return sum(1 for a in q.get("extracted_answers", []) if str(a).strip())


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def process_file(
    p: Path,
    key_prefix_len: int,
    min_common_prefix: int,
    write: bool,
    verbose: bool,
) -> dict:
    with open(p) as f:
        data = json.load(f)

    n_before = len(data)

    # --- Pass 1: group by prefix key ---
    prefix_groups: dict[str, list[dict]] = defaultdict(list)
    prefix_order: list[str] = []
    for q in data:
        key = q.get("input_text", "")[:key_prefix_len]
        if key not in prefix_groups:
            prefix_order.append(key)
        prefix_groups[key].append(q)

    confident_groups = [prefix_groups[k] for k in prefix_order if len(prefix_groups[k]) > 1]
    orphans = [(k, prefix_groups[k][0]) for k in prefix_order if len(prefix_groups[k]) == 1]

    # --- Pass 2: among orphans, group by GT_answer + common-prefix guard ---
    by_gt: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, q in orphans:
        gt = str(q.get("GT_answer", ""))
        by_gt[gt].append((key, q))

    orphan_final_groups: list[list[dict]] = []
    used = set()
    for gt, entries in by_gt.items():
        if len(entries) == 1:
            key, q = entries[0]
            if key not in used:
                orphan_final_groups.append([q])
                used.add(key)
            continue

        remaining = list(entries)
        while remaining:
            base_key, base_q = remaining.pop(0)
            if base_key in used:
                continue
            group = [base_q]
            used.add(base_key)
            still_remaining = []
            for key, q in remaining:
                if key in used:
                    continue
                if _common_prefix_len(base_key, key) >= min_common_prefix:
                    group.append(q)
                    used.add(key)
                else:
                    still_remaining.append((key, q))
            remaining = still_remaining
            orphan_final_groups.append(group)

    all_groups = confident_groups + orphan_final_groups

    if verbose and (len(confident_groups) or any(len(g) > 1 for g in orphan_final_groups)):
        print(f"  Pass 1: {len(confident_groups)} confident group(s), {len(orphans)} orphan(s)")
        print(f"  Pass 2: {sum(1 for g in orphan_final_groups if len(g) > 1)} group(s) merged from orphans")

    kept: list[dict] = []
    n_discarded = 0
    merge_details = []
    for group in all_groups:
        if len(group) == 1:
            kept.append(group[0])
            continue
        gts = [str(q.get("GT_answer", "")) for q in group]
        nonempty_counts = [_n_nonempty(q) for q in group]
        best_idx = max(range(len(group)), key=lambda i: (nonempty_counts[i], i))
        merge_details.append((len(group), gts, nonempty_counts, best_idx))
        kept.append(group[best_idx])
        n_discarded += len(group) - 1

    n_after = len(kept)

    if write and n_discarded > 0:
        backup = _backup_path(p)
        shutil.copy2(p, backup)
        with open(p, "w") as f:
            json.dump(kept, f, indent=2, ensure_ascii=False)

    return {
        "file": str(p),
        "n_before": n_before,
        "n_after": n_after,
        "n_discarded": n_discarded,
        "merge_details": merge_details,
        "written": write and n_discarded > 0,
    }


def main(
    path: Optional[str] = None,
    data_glob: Optional[str] = None,
    key_prefix_len: int = 150,
    min_common_prefix: int = 80,
    write: bool = False,
    verbose: bool = False,
):
    if path is None and data_glob is None:
        print("Provide either --path (single file) or --data_glob (multiple files).")
        return

    paths = [Path(path)] if path else sorted(Path(pp) for pp in glob.glob(data_glob))
    if not paths:
        print("No files matched.")
        return

    if not write:
        print("*** DRY RUN — no files will be modified. Pass --write True to apply. ***\n")

    header = f"{'file':60s} {'before':>8s} {'after':>8s} {'discarded':>10s}"
    print(header)
    print("-" * len(header))

    any_issue = False
    for p in paths:
        try:
            s = process_file(p, key_prefix_len, min_common_prefix, write, verbose)
        except Exception as e:
            print(f"{str(p):60s}  ERROR: {type(e).__name__}: {e}")
            continue

        marker = ""
        if s["n_discarded"] > 0:
            marker = "  <-- had duplicates"
            any_issue = True
        print(f"{s['file']:60s} {s['n_before']:>8d} {s['n_after']:>8d} {s['n_discarded']:>10d}{marker}")

        for group_size, gts, nonempty_counts, best_idx in s["merge_details"]:
            if len(set(gts)) > 1:
                print(f"    WARNING: merged group has MIXED GT_answers {gts} — check --min_common_prefix")

    print()
    if not any_issue:
        print("No duplicates found in any file.")
    elif write:
        print("Duplicates removed where found. Backups created alongside modified files.")
    else:
        print("Duplicates found in some files. Re-run with --write True to remove them.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)