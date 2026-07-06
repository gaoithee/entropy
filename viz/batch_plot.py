#!/usr/bin/env python3
"""
Batch-generate compression-curve plots for every (model, dataset) combo
found under a results directory, reusing plot_compression_curves.py.

Scans {data_root}/{dataset}/results/*.json (or {data_root}/{dataset}/*.json,
same dual-layout support as plot_compression_curves.py) to discover which
model/dataset pairs actually have data, instead of requiring a hardcoded
list -- so it stays correct as more sweep results get uploaded/downloaded
over time without needing to edit this script.

USAGE:
    python batch_plot.py \
        --data_root ../entropy-traces-local \
        --output_root ../figures \
        --x_axis nominal
"""

import glob
import re
from pathlib import Path
from typing import Set, Tuple

from plot_compression_curves import load_trace_points, plot_curves, summarize_pool_exhaustion

# Matches "{model_short}_teacher_traces_eval_{selector}_r{rate}.json" and
# captures model_short. model_short itself may contain underscores/dashes
# (e.g. "gemma-4-26B-A4B-it"), so anchor on the fixed "_teacher_traces_eval_"
# marker instead of trying to split on the first underscore.
FILENAME_RE = re.compile(r"^(.+?)_teacher_traces_eval_.+\.json$")


def discover_pairs(data_root: str) -> Set[Tuple[str, str]]:
    """Return {(dataset, model_short), ...} for every result file found,
    under either the flat or the results/ subfolder layout.
    """
    pairs: Set[Tuple[str, str]] = set()
    root = Path(data_root)

    candidates = list(root.glob("*/*_teacher_traces_eval_*.json")) + \
                 list(root.glob("*/results/*_teacher_traces_eval_*.json"))

    for fpath in candidates:
        dataset = fpath.parent.name if fpath.parent.name != "results" else fpath.parent.parent.name
        m = FILENAME_RE.match(fpath.name)
        if not m:
            continue
        model_short = m.group(1)
        pairs.add((dataset, model_short))

    return pairs


def main(
    data_root: str = "entropy-traces-local",
    output_root: str = "figures",
    x_axis: str = "nominal",
    n_bins: int = 10,
    min_selectors: int = 1,
):
    """
    min_selectors: skip a (dataset, model) pair if fewer than this many
        selectors have any data at all -- avoids generating near-empty,
        single-point plots for combos where the sweep has barely started.
        Default 1 (generate everything that has at least some data); raise
        it (e.g. to 3) once you only want reasonably-populated figures.
    """
    pairs = discover_pairs(data_root)
    if not pairs:
        print(f"No result files found under {data_root!r} (checked both "
              f"flat and results/ subfolder layouts).")
        return

    print(f"Found {len(pairs)} (dataset, model) pairs with data.")

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_ok, n_skipped = 0, 0

    for dataset, model_short in sorted(pairs):
        out_dir = out_root / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_short}.png"

        try:
            # load_trace_points expects a full model id to split on "/", but
            # here we only have model_short (parsed back out of filenames,
            # since the sweep results don't store the org prefix) -- passing
            # model_short directly still works because .split("/")[-1] on a
            # string with no "/" just returns the string unchanged.
            points_by_selector = load_trace_points(model_short, dataset, data_root)
        except FileNotFoundError:
            print(f"  [skip] {dataset}/{model_short}: no files matched after all")
            n_skipped += 1
            continue

        n_selectors_with_data = sum(1 for pts in points_by_selector.values() if pts)
        if n_selectors_with_data < min_selectors:
            print(f"  [skip] {dataset}/{model_short}: only {n_selectors_with_data} "
                  f"selector(s) with data (< min_selectors={min_selectors})")
            n_skipped += 1
            continue

        print(f"  [plot] {dataset}/{model_short} -> {out_path}")
        summarize_pool_exhaustion(points_by_selector)
        plot_curves(
            points_by_selector,
            x_axis=x_axis,
            n_bins=n_bins,
            title=f"{model_short} - {dataset.upper()}",
            output_path=str(out_path),
        )
        n_ok += 1

    print(f"\nDone: {n_ok} plots written, {n_skipped} pairs skipped.")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
