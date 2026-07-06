#!/usr/bin/env python3
"""
Compare all selectors restricted to the SAME actual-retention band, instead
of comparing them across their full (and very different) ranges. This
answers "at matched compression, is selector X actually worse than the
others, or does it just live in a harder region of the curve by
construction?" -- the question raised for `numbers`, whose pool-exhaustion
confines it to a narrow high-compression band regardless of the requested
--retention_rate (see pool_exh diagnostics in evaluate_entropy_splice.py's
own output and in plot_compression_curves.py's summarize_pool_exhaustion).

By default, the band is auto-detected from the reference selector's own
[p25, p75] actual-retention range (so it's always "the band that selector
naturally lives in", not an arbitrarily chosen number) -- override with
--band_lo/--band_hi to force a specific window instead.

USAGE:
    python compare_at_matched_compression.py \
        --model openai/gpt-oss-20b \
        --dataset zebralogic \
        --data_root ../../entropy-traces-local \
        --reference_selector numbers
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from plot_compression_curves import load_trace_points, bootstrap_relative_pr
from batch_plot import discover_pairs


def load_trace_points_patched(model: str, dataset: str, data_root: str, condition: str = "reached_patched"):
    """Same as load_trace_points but pulls `correct` from a different trace
    condition -- default 'reached_patched' (activations injected at the
    selected positions), not 'reached_only' (raw reconstructed text, no
    activation patching). For this comparison we care about what survives
    causally in the residual stream, not how legible the bare token
    reconstruction happens to read.
    """
    import glob
    import json
    from pathlib import Path as _Path

    model_short = model.split("/")[-1]
    patterns = [
        str(_Path(data_root) / dataset / f"{model_short}_teacher_traces_eval_*.json"),
        str(_Path(data_root) / dataset / "results" / f"{model_short}_teacher_traces_eval_*.json"),
    ]
    files = []
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            break
    if not files:
        raise FileNotFoundError(f"No files matched any of: {patterns!r}")

    points_by_selector: Dict[str, List[dict]] = {}

    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)

        selector = data.get("selector", "unknown")
        results = data.get("results", [])

        for entry in results:
            trace = entry.get("trace", {})
            n_thinking = trace.get("n_tokens_thinking")
            n_reached = trace.get("n_tokens_reached")
            cond_info = trace.get(condition, {})
            full_info = trace.get("full_sequence", {})
            cond_correct = cond_info.get("correct")
            full_correct = full_info.get("correct")

            if n_thinking is None or n_reached is None or cond_correct is None:
                continue
            if n_thinking <= 0:
                continue

            points_by_selector.setdefault(selector, []).append({
                "actual_retention": n_reached / n_thinking,
                "nominal_retention": trace.get("retention_rate"),
                "reached_correct": int(bool(cond_correct)),
                "full_correct": int(bool(full_correct)) if full_correct is not None else None,
            })

    return points_by_selector


def band_from_reference(points: List[dict]) -> tuple:
    actual = np.array([p["actual_retention"] for p in points])
    return float(np.percentile(actual, 25)), float(np.percentile(actual, 75))


def restrict_to_band(points: List[dict], lo: float, hi: float) -> List[dict]:
    return [p for p in points if lo <= p["actual_retention"] <= hi]


def run_one(model: str, dataset: str, data_root: str, reference_selector: str,
            band_lo: Optional[float], band_hi: Optional[float], condition: str = "reached_patched"):
    print(f"\n{'=' * 90}")
    print(f"{model} / {dataset}")
    print("=" * 90)

    points_by_selector = load_trace_points_patched(model, dataset, data_root, condition=condition)

    if reference_selector not in points_by_selector:
        print(f"  [skip] reference_selector={reference_selector!r} not found here; "
              f"available: {sorted(points_by_selector)}")
        return

    this_lo, this_hi = band_lo, band_hi
    if this_lo is None or this_hi is None:
        this_lo, this_hi = band_from_reference(points_by_selector[reference_selector])

    print(f"Compression band (actual retention): [{this_lo:.3f}, {this_hi:.3f}]\n")
    print(f"{'selector':20s} {'n_traces_in_band':>17s} {'reached_acc':>12s} "
          f"{'full_acc':>10s} {'PR':>8s} {'PR_90%CI':>18s}")
    print("-" * 90)

    for selector, points in sorted(points_by_selector.items()):
        subset = restrict_to_band(points, this_lo, this_hi)
        n = len(subset)
        if n == 0:
            print(f"{selector:20s} {'0':>17s}  (no traces fall in this band)")
            continue

        reached = np.array([p["reached_correct"] for p in subset])
        full = np.array([p["full_correct"] for p in subset if p["full_correct"] is not None])

        if len(full) == 0:
            print(f"{selector:20s} {n:>17d}  (no full_correct data)")
            continue

        result = bootstrap_relative_pr(reached, full)
        if result is None:
            print(f"{selector:20s} {n:>17d}  (full_acc=0, PR undefined)")
            continue

        pe, lo, hi = result
        print(
            f"{selector:20s} {n:>17d} {reached.mean():>12.3f} "
            f"{full.mean():>10.3f} {pe:>8.3f} [{lo:.3f}, {hi:.3f}]"
        )


def main(
    model: str,
    dataset: Optional[str] = None,
    data_root: str = "entropy-traces-local",
    reference_selector: str = "numbers",
    band_lo: Optional[float] = None,
    band_hi: Optional[float] = None,
    condition: str = "reached_patched",
):
    """
    dataset: if omitted, runs the comparison for every dataset that has data
        for this model in data_root (discovered automatically, same logic
        as batch_plot.py) instead of requiring one call per dataset.
    condition: which per-trace field to read `correct` from. Default
        'reached_patched' (activations injected at selected positions --
        the causally meaningful comparison). Use 'reached_only' for the raw
        reconstructed-text condition (no patching) if you specifically want
        that instead.
    """
    model_short = model.split("/")[-1]

    if dataset is not None:
        run_one(model, dataset, data_root, reference_selector, band_lo, band_hi, condition)
        return

    pairs = discover_pairs(data_root)
    datasets_for_model = sorted(ds for ds, m in pairs if m == model_short)

    if not datasets_for_model:
        raise ValueError(
            f"No datasets found for model_short={model_short!r} under {data_root!r}"
        )

    print(f"Found {len(datasets_for_model)} dataset(s) for {model_short}: "
          f"{datasets_for_model}")

    for ds in datasets_for_model:
        run_one(model, ds, data_root, reference_selector, band_lo, band_hi, condition)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
