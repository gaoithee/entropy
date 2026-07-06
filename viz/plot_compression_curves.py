#!/usr/bin/env python3
"""
Plot "Relative PR vs compression rate" curves per selector, reading the
result JSONs written by evaluate_entropy_splice.py
(data/{dataset}/{model}_teacher_traces_eval_{selector}_r{rate}.json).

Relative PR (performance retention) = reached_accuracy / full_accuracy,
computed per-trace and then aggregated with a bootstrap confidence band --
this normalizes out the baseline difficulty of the dataset/model, so 1.0
means "the compressed CoT recovers full performance", not "the model
solved the problem".

Two x-axis modes:
  --x_axis nominal  -> the --retention_rate passed to evaluate_entropy_splice.py
  --x_axis actual   -> n_tokens_reached / n_tokens_thinking (recommended for
                       numbers/newlines/end_of_sentence, whose nominal rate
                       can diverge a lot from what was actually kept once the
                       pattern pool is exhausted -- see pool_exh in the sweep
                       output tables)

USAGE:
    python plot_compression_curves.py \
        --model openai/gpt-oss-20b \
        --dataset zebralogic \
        --data_root data \
        --x_axis nominal \
        --output gptoss20b_zebralogic.png

Colors follow a fixed selector->color mapping so the same selector always
gets the same color across figures (matching the reference figure's
low_entropy=orange, high_entropy=blue, random=gray, newlines=red,
end_of_sentence=green, numbers=purple).
"""

import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe; still works fine inside Jupyter
import matplotlib.pyplot as plt
import numpy as np

SELECTOR_COLORS = {
    "low_entropy": "#ff7f0e",
    "high_entropy": "#1f77b4",
    "random": "#7f7f7f",
    "newlines": "#d62728",
    "end_of_sentence": "#2ca02c",
    "numbers": "#9467bd",
}

N_BOOTSTRAP = 1000


def load_trace_points(model: str, dataset: str, data_root: str) -> Dict[str, List[dict]]:
    """Load all matching result files for (model, dataset), return
    {selector: [point, ...]}.

    Each point carries per-trace correctness for both the 'reached_only'
    condition and the 'full_sequence' baseline, plus nominal/actual
    retention -- everything needed to compute Relative PR downstream
    without re-reading the files.
    """
    model_short = model.split("/")[-1]
    # Try both layouts: flat (data/{dataset}/{model}_eval_*.json) and the HF
    # upload layout (data/{dataset}/results/{model}_eval_*.json) -- whichever
    # matches first wins, so this works for both a local sweep output dir and
    # a downloaded HF snapshot without the caller needing to know which.
    patterns = [
        str(Path(data_root) / dataset / f"{model_short}_teacher_traces_eval_*.json"),
        str(Path(data_root) / dataset / "results" / f"{model_short}_teacher_traces_eval_*.json"),
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
            reached_info = trace.get("reached_patched", {})
            full_info = trace.get("full_sequence", {})
            reached_correct = reached_info.get("correct")
            full_correct = full_info.get("correct")

            if n_thinking is None or n_reached is None or reached_correct is None:
                continue  # malformed/partial entry, skip rather than crash
            if n_thinking <= 0:
                continue

            points_by_selector.setdefault(selector, []).append({
                "actual_retention": n_reached / n_thinking,
                "nominal_retention": trace.get("retention_rate"),
                "reached_correct": int(bool(reached_correct)),
                "full_correct": int(bool(full_correct)) if full_correct is not None else None,
            })

    return points_by_selector


def bootstrap_relative_pr(reached: np.ndarray, full: np.ndarray, n_boot: int = N_BOOTSTRAP):
    """Given per-trace 0/1 arrays for reached_correct and full_correct at a
    fixed (selector, rate) bucket, bootstrap the ratio of their means
    (Relative PR) to get a point estimate + 90% CI band.

    Resampling is paired (same trace indices for numerator and denominator
    each draw), since reached/full come from the same underlying traces --
    unpaired resampling would overstate the variance.
    """
    n = len(reached)
    if n == 0:
        return None
    full_mean = full.mean()
    if full_mean == 0:
        return None  # baseline itself never correct here; ratio undefined

    point_estimate = reached.mean() / full_mean

    boot_vals = []
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f_mean = full[idx].mean()
        if f_mean == 0:
            continue
        boot_vals.append(reached[idx].mean() / f_mean)

    if not boot_vals:
        return point_estimate, point_estimate, point_estimate

    boot_vals = np.array(boot_vals)
    lo, hi = np.percentile(boot_vals, [5, 95])
    return point_estimate, lo, hi


def bin_by_rate(points: List[dict], x_axis: str, n_bins: Optional[int] = None):
    """Group points into (rate -> {reached: [...], full: [...]}) buckets.

    x_axis='nominal': buckets are the exact --retention_rate values used in
        the sweep (discrete, no binning needed).
    x_axis='actual': buckets are n_bins equal-width bins over [0,1], since
        actual retention is continuous per-trace.
    """
    buckets: Dict[float, Dict[str, List[int]]] = {}

    if x_axis == "nominal":
        for p in points:
            rate = p["nominal_retention"]
            if rate is None:
                continue
            compression = round(1.0 - rate, 4)
            b = buckets.setdefault(compression, {"reached": [], "full": []})
            b["reached"].append(p["reached_correct"])
            if p["full_correct"] is not None:
                b["full"].append(p["full_correct"])
    else:
        n_bins = n_bins or 10
        edges = np.linspace(0, 1, n_bins + 1)
        for p in points:
            compression_x = 1.0 - p["actual_retention"]
            bin_idx = min(np.searchsorted(edges, compression_x, side="right") - 1, n_bins - 1)
            center = (edges[bin_idx] + edges[bin_idx + 1]) / 2
            b = buckets.setdefault(round(float(center), 4), {"reached": [], "full": []})
            b["reached"].append(p["reached_correct"])
            if p["full_correct"] is not None:
                b["full"].append(p["full_correct"])

    return buckets


def plot_curves(
    points_by_selector: Dict[str, List[dict]],
    x_axis: str,
    n_bins: int,
    title: str,
    output_path: str,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    for selector, points in sorted(points_by_selector.items()):
        buckets = bin_by_rate(points, x_axis=x_axis, n_bins=n_bins)
        if not buckets:
            continue

        xs, ys, los, his = [], [], [], []
        for rate in sorted(buckets):
            reached = np.array(buckets[rate]["reached"])
            full = np.array(buckets[rate]["full"])
            if len(full) == 0:
                continue
            result = bootstrap_relative_pr(reached, full)
            if result is None:
                continue
            pe, lo, hi = result
            xs.append(rate)
            ys.append(pe)
            los.append(lo)
            his.append(hi)

        if not xs:
            continue

        color = SELECTOR_COLORS.get(selector, None)
        ax.plot(xs, ys, marker="o", label=selector, color=color, linewidth=2)
        ax.fill_between(xs, los, his, color=color, alpha=0.15)

    ax.set_xlabel("Compression rate")
    ax.set_ylabel("Relative PR")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0))
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved -> {output_path}")


def summarize_pool_exhaustion(points_by_selector: Dict[str, List[dict]]):
    """Print how often actual retention undershoots nominal -- i.e. how
    often the pattern pool (numbers/newlines/end_of_sentence) was the real
    bottleneck, not the requested budget. Same diagnostic used when the
    sweep results were first inspected (see pool_exh column in
    evaluate_entropy_splice.py's own summary table).
    """
    print("\nPool-exhaustion check (actual vs nominal retention):")
    print(f"{'selector':20s} {'n_traces':>9s} {'mean_actual':>12s} {'mean_nominal':>13s} {'pct_undershoot':>15s}")
    for selector, points in sorted(points_by_selector.items()):
        if not points:
            continue
        actual = np.array([p["actual_retention"] for p in points])
        nominal_vals = [p["nominal_retention"] for p in points if p["nominal_retention"] is not None]
        nominal = np.array(nominal_vals) if len(nominal_vals) == len(actual) else None
        undershoot_pct = None
        if nominal is not None and len(nominal) > 0:
            undershoot_pct = 100.0 * np.mean(actual < (nominal - 1e-9))
        print(
            f"{selector:20s} {len(points):>9d} {actual.mean():>12.3f} "
            f"{'n/a' if nominal is None else f'{nominal.mean():.3f}':>13s} "
            f"{'n/a' if undershoot_pct is None else f'{undershoot_pct:.1f}%':>15s}"
        )


def main(
    model: str,
    dataset: str,
    data_root: str = "data",
    x_axis: str = "nominal",
    n_bins: int = 10,
    output: str = "compression_curves.png",
    title: Optional[str] = None,
):
    if x_axis not in ("nominal", "actual"):
        raise ValueError(f"--x_axis must be 'nominal' or 'actual', got {x_axis!r}")

    model_short = model.split("/")[-1]
    if title is None:
        title = f"{model_short} - {dataset.upper()}"

    points_by_selector = load_trace_points(model, dataset, data_root)
    print("Loaded points per selector:")
    for sel, pts in sorted(points_by_selector.items()):
        print(f"  {sel}: {len(pts)} points")

    summarize_pool_exhaustion(points_by_selector)
    plot_curves(points_by_selector, x_axis=x_axis, n_bins=n_bins, title=title, output_path=output)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
