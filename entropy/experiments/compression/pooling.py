"""Compression via token selection and activation pooling.

Ported from neurohike/experiments/compression/pooling.py (branch daniel/entropy-shortcut).
Changes vs original:
  - Inherits from entropy CompressionBase (same logic, different import paths)
  - Added "before_entropy" and "after_entropy" methods (in results but missing from original)
  - Added "numbers" method (used on ZebraLogic)
  - Removed high_curvature (analysis-only, not part of paper results)
  - Removed BaseCfg / Experiment dependency

Selection methods
-----------------
Score-based (k = ceil(retention_rate × thinking_length)):
    low_entropy     : top-k lowest entropy tokens  ← main finding
    high_entropy    : top-k highest entropy tokens (ablation)
    random          : random sample (baseline)
    before_entropy  : token immediately before each high-entropy peak
    after_entropy   : token immediately after each high-entropy peak

Content-based (k determined by matching, retention_rate ignored):
    newline         : tokens containing '\\n'
    end_of_sentence : tokens at sentence boundaries
    numbers         : tokens whose decoded text is purely numeric
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
from tqdm import tqdm

from .base import CompressionBase
from .common import CompressionResult


_SCORE_BASED   = {"low_entropy", "high_entropy", "random", "before_entropy", "after_entropy"}
_CONTENT_BASED = {"newline", "end_of_sentence", "numbers"}
ALL_METHODS    = tuple(_SCORE_BASED | _CONTENT_BASED)


@dataclass
class CompressionPoolingCfg:
    model_name: str
    input_dir: str
    output_dir: str
    retention_rate: float = 0.1
    selection_methods: list[str] = field(default_factory=lambda: list(ALL_METHODS))
    pooling: str = "mean"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    force_boxed_answer: bool = True
    filter_gt_answer: bool = True
    quantization: str | None = None
    attn_implementation: str | None = None


class CompressionPooling(CompressionBase):
    """Token-selection + mean-pooling compression with activation patching.

    For each trace:
      1. Identify thinking region [start_pos, end_pos)
      2. Select k anchor positions by chosen method
      3. Add boundary anchors {0, thinking_length-1}, deduplicate, sort
      4. Partition into segments; mean-pool activations per segment
      5. Replace thinking tokens with dummies; inject pooled activations
         at ALL layers via nnsight
      6. Generate answer; evaluate pass@k and pass_rate
    """

    def __init__(self, cfg: CompressionPoolingCfg):
        if isinstance(cfg.selection_methods, str):
            cfg.selection_methods = cfg.selection_methods.split()
        unknown = set(cfg.selection_methods) - set(ALL_METHODS)
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}. Available: {list(ALL_METHODS)}")

        super().__init__(
            model_name=cfg.model_name,
            input_dir=cfg.input_dir,
            output_dir=cfg.output_dir,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            force_boxed_answer=cfg.force_boxed_answer,
            filter_gt_answer=cfg.filter_gt_answer,
            quantization=cfg.quantization,
            attn_implementation=cfg.attn_implementation,
        )

        self.cfg = cfg
        self.methods = cfg.selection_methods

        rate_label = f"{cfg.retention_rate:.4f}".rstrip("0").rstrip(".")
        self.rate_label = rate_label
        self.output_file = (
            self.output_dir / f"compression_pooling_results_rate_{rate_label}.jsonl"
        )

        print(f"Methods: {self.methods}")
        print(f"Retention rate: {cfg.retention_rate}")
        print(f"Pooling: {cfg.pooling}")

    # ------------------------------------------------------------------
    # Anchor selection
    # ------------------------------------------------------------------

    def _select_anchors(
        self,
        method: str,
        thinking_tokens: list[int],
        thinking_entropies: list[float],
    ) -> list[int] | None:
        """Return sorted anchor positions (relative to thinking start), or None."""
        n = len(thinking_tokens)
        k = max(1, int(self.cfg.retention_rate * n))
        tok = self.student_tokenizer

        if method == "random":
            return sorted(random.sample(range(n), min(k, n)))

        if method == "low_entropy":
            return sorted(sorted(range(n), key=lambda i: thinking_entropies[i])[:k])

        if method == "high_entropy":
            return sorted(sorted(range(n), key=lambda i: -thinking_entropies[i])[:k])

        if method == "before_entropy":
            peaks = sorted(range(n), key=lambda i: -thinking_entropies[i])[:k]
            return sorted(set(max(0, p - 1) for p in peaks))

        if method == "after_entropy":
            peaks = sorted(range(n), key=lambda i: -thinking_entropies[i])[:k]
            return sorted(set(min(n - 1, p + 1) for p in peaks))

        if method == "newline":
            pos = [i for i in range(n)
                   if "\n" in tok.decode([thinking_tokens[i]], skip_special_tokens=False)]
            return pos if pos else None

        if method == "end_of_sentence":
            decoded = {t: tok.decode([t], skip_special_tokens=False)
                       for t in set(thinking_tokens)}
            pos: set[int] = set()
            for i in range(n):
                txt = decoded.get(thinking_tokens[i], "")
                if ".\n" in txt or ". " in txt:
                    pos.add(i)
                if txt.endswith(".") and i + 1 < n:
                    nxt = decoded.get(thinking_tokens[i + 1], "")
                    if nxt.startswith(("\n", " ")):
                        pos.add(i)
            return sorted(pos) if pos else None

        if method == "numbers":
            pos = [i for i in range(n)
                   if tok.decode([thinking_tokens[i]], skip_special_tokens=False).strip().isdigit()]
            return pos if pos else None

        raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    # Segment pooling
    # ------------------------------------------------------------------

    @staticmethod
    def _pool(acts: torch.Tensor, anchors: list[int]) -> list[torch.Tensor]:
        """Mean-pool activations within segments defined by anchors."""
        all_a = sorted(set(anchors) | {0, acts.shape[0] - 1})
        segs = []
        for i in range(len(all_a) - 1):
            s = all_a[i] if i == 0 else all_a[i] + 1
            e = all_a[i + 1]
            if s <= e:
                segs.append((s, e))
        return [acts[s:e + 1].mean(dim=0) for s, e in segs]

    # ------------------------------------------------------------------
    # Single trace processing
    # ------------------------------------------------------------------

    def _process_trace(
        self,
        q_id: int,
        t_idx: int,
        prompt_tokens: list[int],
        trace: dict,
        method: str,
    ) -> CompressionResult | None | str:
        tokens     = trace["tokens"]
        activations = trace["activations"]       # [n_tok, n_layers, hidden_dim]
        entropies  = trace["entropies_hf"]       # list[float], one per trace token

        bounds = self._find_thinking_boundaries(tokens)
        if bounds is None:
            tqdm.write(f"Warning: Q{q_id} T{t_idx} — no thinking boundaries, skipping")
            return None

        s, e = bounds
        n = e - s
        if n < 1:
            return None

        thinking_tokens    = tokens[s:e]
        thinking_entropies = entropies[s:e]
        thinking_acts      = activations[s:e]   # [n, n_layers, hidden_dim]

        anchors = self._select_anchors(method, thinking_tokens, thinking_entropies)
        if anchors is None:
            return "NOT_APPLICABLE"

        if len(anchors) >= n:
            pooled = [thinking_acts[i].to(self.student_model.device) for i in range(n)]
        else:
            pooled = [p.to(self.student_model.device)
                      for p in self._pool(thinking_acts, anchors)]

        if not pooled:
            return None

        num_dummies = len(pooled)
        full_seq = (
            prompt_tokens
            + self.start_thinking_ids
            + [self.dummy_token_id] * num_dummies
            + self.end_thinking_ids
            + self.answer_suffix_ids
        )
        input_ids = torch.tensor([full_seq], device=self.student_model.device, dtype=torch.long)

        result = self._generate_with_patching(
            input_ids, pooled, len(prompt_tokens), self.answer_max_new_tokens
        )
        result.original_thinking_length = n
        result.compression_ratio = num_dummies / n
        return result

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        total_orig_pr = 0.0
        total_pr: dict[str, float] = {m: 0.0 for m in self.methods}
        total_na: dict[str, int]   = {m: 0     for m in self.methods}
        n_processed = 0

        print(f"\nProcessing {len(self.question_files)} questions")

        for question_file in tqdm(self.question_files, desc="Questions", colour="blue"):
            try:
                data = torch.load(question_file, weights_only=False)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                tqdm.write(f"[OOM] {question_file.name}")
                continue

            q_id          = int(question_file.stem.split("_")[1])
            gt            = data["GT_answer"]
            prompt_tokens = data["prompt_tokens"]
            traces        = data["traces"]
            filter_val    = gt if self.cfg.filter_gt_answer else None

            orig_answers = [t["answer"] for t in traces]
            _, orig_pr   = self._compute_metrics(gt, orig_answers)

            method_results: dict[str, list[CompressionResult]] = {m: [] for m in self.methods}
            method_na:      dict[str, int]                     = {m: 0  for m in self.methods}

            for t_idx, trace in enumerate(tqdm(traces, desc=f"  Q{q_id}", leave=False, colour="green")):
                for method in self.methods:
                    result = self._process_trace(q_id, t_idx, prompt_tokens, trace, method)
                    if result == "NOT_APPLICABLE":
                        method_na[method] += 1
                    elif result is not None:
                        method_results[method].append(result)

            # Per-method metrics
            method_metrics: dict[str, float | None] = {}
            for method in self.methods:
                if method_na[method] == len(traces):
                    method_metrics[method] = None
                else:
                    _, pr = self._compute_metrics(gt, [r.answer for r in method_results[method]])
                    method_metrics[method] = pr

            total_orig_pr += orig_pr
            for method in self.methods:
                pr = method_metrics[method]
                if pr is not None:
                    total_pr[method] += pr
                else:
                    total_na[method] += 1
            n_processed += 1

            record = {
                "question_id": q_id,
                "GT_answer": gt,
                "original_traces": {
                    "answers": orig_answers,
                    "pass_rate": orig_pr,
                },
                "methods": {
                    m: {
                        "answers": [r.answer for r in method_results[m]],
                        "compression_ratios": [r.compression_ratio for r in method_results[m]],
                        "pass_rate": method_metrics[m],
                    }
                    for m in self.methods
                },
                "experiment_params": {
                    "retention_rate": self.cfg.retention_rate,
                    "selection_methods": self.methods,
                    "model": self.model_name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            with open(self.output_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            del data
            torch.cuda.empty_cache()

        # Summary
        if n_processed:
            print(f"\n{'='*50}")
            print(f"Questions processed: {n_processed}")
            print(f"Retention rate:      {self.cfg.retention_rate}")
            print(f"Original pass_rate:  {total_orig_pr / n_processed:.4f}")
            for m in self.methods:
                n_app = n_processed - total_na[m]
                avg = total_pr[m] / n_app if n_app > 0 else float("nan")
                print(f"  {m:20s}: {avg:.4f}  ({n_app}/{n_processed} applicable)")
            print(f"\nResults → {self.output_file}")
