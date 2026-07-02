"""Trace activation and entropy collection experiment.

Ported from neurohike/experiments/trace_act_ent.py (branch daniel/entropy-shortcut).
Changes vs original:
  - Model thinking token config read from entropy.models.registry (not hardcoded)
  - Dataset config read from entropy.datasets.registry
  - BaseCfg replaced with a simpler TraceCollectionCfg dataclass
  - Removed teacher_model_name (unused here)
  - Same forward-pass logic, same output format (question_XXXX.pth + .jsonl)
  - OOM handled gracefully: traces that don't fit in memory are skipped

Output format per question_XXXX.pth
------------------------------------
{
    "input_text": str,
    "prompt_tokens": list[int],
    "GT_answer": str,
    "traces": [
        {
            "text": str,
            "tokens": list[int],
            "activations": Tensor [n_tokens, n_layers, hidden_dim],
            "entropies_hf": list[float],   # top-k=20, HF forward pass
            "entropies_vllm": list[float], # from original vLLM generation
            "answer": str,
        },
        ...
    ],
    "num_layers": int,
    "model_num_layers": int,
    "layers_collected": list[int],
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..core.model_loader import load_model_and_tokenizer
from ..core.trace_loader import get_reasoning_traces
from ..core.utils import extract_boxed_answer


@dataclass
class TraceCollectionCfg:
    model_name: str
    data_name: str
    output_dir: str
    top_k_for_entropy: int = 20
    max_traces: int | None = None
    max_seq_length: int | None = None
    max_questions: int | None = None
    layers: list[int] | list[float] | None = None
    quantization: str | None = None
    attn_implementation: str | None = None


class TraceActivationEntropy:
    """Collect activations and per-token entropy from pre-generated traces.

    Supports:
      - All model layers (default) or specific layers by index or percentile
      - Resumability (skips already-saved question_XXXX.pth files)
      - OOM handling: traces too long for GPU memory are skipped gracefully
      - max_seq_length as optional soft hint (skip before attempting forward pass)
      - max_questions limit for testing
    """

    def __init__(self, cfg: TraceCollectionCfg):
        self.cfg = cfg
        self.output_path = Path(cfg.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        print(f"Loading traces for {cfg.model_name} on {cfg.data_name}")
        self.reasoning_traces = get_reasoning_traces(cfg.model_name, cfg.data_name)
        if cfg.max_questions is not None:
            self.reasoning_traces = self.reasoning_traces[:cfg.max_questions]
            print(f"  Limiting to {cfg.max_questions} question(s)")
        print(f"  {len(self.reasoning_traces)} questions loaded")

        print(f"Loading model {cfg.model_name}")
        self.model, self.tokenizer, self.config = load_model_and_tokenizer(
            cfg.model_name,
            quantization=cfg.quantization,
            attn_implementation=cfg.attn_implementation,
        )
        self.num_layers: int = self.config["num_hidden_layers"]

        # Resolve which layers to collect
        if cfg.layers is None:
            self.layers_to_collect = list(range(self.num_layers))
        elif all(isinstance(v, float) for v in cfg.layers):
            self.layers_to_collect = sorted(set(
                min(int(round(p * self.num_layers)), self.num_layers - 1)
                for p in cfg.layers
            ))
        else:
            self.layers_to_collect = sorted(set(int(v) for v in cfg.layers))

        self._partial = cfg.layers is not None
        self._suffix = "_partial" if self._partial else ""
        self.jsonl_output = self.output_path / f"trace_activations_entropy{self._suffix}.jsonl"

        print(f"  Collecting {len(self.layers_to_collect)} / {self.num_layers} layers")

    # ------------------------------------------------------------------

    def _compute_top_k_entropy(self, logits: torch.Tensor) -> float:
        top_k_logits, _ = torch.topk(logits, self.cfg.top_k_for_entropy, dim=-1)
        probs = F.softmax(top_k_logits, dim=-1)
        return (-torch.sum(probs * torch.log(probs + 1e-10), dim=-1)).item()

    def _collect_from_trace(
        self,
        trace_text: str,
        prompt_tokens: list[int],
        trace_tokens: list[int],
        vllm_entropy: list[float],
    ) -> dict[str, Any] | None:
        """Forward pass with hidden states; collect activations + HF entropies.

        Returns None if the sequence is too long to fit in GPU memory.
        """
        prompt_len = len(prompt_tokens)
        trace_len = len(trace_tokens)
        full_ids = torch.tensor(
            [prompt_tokens + trace_tokens], dtype=torch.long, device=self.model.device
        )

        try:
            with torch.no_grad():
                outputs = self.model(full_ids, output_hidden_states=True, use_cache=False)
        except torch.OutOfMemoryError:
            del full_ids
            torch.cuda.empty_cache()
            return None

        # Entropies for trace tokens
        logits_cpu = outputs.logits[0].detach().cpu()
        hf_entropies = []
        for i in range(trace_len):
            idx = prompt_len + i - 1 if i > 0 else prompt_len - 1
            hf_entropies.append(self._compute_top_k_entropy(logits_cpu[idx]))
        del logits_cpu

        # Activations at selected layers, trace tokens only
        hidden = outputs.hidden_states
        all_acts = []
        for pos in range(prompt_len, prompt_len + trace_len):
            layer_acts = [hidden[l][0, pos].detach().cpu() for l in self.layers_to_collect]
            all_acts.append(torch.stack(layer_acts))
        stacked = torch.stack(all_acts)  # [trace_len, n_layers, hidden_dim]

        del outputs, hidden, full_ids
        torch.cuda.empty_cache()

        return {
            "text": trace_text,
            "tokens": trace_tokens,
            "activations": stacked,
            "entropies_hf": hf_entropies,
            "entropies_vllm": vllm_entropy,
            "answer": extract_boxed_answer(trace_text),
        }

    def _pending_indices(self) -> list[int]:
        existing = set()
        for f in self.output_path.glob("question_*.pth"):
            is_partial = f.stem.endswith("_partial")
            if self._partial != is_partial:
                continue
            try:
                existing.add(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        if existing:
            print(f"  Resuming: {len(existing)} questions already done")
        return [i for i in range(len(self.reasoning_traces)) if i not in existing]

    def run(self):
        pending = self._pending_indices()
        if not pending:
            print("All questions already processed.")
            return

        done = len(self.reasoning_traces) - len(pending)
        for idx in tqdm(pending, desc="Questions", initial=done,
                        total=len(self.reasoning_traces), colour="blue"):
            ele = self.reasoning_traces[idx]
            prompt_tokens = ele["prompt_tokens"]
            traces_out = []
            count = 0

            for i, text in enumerate(tqdm(ele.get("traces", []), desc="  Traces",
                                          leave=False, colour="green")):
                seq_len = len(prompt_tokens) + len(ele["traces_tokens"][i])

                # Optional soft cap: skip before attempting forward pass
                if self.cfg.max_seq_length and seq_len > self.cfg.max_seq_length:
                    tqdm.write(f"  Skipping q{idx} t{i}: {seq_len} > {self.cfg.max_seq_length}")
                    continue

                result = self._collect_from_trace(
                    text, prompt_tokens,
                    ele["traces_tokens"][i],
                    ele["traces_entropy"][i],
                )
                if result is None:
                    tqdm.write(f"  OOM q{idx} t{i}: seq_len={seq_len}, skipping")
                    continue

                traces_out.append(result)
                count += 1
                if self.cfg.max_traces and count >= self.cfg.max_traces:
                    break

            question_result = {
                "input_text": ele["input_text"],
                "prompt_tokens": prompt_tokens,
                "GT_answer": ele.get("GT_answer") or ele.get("gt_answer") or ele.get("answer", ""),
                "traces": traces_out,
                "num_layers": len(self.layers_to_collect),
                "model_num_layers": self.num_layers,
                "layers_collected": self.layers_to_collect,
            }

            out_file = self.output_path / f"question_{idx:04d}{self._suffix}.pth"
            torch.save(question_result, out_file)

            # Append lightweight JSONL record
            with open(self.jsonl_output, "a") as f:
                readable = {
                    "input_text": question_result["input_text"],
                    "GT_answer": question_result["GT_answer"],
                    "traces": [
                        {
                            "activations_shape": list(t["activations"].shape),
                            "entropies_hf_len": len(t["entropies_hf"]),
                            "answer": t["answer"],
                        }
                        for t in traces_out
                    ],
                    "layers_collected": self.layers_to_collect,
                }
                f.write(json.dumps(readable) + "\n")

            torch.cuda.empty_cache()

        print(f"Done. Output: {self.output_path}")