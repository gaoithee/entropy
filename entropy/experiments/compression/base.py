"""Base class for compression experiments.

Ported from neurohike/experiments/compression/base.py (branch daniel/entropy-shortcut).
Changes vs original:
  - Imports from entropy.* instead of neurohike.*
  - get_thinking_tokens imported from entropy.models.registry
  - load_essentials_model → entropy.core.model_loader.load_model_and_tokenizer
  - extract_boxed_answer → entropy.core.utils
  - BaseCfg / Experiment base classes inlined (no neurohike.experiments.base dependency)
  - Everything else (nnsight patching, sampling, metrics, boundary detection) unchanged
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from nnsight import LanguageModel
from tqdm import tqdm

from ...core.model_loader import load_model_and_tokenizer
from ...core.utils import extract_boxed_answer
from ...models.registry import get_thinking_tokens
from .common import CompressionResult


class CompressionBase:
    """Base class for compression experiments.

    Handles shared setup: model loading (via nnsight), thinking token
    resolution, question file discovery, answer suffix configuration.

    Provides shared methods:
      _generate_with_patching   — token-by-token generation with activation injection
      _find_thinking_boundaries — locate CoT region in token sequence
      _compute_metrics          — pass@k and pass_rate
      _sample_token             — temperature + top-p sampling
      _collapse_special_tokens  — readability helper for logging
    """

    def __init__(
        self,
        model_name: str,
        input_dir: str,
        output_dir: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        force_boxed_answer: bool = True,
        filter_gt_answer: bool = True,
        quantization: str | None = None,
        attn_implementation: str | None = None,
    ):
        self.model_name = model_name
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.force_boxed_answer = force_boxed_answer
        self.filter_gt_answer = filter_gt_answer

        # Load model via nnsight for activation patching
        print(f"Loading {model_name} via nnsight")
        self.student_model, self.student_tokenizer, config = load_model_and_tokenizer(
            model_name,
            model_type="nnsight",
            quantization=quantization,
            attn_implementation=attn_implementation,
        )
        self.num_layers: int = config["num_hidden_layers"]
        print(f"  {self.num_layers} layers")

        # Thinking token config
        self.thinking_tokens = get_thinking_tokens(model_name)
        self._resolve_thinking_token_ids()

        # Pad / dummy token
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        self.dummy_token_id = self.student_tokenizer.pad_token_id

        # Answer-forcing suffix
        self.answer_suffix = r"Therefore, the final answer is \boxed{"
        self.answer_suffix_ids = self.student_tokenizer.encode(
            self.answer_suffix, add_special_tokens=False
        )
        self.answer_max_new_tokens = 10

        print(f"Force boxed answer: {self.force_boxed_answer}")
        if self.force_boxed_answer:
            print(f"Answer suffix ({len(self.answer_suffix_ids)} tokens): {self.answer_suffix!r}")

        # Discover question files
        self.question_files = sorted(
            self.input_dir.glob("question_*.pth"),
            key=lambda f: int(f.stem.split("_")[1]),
        )[:200]
        if not self.question_files:
            raise FileNotFoundError(f"No question_*.pth files found in {self.input_dir}")
        print(f"Found {len(self.question_files)} question files")

    # ------------------------------------------------------------------
    # Thinking token resolution
    # ------------------------------------------------------------------

    def _resolve_thinking_token_ids(self):
        explicit_start = self.thinking_tokens.get("start_token_ids")
        explicit_end   = self.thinking_tokens.get("end_token_ids")

        if explicit_start is not None:
            self.start_thinking_ids = explicit_start
        else:
            self.start_thinking_ids = self.student_tokenizer.encode(
                self.thinking_tokens["start_token"], add_special_tokens=False
            )

        if explicit_end is not None:
            self.end_thinking_ids = explicit_end
        else:
            self.end_thinking_ids = self.student_tokenizer.encode(
                self.thinking_tokens["end_token"], add_special_tokens=False
            )

        print(f"Start thinking IDs: {self.start_thinking_ids}")
        print(f"End thinking IDs:   {self.end_thinking_ids}")

    # ------------------------------------------------------------------
    # Model layer access
    # ------------------------------------------------------------------

    def _get_model_layers(self):
        """Get the transformer layers module (architecture-agnostic)."""
        m = self.student_model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers          # Llama / Qwen / Gemma / Phi style
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h         # GPT-2 style
        if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
            return m.gpt_neox.layers       # GPT-NeoX style
        raise ValueError(f"Unknown model architecture for {self.model_name}")

    # ------------------------------------------------------------------
    # Thinking boundary detection
    # ------------------------------------------------------------------

    def _find_thinking_boundaries(
        self,
        tokens: list[int],
    ) -> tuple[int, int] | None:
        """Return (start_pos, end_pos) of thinking region, or None.

        start_pos : index of first token AFTER start_thinking delimiter
        end_pos   : index of first token OF end_thinking delimiter
        """
        start_pos = None
        n_start = len(self.start_thinking_ids)
        for i in range(len(tokens) - n_start + 1):
            if tokens[i:i + n_start] == self.start_thinking_ids:
                start_pos = i + n_start
                break
        if start_pos is None:
            return None

        n_end = len(self.end_thinking_ids)
        end_pos = None
        for i in range(start_pos, len(tokens) - n_end + 1):
            if tokens[i:i + n_end] == self.end_thinking_ids:
                end_pos = i
                break
        if end_pos is None:
            return None

        return start_pos, end_pos

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Temperature + top-p sampling. Unchanged from neurohike."""
        scaled = logits / self.temperature
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum_probs > self.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scaled[remove.scatter(0, sorted_idx, remove)] = float("-inf")
        return torch.multinomial(F.softmax(scaled, dim=-1), num_samples=1)

    # ------------------------------------------------------------------
    # Generation with activation patching
    # ------------------------------------------------------------------

    def _generate_with_patching(
        self,
        input_ids: torch.Tensor,
        activations_to_inject: list[torch.Tensor],
        prompt_len: int,
        max_new_tokens: int | None = None,
    ) -> CompressionResult:
        """Token-by-token generation with activation injection at ALL layers.

        Prompt structure expected:
          [question][start_thinking][dummy x N][end_thinking][answer_suffix]

        Activations are injected at the N dummy positions on the first forward
        pass.  Subsequent steps use the KV cache and need no patching.

        Unchanged from neurohike except import paths.
        """
        model: LanguageModel = self.student_model
        layers = self._get_model_layers()
        device = input_ids.device

        inject_start = prompt_len + len(self.start_thinking_ids)
        inject_positions = list(range(inject_start, inject_start + len(activations_to_inject)))
        gen_limit = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        generated_tokens: list[int] = []
        past_key_values = None

        for step in range(gen_limit):
            if step == 0:
                with torch.no_grad():
                    with model.trace(input_ids, use_cache=True):
                        if inject_positions:
                            for layer_idx in range(self.num_layers):
                                layer_out = layers[layer_idx].output
                                hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out
                                for pos_idx, pos in enumerate(inject_positions):
                                    act = activations_to_inject[pos_idx][layer_idx, :]
                                    hidden[:, pos, :] = act
                        logits      = model.lm_head.output.save()
                        cache_out   = model.output.save()

                past_key_values = cache_out.past_key_values
                current_logits  = logits[0, -1, :]

            else:
                last = torch.tensor(
                    [[generated_tokens[-1]]], device=device, dtype=input_ids.dtype
                )
                with torch.no_grad():
                    with model.trace(last, past_key_values=past_key_values, use_cache=True):
                        logits    = model.lm_head.output.save()
                        cache_out = model.output.save()

                past_key_values = cache_out.past_key_values
                current_logits  = logits[0, -1, :]

            next_id = self._sample_token(current_logits).item()
            generated_tokens.append(next_id)
            if next_id == self.student_tokenizer.eos_token_id:
                break

        compressed_prompt = self.student_tokenizer.decode(
            input_ids[0].tolist(), skip_special_tokens=False
        )
        generated_text = self.student_tokenizer.decode(
            generated_tokens, skip_special_tokens=False
        )

        full_for_answer = (self.answer_suffix + generated_text) if self.force_boxed_answer else generated_text
        answer = extract_boxed_answer(full_for_answer)

        return CompressionResult(
            compressed_prompt=compressed_prompt,
            generated_text=generated_text,
            answer=answer,
            num_peaks_used=len(activations_to_inject),
            original_thinking_length=0,
            compression_ratio=0.0,
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _normalize_answer(self, answer: str) -> str:
        if not isinstance(answer, str):
            answer = str(answer)
        return answer.strip().lower()

    def _compute_metrics(
        self,
        gt_answer: str,
        answers: list[str],
    ) -> tuple[int, float]:
        """Return (pass_at_k, pass_rate). Permissive substring matching."""
        if not answers:
            return 0, 0.0
        gt = self._normalize_answer(gt_answer)
        correct = sum(1 for a in answers if gt in self._normalize_answer(a))
        return (1 if correct > 0 else 0), correct / len(answers)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _collapse_special_tokens(self, text: str, threshold: int = 10) -> str:
        """Collapse long runs of repeated special tokens for readability."""
        for tok in self.student_tokenizer.all_special_tokens:
            escaped = re.escape(tok)
            pattern = rf"({escaped}){{{threshold},}}"
            def _rep(m, _t=tok):
                return f"{_t}x{len(m.group(0)) // len(_t)}"
            text = re.sub(pattern, _rep, text)
        return text
