#!/usr/bin/env python3
"""
Evaluate token-selection strategies (inside CoT) and the no-CoT baseline
on teacher_traces.json, mirroring evaluate_attribution.py but with
vLLM-precomputed entropy / structural criteria instead of GIM scores,
and no .pth/activation-collection dependency.

Conditions per trace (patched ones optional via --skip_patched):
  1. full_sequence    - prompt + start_think + full thinking + end_thinking (baseline)
  2. reached_only     - prompt + start_think + selector-chosen thinking tokens + end_thinking
  3. reached_patched  - same tokens, residual stream patched from full run   [skippable]
  4. random_only      - prompt + start_think + random thinking tokens (same count) + end_thinking
  5. random_patched   - random tokens, patched                              [skippable]
  6. no_cot           - prompt + start_think + end_think + suffix, NO thinking content at all

FIX (vs earlier version): full_sequence / random_only / reached_only now
correctly include `start_ids` (the <|start|>assistant<|channel|>analysis
<|message|> boundary) before the spliced thinking content. Previously
`_find_thinking_boundaries` returns `t_start` already *past* start_ids
(by construction), so the reconstructed prefix was missing the channel-
open marker entirely -- the model was being asked to "continue" thinking
content with no signal that it was in the analysis channel at all. This
is now fixed in all three prefix-construction sites. `_build_no_cot` was
already correct (it includes start_ids explicitly).

Because start_ids is now prepended before the spliced/random/full token
span, `patch_offset` passed to `generate_with_patching` must shift by
`len(start_ids)` as well, since that's where the spliced content now
actually starts inside the new sequence (immediately after
prompt_tokens + start_ids, not immediately after prompt_tokens alone).

SWEEP SUPPORT: --retention_rate and --selector both accept comma-separated
lists, e.g. --retention_rate "0.01,0.05,0.1,0.5,1.0" --selector
"low_entropy,random". All combinations run under a SINGLE model load (no
reload between combos). Conditions that don't depend on the combo are
computed once per trace and reused:
    - full_sequence, no_cot, full_resid (patching source): independent of
      both selector and retention_rate - computed once per trace.
    - random_only, random_patched: depend on retention_rate (via budget)
      but NOT on selector - computed once per (trace, retention_rate) and
      reused across all selectors at that rate.
    - reached_only, reached_patched: depend on both selector and
      retention_rate - computed for every (selector, retention_rate) pair.
One output JSON file is written per (selector, retention_rate) combo,
named as before; a combined summary table is printed at the end covering
every combo in the sweep.

`--selector` values (see per-function docstrings below for what each does):
  low_entropy, high_entropy, numbers, newlines, end_of_sentence, random

`no_cot` boundaries (start_think/end_think token ids) are model-dependent and
are resolved via entropy.models.registry.get_thinking_tokens - NOT reimplemented
here.

Usage:
    # Single combo (original behavior)
    python evaluate_entropy_splice.py \
        --model openai/gpt-oss-20b \
        --traces_file data/zebralogic/gpt-oss-20b_teacher_traces.json \
        --retention_rate 0.1 \
        --selector low_entropy

    # Sweep: many retention rates, one selector, one model load
    python evaluate_entropy_splice.py \
        --model openai/gpt-oss-20b \
        --traces_file data/aime2025/gpt-oss-20b_teacher_traces.json \
        --retention_rate "0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00" \
        --selector low_entropy \
        --max_new_tokens 150 \
        --skip_patched True

    # Sweep: many retention rates AND many selectors, one model load
    python evaluate_entropy_splice.py \
        --model openai/gpt-oss-20b \
        --traces_file data/aime2025/gpt-oss-20b_teacher_traces.json \
        --retention_rate "0.01,0.1,0.5,1.0" \
        --selector "low_entropy,high_entropy,numbers,newlines,end_of_sentence,random" \
        --max_new_tokens 150 \
        --skip_patched True

    # Smoke test: 1 question, 1 trace, generous max_new_tokens
    python evaluate_entropy_splice.py \
        --model openai/gpt-oss-20b \
        --traces_file data/zebralogic/gpt-oss-20b_teacher_traces.json \
        --retention_rate 0.5 \
        --selector low_entropy \
        --max_questions 1 \
        --max_traces 1 \
        --max_new_tokens 50 \
        --skip_patched True
"""

import gc
import json
import random
import re
from pathlib import Path
from typing import List, Optional

import torch
from nnsight import LanguageModel
from entropy.core.utils import check_correct

# ---------------------------------------------------------------------------
# Model / layer access
# ---------------------------------------------------------------------------

def get_model_layers(lm: LanguageModel):
    m = lm
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        return m.gpt_neox.layers
    raise ValueError("Unknown model architecture")


def greedy_generate(
    lm: LanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 10,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """Plain greedy generation, no patching."""
    generated_tokens: List[int] = []
    past_key_values = None
    device = input_ids.device

    for step in range(max_new_tokens):
        if step == 0:
            with torch.no_grad():
                with lm.trace(input_ids, use_cache=True):
                    logits = lm.lm_head.output.save()
                    cache_output = lm.output.save()
            past_key_values = cache_output.past_key_values
            current_logits = logits[0, -1, :]
        else:
            last = torch.tensor([[generated_tokens[-1]]], device=device, dtype=input_ids.dtype)
            with torch.no_grad():
                with lm.trace(last, past_key_values=past_key_values, use_cache=True):
                    logits = lm.lm_head.output.save()
                    cache_output = lm.output.save()
            past_key_values = cache_output.past_key_values
            current_logits = logits[0, -1, :]

        next_id = current_logits.argmax().item()
        if eos_token_id is not None and next_id == eos_token_id:
            break
        generated_tokens.append(next_id)

    return generated_tokens


def collect_residual_stream(lm: LanguageModel, num_layers: int, tokens: torch.Tensor) -> torch.Tensor:
    """Collect residual stream (block output) at every layer. [n_layers, seq_len, d_model]."""
    layers = get_model_layers(lm)
    saves = {}
    with torch.no_grad():
        with lm.trace(tokens):
            for L in range(num_layers):
                block_out = layers[L].output
                hidden = block_out[0] if isinstance(block_out, tuple) else block_out
                saves[L] = hidden.save()

    out = []
    for L in range(num_layers):
        val = saves[L]
        if val.dim() == 2:
            val = val.unsqueeze(0)
        out.append(val[0])
    return torch.stack(out, dim=0)


def generate_with_patching(
    lm: LanguageModel,
    num_layers: int,
    short_tokens: torch.Tensor,
    full_resid: torch.Tensor,
    reached_positions: List[int],
    patch_offset: int,
    tokenizer,
    max_new_tokens: int = 10,
) -> str:
    """Prefill with residual-stream patching at the spliced positions, then greedy-generate.

    `reached_positions` are absolute positions in the FULL sequence (prompt + trace).
    `patch_offset` is where the spliced tokens start in `short_tokens`
    (== len(prompt_tokens) + len(start_ids), since start_ids is now always
    prepended before the spliced/random/full thinking span -- see module
    docstring FIX note).
    """
    layers = get_model_layers(lm)
    n_patch = len(reached_positions)
    generated_tokens = []
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            with torch.no_grad():
                with lm.trace(short_tokens, use_cache=True):
                    for L in range(num_layers):
                        block_out = layers[L].output
                        hidden = block_out[0] if isinstance(block_out, tuple) else block_out
                        for i in range(n_patch):
                            new_pos = patch_offset + i
                            orig_pos = reached_positions[i]
                            hidden[:, new_pos, :] = full_resid[L, orig_pos, :]
                    logits = lm.lm_head.output.save()
                    cache_output = lm.output.save()
            past_key_values = cache_output.past_key_values
            current_logits = logits[0, -1, :]
        else:
            last_token = torch.tensor(
                [[generated_tokens[-1]]], device=short_tokens.device, dtype=short_tokens.dtype
            )
            with torch.no_grad():
                with lm.trace(last_token, past_key_values=past_key_values, use_cache=True):
                    logits = lm.lm_head.output.save()
                    cache_output = lm.output.save()
            past_key_values = cache_output.past_key_values
            current_logits = logits[0, -1, :]

        next_id = current_logits.argmax().item()
        if next_id == tokenizer.eos_token_id:
            break
        generated_tokens.append(next_id)

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def load_lm(model_name: str, quantization: str | None = None, attn_implementation: str | None = "eager"):
    """Load model + tokenizer + config via the shared entropy.core.model_loader.

    NOTE: entropy.core.model_loader.load_model_and_tokenizer always uses
    nnsight.LanguageModel for model_type="nnsight", which fails for
    multimodal-registered models (e.g. Gemma-4, registered as
    AutoModelForImageTextToText even when used text-only -- see traceback
    from gemma-4-E4B-it). That bug lives in the shared loader, not here;
    this wrapper works around it locally by swapping in VisionLanguageModel
    when needed, but the proper fix is in core/model_loader.py so every
    other experiment (CompressionBase included) benefits too.
    NOT VERIFIED: whether VisionLanguageModel exposes the same
    .trace()/.lm_head.output/.output/.layers[L].output hooks used below.
    """
    from entropy.core.model_loader import load_model_and_tokenizer

    m = model_name.lower()
    if "gemma-4" in m or "gemma4" in m:
        from transformers import AutoTokenizer
        from nnsight import VisionLanguageModel

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        kwargs = {"dtype": torch.bfloat16, "device_map": "auto", "trust_remote_code": True}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        print("Loading via VisionLanguageModel (gemma-4 workaround)")
        model = VisionLanguageModel(model_name, **kwargs)

        raw_config = model._model.config.to_dict() if hasattr(model, "_model") else model.config.to_dict()
        text_config = raw_config.get("text_config", raw_config)
        config = {
            "num_hidden_layers": text_config.get("num_hidden_layers", text_config.get("n_layer", 0)),
            "hidden_size": text_config.get("hidden_size", text_config.get("n_embd", 0)),
        }
        return model, tokenizer, config

    print("Loading via shared load_model_and_tokenizer (nnsight.LanguageModel)")
    return load_model_and_tokenizer(
        model_name, model_type="nnsight",
        quantization=quantization, attn_implementation=attn_implementation,
    )


# ---------------------------------------------------------------------------
# Thinking-region boundary detection
# ---------------------------------------------------------------------------

def _find_thinking_boundaries(tokens: List[int], start_ids: List[int], end_ids: List[int]):
    """Locate the thinking region inside `tokens` (a single trace).

    Note: for gpt-oss with this dataset's chat template, start_ids
    (<|channel|>analysis<|message|>) are the *first* tokens generated by
    the model -- i.e. they live inside trace_tokens, not prompt_tokens.
    The returned `start_pos` is already PAST start_ids (start_pos = i + n_s),
    i.e. it points to the first token of actual thinking CONTENT, not to
    start_ids itself. Callers that reconstruct a prefix from
    tokens[start_pos:end_pos] must therefore re-prepend start_ids
    themselves if they want the channel-open marker present -- see
    evaluate_entropy_splice.main() for where this is done.
    end_ids may be missing if generation was truncated at max_tokens
    before the model reached the final channel (~16% of traces on
    Zebra Logic with max_tokens=16384) - callers should treat None as
    "skip this trace", not retry with a fallback boundary.
    """
    n_s = len(start_ids)
    start_pos = None
    for i in range(len(tokens) - n_s + 1):
        if tokens[i:i + n_s] == start_ids:
            start_pos = i + n_s
            break
    if start_pos is None:
        return None
    n_e = len(end_ids)
    for i in range(start_pos, len(tokens) - n_e + 1):
        if tokens[i:i + n_e] == end_ids:
            return start_pos, i
    return None  # truncated trace: no end_thinking found


# ---------------------------------------------------------------------------
# Selection criteria ("inside CoT")
# ---------------------------------------------------------------------------
# All selectors take an absolute position range [start_pos, end_pos) into the
# *full* (prompt + trace) token/entropy arrays and return a sorted list of
# absolute positions within that range, sized to a "budget" derived from
# retention_rate for the entropy/random selectors, or capped at that same
# budget (truncated, not resampled) for the categorical selectors (numbers,
# newlines, end_of_sentence) since those can't be arbitrarily up/down-sized.

def _budget(start_pos: int, end_pos: int, retention_rate: float) -> int:
    thinking_len = end_pos - start_pos
    raw = int(retention_rate * thinking_len)
    return max(1, min(raw, thinking_len))


def _reached_from_entropy(
    entropies: List[float], start_pos: int, end_pos: int,
    retention_rate: float, mode: str,
) -> List[int]:
    """Select positions within [start_pos, end_pos) by entropy, sorted ascending.

    mode: 'low' (lowest-entropy / most confident tokens) or
          'high' (highest-entropy / most surprising tokens)
    """
    num_peaks = _budget(start_pos, end_pos, retention_rate)
    indexed = [(entropies[i], i) for i in range(start_pos, end_pos)]
    indexed.sort(key=lambda x: x[0], reverse=(mode == "high"))
    return sorted(i for _, i in indexed[:num_peaks])


def _reached_numbers_only(
    tokens: List[int], start_pos: int, end_pos: int,
    tokenizer, retention_rate: float,
) -> List[int]:
    """Positions whose decoded text is purely numeric (digits, optional sign/
    decimal separators, tolerant of a leading tokenizer space, e.g. ' 12', '3',
    '0.5', '1,000').

    CAUTION (byte-level BPE): a multi-digit number can be split across several
    tokens (e.g. "1234" -> "12" + "34"), and whether a token carries a leading
    space depends on what preceded it. This decodes each token in isolation,
    so it will miss numeric *fragments* that don't independently look numeric
    once decoded -- good enough as a first pass, but not a fully faithful
    "is this token part of a number" detector. If that matters, switch to a
    cumulative-text + char-offset remap like `_reached_end_of_sentence_only`.
    """
    budget = _budget(start_pos, end_pos, retention_rate)
    pattern = re.compile(r"\s*[+-]?\d[\d.,]*\s*$")
    matched = [
        i for i in range(start_pos, end_pos)
        if pattern.fullmatch(tokenizer.decode([tokens[i]]))
    ]
    if len(matched) > budget:
        matched = matched[:budget]
    return matched


def _reached_newlines_only(
    tokens: List[int], start_pos: int, end_pos: int,
    tokenizer, retention_rate: float,
) -> List[int]:
    """Positions whose decoded text contains a newline character."""
    budget = _budget(start_pos, end_pos, retention_rate)
    matched = [
        i for i in range(start_pos, end_pos)
        if "\n" in tokenizer.decode([tokens[i]])
    ]
    if len(matched) > budget:
        matched = matched[:budget]
    return matched


def _reached_end_of_sentence_only(
    tokens: List[int], start_pos: int, end_pos: int,
    tokenizer, retention_rate: float,
) -> List[int]:
    """Positions at sentence-ending punctuation boundaries ('. ', '.\\n', '? ',
    '! '), scanning the whole thinking region (not just the last sentence --
    contrast with evaluate_attribution.py's _find_last_sentence_start, which
    only locates ONE boundary near the end for post-hoc filtering).

    Uses a cumulative-text + char-offset remap (rather than isolated
    per-token decode) so multi-token punctuation sequences are handled
    correctly regardless of tokenizer boundary quirks.
    """
    budget = _budget(start_pos, end_pos, retention_rate)

    ids_region = tokens[start_pos:end_pos]
    token_texts = [tokenizer.decode([tid]) for tid in ids_region]
    cumulative = ""
    char_starts = []
    for t in token_texts:
        char_starts.append(len(cumulative))
        cumulative += t

    pattern = re.compile(r"[.!?]\s")
    boundary_chars = [m.end() for m in pattern.finditer(cumulative)]

    matched = []
    for bc in boundary_chars:
        for local_pos, cs in enumerate(char_starts):
            if cs >= bc:
                matched.append(start_pos + local_pos)
                break

    matched = sorted(set(matched))
    if len(matched) > budget:
        matched = matched[:budget]
    return matched


def _sample_random_positions(lo: int, hi: int, n_sample: int, exclude: List[int], seed: int) -> List[int]:
    rng = random.Random(seed)
    candidates = [i for i in range(lo, hi) if i not in set(exclude)]
    if len(candidates) <= n_sample:
        return sorted(candidates)
    return sorted(rng.sample(candidates, n_sample))


_VALID_SELECTORS = ("low_entropy", "high_entropy", "numbers", "newlines", "end_of_sentence", "random")


def select_positions(
    selector: str,
    tokens: List[int],
    entropies: List[float],
    start_pos: int,
    end_pos: int,
    retention_rate: float,
    tokenizer,
    seed: int,
) -> List[int]:
    """Dispatch to the right selection criterion. Returns sorted absolute positions."""
    if selector == "low_entropy":
        return _reached_from_entropy(entropies, start_pos, end_pos, retention_rate, "low")
    elif selector == "high_entropy":
        return _reached_from_entropy(entropies, start_pos, end_pos, retention_rate, "high")
    elif selector == "numbers":
        return _reached_numbers_only(tokens, start_pos, end_pos, tokenizer, retention_rate)
    elif selector == "newlines":
        return _reached_newlines_only(tokens, start_pos, end_pos, tokenizer, retention_rate)
    elif selector == "end_of_sentence":
        return _reached_end_of_sentence_only(tokens, start_pos, end_pos, tokenizer, retention_rate)
    elif selector == "random":
        budget = _budget(start_pos, end_pos, retention_rate)
        return _sample_random_positions(start_pos, end_pos, budget, exclude=[], seed=seed)
    else:
        raise ValueError(f"Unknown selector: {selector!r}, expected one of {_VALID_SELECTORS}")


# ---------------------------------------------------------------------------
# No-CoT baseline ("outside CoT")
# ---------------------------------------------------------------------------

def _build_no_cot(prompt_tokens: List[int], start_ids: List[int], end_ids: List[int]) -> List[int]:
    """prompt + start_think + end_think + suffix -- zero thinking content.

    start_ids/end_ids are resolved by the caller via
    entropy.models.registry.get_thinking_tokens (model-dependent), NOT
    reimplemented here.
    """
    return prompt_tokens + start_ids + end_ids


# Column widths for aligned output
_COL_TRACE = 10
_COL_INFO = 48


def _fmt_bool(v: bool) -> str:
    return " True" if v else "False"


def _parse_list_arg(value, cast=float) -> List:
    """Accept either a single value or a comma-separated string; return a list."""
    if isinstance(value, (list, tuple)):
        return [cast(v) for v in value]
    s = str(value)
    return [cast(v.strip()) for v in s.split(",") if v.strip()]


def main(
    model: str,
    traces_file: str,
    device: str = "cuda",
    output_file: Optional[str] = None,
    max_new_tokens: int = 10,
    retention_rate="0.1",
    selector: str = "low_entropy",
    seed: int = 42,
    max_questions: Optional[int] = None,
    max_traces: Optional[int] = None,
    trace_offset: int = 0,
    question_offset: int = 0,
    skip_patched: bool = False,
):
    """Evaluate one or more token-selection criteria against full-CoT, random,
    and no-CoT baselines, across one or more retention rates -- all under a
    SINGLE model load.

    Args:
        retention_rate: float, or comma-separated string of floats (e.g.
            "0.01,0.05,0.1,0.5,1.0") to sweep multiple retention rates in
            one process without reloading the model.
        selector: one of _VALID_SELECTORS, or a comma-separated string of
            several (e.g. "low_entropy,random") to sweep multiple selectors
            in one process. See module docstring for what each does.
    """
    rates = _parse_list_arg(retention_rate, cast=float)
    selectors = _parse_list_arg(selector, cast=str)
    for sel in selectors:
        if sel not in _VALID_SELECTORS:
            raise ValueError(f"--selector must be one of {_VALID_SELECTORS}, got {sel!r}")

    traces_path = Path(traces_file)
    with open(traces_path) as f:
        questions = json.load(f)
    if question_offset:
        questions = questions[question_offset:]
    if max_questions is not None:
        questions = questions[:max_questions]

    print(f"Loading model: {model}")
    lm, tokenizer, config = load_lm(model)
    lm.eval()
    tokenizer = lm.tokenizer

    from entropy.models.registry import get_thinking_tokens
    thinking_cfg = get_thinking_tokens(model)
    start_ids = thinking_cfg["start_token_ids"] or tokenizer.encode(
        thinking_cfg["start_token"], add_special_tokens=False
    )
    end_ids = thinking_cfg["end_token_ids"] or tokenizer.encode(
        thinking_cfg["end_token"], add_special_tokens=False
    )

    num_layers = config["num_hidden_layers"]
    print(f"Model: {num_layers} layers")
    print(f"Sweep: {len(selectors)} selector(s) x {len(rates)} retention_rate(s) "
          f"= {len(selectors) * len(rates)} combo(s). skip_patched={skip_patched}")

    # results_questions[(selector, rate)] -> list of per-question dicts
    combo_keys = [(sel, r) for sel in selectors for r in rates]
    results_questions = {k: [] for k in combo_keys}
    stats = {
        k: {
            "full_correct": 0, "reached_correct": 0, "patched_correct": 0,
            "random_correct": 0, "random_patched_correct": 0, "no_cot_correct": 0,
            "total": 0,
        }
        for k in combo_keys
    }
    truncated_skipped = 0

    # start_ids is now always prepended before any spliced/random/full
    # thinking span (see FIX note in module docstring), so the offset at
    # which the spliced span actually begins inside the reconstructed
    # sequence is len(prompt_tokens) + len(start_ids), not just
    # len(prompt_tokens). Used for residual-stream patching alignment.
    patch_base_offset_extra = len(start_ids)

    for q_idx, q in enumerate(questions):
        gt_answer = str(q["GT_answer"])
        prompt_tokens = q["prompt_tokens"]

        traces_tokens = q["traces_tokens"]
        traces_entropy = q["traces_entropy"]
        extracted_answers = q.get("extracted_answers", [None] * len(traces_tokens))
        if trace_offset:
            traces_tokens = traces_tokens[trace_offset:]
            traces_entropy = traces_entropy[trace_offset:]
            extracted_answers = extracted_answers[trace_offset:]

        n_good_traces_this_question = 0

        for t_idx, (trace_tokens, trace_entropy, orig_extracted) in enumerate(
            zip(traces_tokens, traces_entropy, extracted_answers)
        ):
            if max_traces is not None and n_good_traces_this_question >= max_traces:
                break  # already have enough good traces for this question

            full_ids_list = prompt_tokens + trace_tokens
            full_entropy = [0.0] * len(prompt_tokens) + list(trace_entropy)

            boundaries = _find_thinking_boundaries(trace_tokens, start_ids, end_ids)
            if boundaries is None:
                print(f"  Q{q_idx} Trace {t_idx}: no end_thinking boundary "
                      f"(truncated at {len(trace_tokens)} tokens), skipping (does not count toward budget)")
                truncated_skipped += 1
                continue

            # Pre-filter: original trace had no \boxed{} extracted at generation
            # time -> skip without counting against max_traces (mirrors the
            # 'truncated'/'no_boxed' pathology categories from
            # reevaluate_trace_correct.py / find_good_trace.py).
            if orig_extracted is not None and not str(orig_extracted).strip():
                print(f"  Q{q_idx} Trace {t_idx}: original extracted_answers empty (no boxed), "
                      f"skipping (does not count toward budget)")
                continue

            t_start, t_end = boundaries
            start_pos = len(prompt_tokens) + t_start
            end_pos = len(prompt_tokens) + t_end

            trace_seed_base = seed + q_idx * 1000 + t_idx
            print(f"  Q{q_idx} T{t_idx}: {end_pos - start_pos} thinking tokens, starting generations... "
                  f"({n_good_traces_this_question}/{max_traces if max_traces is not None else '?'} "
                  f"good traces so far for this question)")

            try:
                # ---- Conditions independent of (selector, rate): compute ONCE per trace ----
                # NOTE: start_ids prepended here -- FIX, see module docstring.
                full_seq_ids = torch.tensor(
                    [prompt_tokens + start_ids + trace_tokens[t_start:t_end] + end_ids], device=device
                )
                gen_ids = greedy_generate(lm, full_seq_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                full_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                full_correct = check_correct(full_gen_text, gt_answer)

                no_cot_ids = torch.tensor([_build_no_cot(prompt_tokens, start_ids, end_ids)], device=device)
                gen_ids = greedy_generate(lm, no_cot_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                no_cot_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                no_cot_correct = check_correct(no_cot_gen_text, gt_answer)

                full_resid = None
                if not skip_patched:
                    full_ids = torch.tensor([full_ids_list], device=device)
                    full_resid = collect_residual_stream(lm, num_layers, full_ids)

                # ---- Conditions independent of selector, cached per rate ----
                random_cache = {}  # rate -> (random_positions, random_correct, random_patched_correct)

                for rate in rates:
                    budget = _budget(start_pos, end_pos, rate)
                    random_positions = _sample_random_positions(
                        start_pos, end_pos, budget, exclude=[], seed=trace_seed_base,
                    )
                    # NOTE: start_ids prepended here -- FIX, see module docstring.
                    random_ids = torch.tensor(
                        [prompt_tokens + start_ids + [full_ids_list[p] for p in random_positions] + end_ids],
                        device=device,
                    )
                    gen_ids = greedy_generate(lm, random_ids, max_new_tokens=max_new_tokens,
                                               eos_token_id=tokenizer.eos_token_id)
                    random_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    random_correct = check_correct(random_gen_text, gt_answer)

                    random_patched_correct = None
                    if not skip_patched:
                        gen_text = generate_with_patching(
                            lm, num_layers, random_ids, full_resid, random_positions,
                            patch_offset=len(prompt_tokens) + patch_base_offset_extra,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                        )
                        random_patched_correct = check_correct(gen_text, gt_answer)

                    random_cache[rate] = {
                        "positions": random_positions,
                        "gen_text": random_gen_text,
                        "correct": random_correct,
                        "patched_correct": random_patched_correct,
                    }

                # ---- Conditions depending on (selector, rate) ----
                for sel in selectors:
                    for rate in rates:
                        key = (sel, rate)
                        reached_positions = select_positions(
                            sel, full_ids_list, full_entropy, start_pos, end_pos,
                            rate, tokenizer, seed=trace_seed_base,
                        )
                        if not reached_positions:
                            continue

                        # NOTE: start_ids prepended here -- FIX, see module docstring.
                        short_ids = torch.tensor(
                            [prompt_tokens + start_ids + [full_ids_list[p] for p in reached_positions] + end_ids],
                            device=device,
                        )
                        gen_ids = greedy_generate(lm, short_ids, max_new_tokens=max_new_tokens,
                                                   eos_token_id=tokenizer.eos_token_id)
                        reached_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        reached_correct = check_correct(reached_gen_text, gt_answer)

                        patched_correct = None
                        if not skip_patched:
                            gen_text = generate_with_patching(
                                lm, num_layers, short_ids, full_resid, reached_positions,
                                patch_offset=len(prompt_tokens) + patch_base_offset_extra,
                                tokenizer=tokenizer,
                                max_new_tokens=max_new_tokens,
                            )
                            patched_correct = check_correct(gen_text, gt_answer)

                        rc = random_cache[rate]
                        trace_result = {
                            "trace_index": t_idx,
                            "selector": sel,
                            "retention_rate": rate,
                            "n_tokens_thinking": end_pos - start_pos,
                            "n_tokens_reached": len(reached_positions),
                            "full_sequence": {"generated_answer": full_gen_text, "correct": full_correct},
                            "reached_only": {"generated_answer": reached_gen_text, "correct": reached_correct},
                            "random_only": {"generated_answer": rc["gen_text"], "correct": rc["correct"]},
                            "no_cot": {"generated_answer": no_cot_gen_text, "correct": no_cot_correct},
                        }
                        if not skip_patched:
                            trace_result["reached_patched"] = {"correct": patched_correct}
                            trace_result["random_patched"] = {"correct": rc["patched_correct"]}

                        s = stats[key]
                        s["total"] += 1
                        if full_correct:
                            s["full_correct"] += 1
                        if reached_correct:
                            s["reached_correct"] += 1
                        if rc["correct"]:
                            s["random_correct"] += 1
                        if no_cot_correct:
                            s["no_cot_correct"] += 1
                        if not skip_patched:
                            if patched_correct:
                                s["patched_correct"] += 1
                            if rc["patched_correct"]:
                                s["random_patched_correct"] += 1

                        results_questions[key].append({
                            "question_id": q_idx,
                            "GT_answer": gt_answer,
                            "trace": trace_result,
                        })

                        print(
                            f"  Q{q_idx} T{t_idx} sel={sel:14s} rate={rate:.2f}  "
                            f"full={_fmt_bool(full_correct)}  reached={_fmt_bool(reached_correct)}  "
                            f"random={_fmt_bool(rc['correct'])}  no_cot={_fmt_bool(no_cot_correct)}"
                        )

                del full_resid
                n_good_traces_this_question += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"  Q{q_idx} Trace {t_idx}: [OOM], skipping "
                          f"(does not count toward budget, trying next trace)")
                else:
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if (q_idx + 1) % 5 == 0 or q_idx == len(questions) - 1:
            print(f"  ...processed {q_idx + 1}/{len(questions)} questions")

    # ---- Write one output file per combo, print combined summary ----
    print()
    header = (
        f"{'selector':16s} {'rate':>6s} {'total':>6s} {'full':>7s} "
        f"{'reached':>8s} {'random':>7s} {'no_cot':>7s}"
    )
    print(header)
    print("-" * len(header))

    for sel, rate in combo_keys:
        s = stats[(sel, rate)]
        total = s["total"]

        def _pct(k):
            return f"{s[k]}/{total}" if total else "  n/a"

        print(
            f"{sel:16s} {rate:>6.2f} {total:>6d} "
            f"{_pct('full_correct'):>7s} {_pct('reached_correct'):>8s} "
            f"{_pct('random_correct'):>7s} {_pct('no_cot_correct'):>7s}"
        )

        summary = {
            "total_traces": total,
            "full_sequence_accuracy": s["full_correct"] / total if total else 0,
            "reached_only_accuracy": s["reached_correct"] / total if total else 0,
            "random_only_accuracy": s["random_correct"] / total if total else 0,
            "no_cot_accuracy": s["no_cot_correct"] / total if total else 0,
        }
        if not skip_patched:
            summary["reached_patched_accuracy"] = s["patched_correct"] / total if total else 0
            summary["random_patched_accuracy"] = s["random_patched_correct"] / total if total else 0

        combo_output_file = output_file
        if combo_output_file is None or len(combo_keys) > 1:
            suffix = "_nopatch" if skip_patched else ""
            combo_output_file = str(
                traces_path.with_name(
                    traces_path.stem + f"_eval_{sel}_r{rate}{suffix}.json"
                )
            )

        out = {
            "model": model,
            "traces_file": str(traces_path),
            "selector": sel,
            "retention_rate": rate,
            "skip_patched": skip_patched,
            "seed": seed,
            "truncated_skipped": truncated_skipped,
            "results": results_questions[(sel, rate)],
            "summary": summary,
        }
        with open(combo_output_file, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\ntruncated_skipped: {truncated_skipped}")
    print(f"\nWrote {len(combo_keys)} result file(s) alongside {traces_path}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)