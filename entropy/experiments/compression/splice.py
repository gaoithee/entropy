#!/usr/bin/env python3
"""
Evaluate entropy-based token selection quality on teacher_traces.json,
mirroring evaluate_attribution.py but with entropy (vLLM, traces_entropy)
instead of GIM scores, and no .pth/activation-collection dependency.

Five conditions per trace, same as evaluate_attribution.py:
  1. full_sequence    — prompt + full thinking + end_thinking (baseline)
  2. reached_only     — prompt + entropy-selected thinking tokens + end_thinking
  3. reached_patched  — same tokens, residual stream patched from full run
  4. random_only      — prompt + random thinking tokens (same count) + end_thinking
  5. random_patched   — random tokens, patched

Usage:
    python evaluate_entropy_splice.py \
        --model openai/gpt-oss-20b \
        --traces_file data/zebralogic/gpt-oss-20b_teacher_traces.json \
        --retention_rate 0.1 \
        --entropy_mode low
"""

import gc
import json
import random
from pathlib import Path
from typing import List, Optional

import torch
from nnsight import LanguageModel


# ---------------------------------------------------------------------------
# Model / layer access (mirrors CompressionBase._get_model_layers)
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
    """Plain greedy generation, no patching. Mirrors nnsight_ifr.greedy_generate."""
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
    `patch_offset` is where the spliced tokens start in `short_tokens` (== len(prompt)).
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


def check_correct(generated: str, gt_answer: str) -> bool:
    return gt_answer.strip().lower() in generated.strip().lower()


# ---------------------------------------------------------------------------
# Entropy-based selection (replaces _reached_from_scores)
# ---------------------------------------------------------------------------

def _find_thinking_boundaries(tokens: List[int], start_ids: List[int], end_ids: List[int]):
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
    return None


def _reached_from_entropy(
    entropies: List[float], start_pos: int, end_pos: int,
    retention_rate: float, mode: str,
) -> List[int]:
    """Select positions within [start_pos, end_pos) by entropy, sorted ascending."""
    thinking_len = end_pos - start_pos
    num_peaks = max(1, int(retention_rate * thinking_len))
    indexed = [(entropies[i], i) for i in range(start_pos, end_pos)]
    indexed.sort(key=lambda x: x[0], reverse=(mode == "high"))
    return sorted(i for _, i in indexed[:num_peaks])


def _sample_random_positions(lo: int, hi: int, n_sample: int, exclude: List[int], seed: int) -> List[int]:
    rng = random.Random(seed)
    candidates = [i for i in range(lo, hi) if i not in set(exclude)]
    if len(candidates) <= n_sample:
        return sorted(candidates)
    return sorted(rng.sample(candidates, n_sample))


# Column widths for aligned output — unchanged from evaluate_attribution.py
_COL_TRACE = 10
_COL_INFO = 48


def _fmt_bool(v: bool) -> str:
    return " True" if v else "False"


def main(
    model: str,
    traces_file: str,
    device: str = "cuda",
    output_file: Optional[str] = None,
    max_new_tokens: int = 10,
    retention_rate: float = 0.1,
    entropy_mode: str = "low",  # "low" or "high"
    seed: int = 42,
    max_questions: Optional[int] = None,
):
    traces_path = Path(traces_file)
    with open(traces_path) as f:
        questions = json.load(f)
    if max_questions is not None:
        questions = questions[:max_questions]

    if output_file is None:
        output_file = str(
            traces_path.with_name(traces_path.stem + f"_eval_entropy_{entropy_mode}_r{retention_rate}.json")
        )

    print(f"Loading model: {model}")
    lm = LanguageModel(
        model, device_map=device, dispatch=True,
        attn_implementation="eager", trust_remote_code=True,
        dtype=torch.bfloat16 if "cuda" in device else torch.float32,
    )
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

    config = lm.config
    num_layers = getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer")
    print(f"Model: {num_layers} layers")
    print(f"Entropy mode: {entropy_mode}, retention_rate: {retention_rate}")

    print()
    header_info = "tokens".ljust(_COL_INFO)
    print(f"  {''.ljust(_COL_TRACE)}{header_info}full   reached  patched  rand     rand_pat")
    print(f"  {''.ljust(_COL_TRACE)}{''.ljust(_COL_INFO)}{'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    results_questions = []
    stats = {
        "full_correct": 0, "reached_correct": 0, "patched_correct": 0,
        "random_correct": 0, "random_patched_correct": 0, "total": 0,
    }

    for q_idx, q in enumerate(questions):
        gt_answer = str(q["GT_answer"])
        prompt_tokens = q["prompt_tokens"]
        results_traces = []

        for t_idx, (trace_tokens, trace_entropy) in enumerate(zip(q["traces_tokens"], q["traces_entropy"])):
            full_ids_list = prompt_tokens + trace_tokens
            # entropies are per-generated-token only; pad with zeros for the
            # prompt region (never selected — boundaries are inside the trace)
            full_entropy = [0.0] * len(prompt_tokens) + list(trace_entropy)

            boundaries = _find_thinking_boundaries(trace_tokens, start_ids, end_ids)
            if boundaries is None:
                print(f"  Trace {t_idx}: no thinking boundaries, skipping")
                continue
            t_start, t_end = boundaries
            # shift into full_ids_list coordinates
            start_pos = len(prompt_tokens) + t_start
            end_pos = len(prompt_tokens) + t_end

            reached_positions = _reached_from_entropy(
                full_entropy, start_pos, end_pos, retention_rate, entropy_mode
            )
            if not reached_positions:
                print(f"  Trace {t_idx}: no reached positions, skipping")
                continue

            random_positions = _sample_random_positions(
                start_pos, end_pos, len(reached_positions), reached_positions,
                seed=seed + q_idx * 1000 + t_idx,
            )

            full_ids = torch.tensor([full_ids_list], device=device)

            # short sequence: prompt + selected tokens + end_thinking (so the
            # model still sees a well-formed Harmony "channel close")
            def _build_short(positions):
                inner = [full_ids_list[p] for p in positions]
                return prompt_tokens + inner + end_ids

            short_ids_list = _build_short(reached_positions)
            short_ids = torch.tensor([short_ids_list], device=device)

            random_ids_list = _build_short(random_positions)
            random_ids = torch.tensor([random_ids_list], device=device)

            reached_token_strs = [tokenizer.decode([full_ids_list[p]]) for p in reached_positions]
            tokens_reached_pipe_separated = "|".join(reached_token_strs)

            trace_result = {
                "trace_index": t_idx,
                "n_tokens_thinking": end_pos - start_pos,
                "n_tokens_reached": len(reached_positions),
                "tokens_reached_pipe_separated": tokens_reached_pipe_separated,
            }

            trace_label = f"Trace {t_idx}:".ljust(_COL_TRACE)
            info = f"{end_pos - start_pos} thinking, {len(reached_positions)} reached".ljust(_COL_INFO)

            try:
                # Condition 1: full_sequence (prompt + full thinking + end_thinking)
                full_seq_ids = torch.tensor(
                    [prompt_tokens + trace_tokens[t_start:t_end] + end_ids], device=device
                )
                gen_ids = greedy_generate(lm, full_seq_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                full_correct = check_correct(gen_text, gt_answer)
                trace_result["full_sequence"] = {"generated_answer": gen_text, "correct": full_correct}
                if full_correct:
                    stats["full_correct"] += 1

                # Condition 2: reached_only
                gen_ids = greedy_generate(lm, short_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                reached_correct = check_correct(gen_text, gt_answer)
                trace_result["reached_only"] = {"generated_answer": gen_text, "correct": reached_correct}
                if reached_correct:
                    stats["reached_correct"] += 1

                # Condition 3: reached_patched
                full_resid = collect_residual_stream(lm, num_layers, full_ids)
                gen_text = generate_with_patching(
                    lm, num_layers, short_ids, full_resid, reached_positions,
                    patch_offset=len(prompt_tokens), tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                )
                patched_correct = check_correct(gen_text, gt_answer)
                trace_result["reached_patched"] = {"generated_answer": gen_text, "correct": patched_correct}
                if patched_correct:
                    stats["patched_correct"] += 1

                # Condition 4: random_only
                gen_ids = greedy_generate(lm, random_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                random_correct = check_correct(gen_text, gt_answer)
                trace_result["random_only"] = {"generated_answer": gen_text, "correct": random_correct}
                if random_correct:
                    stats["random_correct"] += 1

                # Condition 5: random_patched
                gen_text = generate_with_patching(
                    lm, num_layers, random_ids, full_resid, random_positions,
                    patch_offset=len(prompt_tokens), tokenizer=tokenizer,
                    max_new_tokens=max_new_tokens,
                )
                random_patched_correct = check_correct(gen_text, gt_answer)
                trace_result["random_patched"] = {"generated_answer": gen_text, "correct": random_patched_correct}
                if random_patched_correct:
                    stats["random_patched_correct"] += 1

                stats["total"] += 1

                print(f"  {trace_label}{info}"
                      f"{_fmt_bool(full_correct)}  "
                      f"{_fmt_bool(reached_correct):>7}  "
                      f"{_fmt_bool(patched_correct):>7}  "
                      f"{_fmt_bool(random_correct):>7}  "
                      f"{_fmt_bool(random_patched_correct):>7}")

                del full_resid
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"  {trace_label}{info}[OOM]")
                    trace_result["error"] = "OOM"
                else:
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results_traces.append(trace_result)

        results_questions.append({
            "question_id": q_idx,
            "GT_answer": gt_answer,
            "traces": results_traces,
        })

    total = stats["total"]
    output = {
        "model": model,
        "traces_file": str(traces_path),
        "entropy_mode": entropy_mode,
        "retention_rate": retention_rate,
        "seed": seed,
        "results": results_questions,
        "summary": {
            "total_traces": total,
            "full_sequence_accuracy": stats["full_correct"] / total if total else 0,
            "reached_only_accuracy": stats["reached_correct"] / total if total else 0,
            "reached_patched_accuracy": stats["patched_correct"] / total if total else 0,
            "random_only_accuracy": stats["random_correct"] / total if total else 0,
            "random_patched_accuracy": stats["random_patched_correct"] / total if total else 0,
        },
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"  {''.ljust(_COL_TRACE)}{''.ljust(_COL_INFO)}{'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")
    if total:
        def _pct(k): return f"{stats[k]}/{total}"
        print(f"  {'Total:'.ljust(_COL_TRACE)}{''.ljust(_COL_INFO)}"
              f"{_pct('full_correct'):>5}  "
              f"{_pct('reached_correct'):>7}  "
              f"{_pct('patched_correct'):>7}  "
              f"{_pct('random_correct'):>7}  "
              f"{_pct('random_patched_correct'):>7}")

    print(f"\nResults written to {output_file}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)