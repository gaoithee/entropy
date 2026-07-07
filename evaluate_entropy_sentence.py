#!/usr/bin/env python3
"""
Sentence-level variant of evaluate_entropy_splice.py.

Instead of scoring/selecting individual tokens inside the thinking region,
this splits the CoT into sentences (using the same end-of-sentence
delimiter logic as the token-level `end_of_sentence` selector) and scores/
selects whole SENTENCES. Reuses model loading, sequence building, boundary
detection, and generation routines from evaluate_entropy_splice.py rather
than duplicating them.

Selectors (sentence-level):
  - high_entropy:          keep sentences with the HIGHEST mean (normalized)
                            entropy, i.e. sum(token_entropy)/len(sentence).
  - low_entropy:            keep sentences with the LOWEST mean entropy.
  - numbers:                keep sentences with the HIGHEST numeric-token
                            density (fraction of tokens containing a digit).
  - low_entropy_no_numbers: keep sentences with the LOWEST mean entropy,
                            computed over only the NON-numeric tokens in the
                            sentence. A sentence with zero non-numeric
                            tokens is excluded from the candidate pool
                            entirely (cannot be scored on this criterion).
  - random:                 uniformly sample whole sentences (seeded).

Budget: same `_budget(start_pos, end_pos, retention_rate)` token-count
budget as the token-level script, so results stay comparable across the two
granularities. Sentences are added greedily in ranked order until the
cumulative token count would exceed the budget (pool-exhaustion semantics
identical in spirit to the token-level numbers/newlines/end_of_sentence
selectors: a very number-poor or short-sentence trace can genuinely run out
of candidates before filling the nominal budget).

Reached-only: concatenate the tokens of the selected sentences, in their
ORIGINAL order (never scrambled), exactly like the token-level invariant.

Patched (two independent variants, both computed per run):
  - pooled:     each selected sentence is represented, per layer, by the
                MEAN of that sentence's token activations. Every position
                the sentence occupies in the reconstructed short sequence is
                patched with this SAME pooled vector.
  - last_token: each selected sentence is represented, per layer, by the
                activation AT ITS LAST TOKEN. Every position the sentence
                occupies in the reconstructed short sequence is patched with
                this SAME last-token vector.
  In both cases all positions belonging to one sentence receive an
  IDENTICAL vector (this is the point of the ablation: does a single
  summary vector per sentence carry as much signal as the token-resolved
  patching in the original script).
"""

import gc
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from entropy.core.utils import check_correct
from evaluate_entropy_splice import (
    get_model_layers,
    greedy_generate,
    collect_residual_stream,
    load_precomputed_full_resid,
    load_lm,
    parse_patch_layers,
    _find_thinking_boundaries,
    _build_sequence,
    _build_no_cot,
    _resolve_suffix_ids,
    _SUFFIX_VARIANTS,
    _DEFAULT_SUFFIX_VARIANT,
    _budget,
    _NUMERIC_RE,
    _sample_random_positions,
    _parse_list_arg,
    _fmt_bool,
)

_VALID_SENTENCE_SELECTORS = (
    "high_entropy", "low_entropy", "numbers", "low_entropy_no_numbers", "random"
)


# ---------------------------------------------------------------------------
# Sentence segmentation (reuses the token-level end_of_sentence criterion,
# but returns ALL boundaries -- unbudgeted -- to actually carve the thinking
# region into contiguous sentence spans, not just a capped selection).
# ---------------------------------------------------------------------------

def _find_all_sentence_end_positions(tokens, start_pos, end_pos, tokenizer) -> List[int]:
    """Positions (within [start_pos, end_pos)) of tokens that themselves
    carry a sentence-ending period, using the exact same criterion as
    evaluate_entropy_splice._reached_end_of_sentence_only (token-pair based,
    matches only '.', not '!'/'?'). Unlike that function this is NOT
    budget-capped: it returns every boundary found, since we need all of
    them to segment the region into sentences.
    """
    decode_end = min(end_pos + 1, len(tokens))
    decoded = {}
    for tok_id in set(tokens[start_pos:decode_end]):
        decoded[tok_id] = tokenizer.decode([tok_id], skip_special_tokens=False)

    matched = set()
    for i in range(start_pos, min(end_pos, len(tokens))):
        text = decoded.get(tokens[i], "")
        if ".\n" in text or ". " in text:
            matched.add(i)
        if text.endswith(".") and i + 1 < len(tokens):
            next_text = decoded.get(tokens[i + 1], "")
            if next_text.startswith("\n") or next_text.startswith(" "):
                matched.add(i)
    return sorted(matched)


def _segment_sentences(tokens, start_pos, end_pos, tokenizer) -> List[Tuple[int, int]]:
    """Split [start_pos, end_pos) into contiguous, non-overlapping sentence
    spans (seg_start, seg_end) with seg_end EXCLUSIVE, using sentence-ending
    boundaries. The boundary token itself is included in the sentence it
    closes. If no boundary is found in the whole region, the region is
    treated as a single sentence (mirrors the token-level selector falling
    back to "no matches" gracefully rather than crashing).
    """
    boundaries = _find_all_sentence_end_positions(tokens, start_pos, end_pos, tokenizer)
    spans = []
    seg_start = start_pos
    for b in boundaries:
        seg_end = b + 1  # boundary token included in this sentence
        if seg_end > seg_start:
            spans.append((seg_start, seg_end))
        seg_start = seg_end
    if seg_start < end_pos:
        spans.append((seg_start, end_pos))
    return spans


# ---------------------------------------------------------------------------
# Per-sentence scoring
# ---------------------------------------------------------------------------

def _sentence_mean_entropy(seg, entropies, exclude_numeric=False, tokens=None, tokenizer=None):
    seg_start, seg_end = seg
    if exclude_numeric:
        assert tokens is not None and tokenizer is not None
        vals = [
            entropies[i] for i in range(seg_start, seg_end)
            if not _NUMERIC_RE.search(tokenizer.decode([tokens[i]]))
        ]
    else:
        vals = entropies[seg_start:seg_end]
    if not vals:
        return None  # signals "not scorable" (e.g. all-numeric sentence under no_numbers)
    return sum(vals) / len(vals)


def _sentence_numeric_density(seg, tokens, tokenizer):
    seg_start, seg_end = seg
    n = seg_end - seg_start
    if n == 0:
        return 0.0
    n_numeric = sum(
        1 for i in range(seg_start, seg_end)
        if _NUMERIC_RE.search(tokenizer.decode([tokens[i]]))
    )
    return n_numeric / n


def _greedy_fill_by_budget(ranked_segs: List[Tuple[int, int]], budget: int) -> List[Tuple[int, int]]:
    """Add whole sentences in ranked (best-first) order until the next
    addition would exceed `budget` tokens. Mirrors the token-level
    selectors' pool-exhaustion semantics: if the trace doesn't have enough
    scorable sentences, fewer tokens than budget are returned rather than
    erroring. Returns the selected spans, re-sorted back into ORIGINAL
    document order (never scrambled) before the caller flattens them.
    """
    selected = []
    used = 0
    for seg in ranked_segs:
        seg_len = seg[1] - seg[0]
        if used + seg_len > budget and selected:
            # already have at least one sentence and the next would overflow
            break
        selected.append(seg)
        used += seg_len
        if used >= budget:
            break
    return sorted(selected, key=lambda s: s[0])


def _sentence_contains_gt_answer(seg, tokens, tokenizer, gt_answer: str) -> bool:
    seg_start, seg_end = seg
    text = tokenizer.decode(tokens[seg_start:seg_end], skip_special_tokens=True)
    return gt_answer in text


def select_sentences(
    selector: str,
    sentences: List[Tuple[int, int]],
    tokens,
    entropies,
    start_pos: int,
    end_pos: int,
    retention_rate: float,
    tokenizer,
    seed: int,
    filter_out: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """Returns the selected sentence spans, in ORIGINAL order.

    filter_out: if given (the ground-truth answer string), any sentence
        whose decoded text literally CONTAINS this string is dropped from
        the candidate pool BEFORE scoring/ranking -- mirrors the token-level
        --filter_gt_answer behavior, but applied at sentence granularity: a
        whole sentence is excluded if the answer leaks anywhere inside it.
        Default None: no filtering.
    """
    budget = _budget(start_pos, end_pos, retention_rate)

    if filter_out is not None:
        sentences = [
            s for s in sentences
            if not _sentence_contains_gt_answer(s, tokens, tokenizer, filter_out)
        ]

    if selector == "random":
        rng = random.Random(seed)
        order = list(range(len(sentences)))
        rng.shuffle(order)
        ranked = [sentences[i] for i in order]
        return _greedy_fill_by_budget(ranked, budget)

    if selector == "numbers":
        scored = [(_sentence_numeric_density(s, tokens, tokenizer), s) for s in sentences]
        scored.sort(key=lambda x: x[0], reverse=True)  # highest density first
        ranked = [s for _, s in scored]
        return _greedy_fill_by_budget(ranked, budget)

    if selector in ("high_entropy", "low_entropy"):
        scored = [(_sentence_mean_entropy(s, entropies), s) for s in sentences]
        scored = [(sc, s) for sc, s in scored if sc is not None]
        reverse = (selector == "high_entropy")
        scored.sort(key=lambda x: x[0], reverse=reverse)
        ranked = [s for _, s in scored]
        return _greedy_fill_by_budget(ranked, budget)

    if selector == "low_entropy_no_numbers":
        scored = [
            (_sentence_mean_entropy(s, entropies, exclude_numeric=True, tokens=tokens, tokenizer=tokenizer), s)
            for s in sentences
        ]
        # sentences with score=None are all-numeric -> not scorable, excluded
        scored = [(sc, s) for sc, s in scored if sc is not None]
        scored.sort(key=lambda x: x[0])  # lowest first
        ranked = [s for _, s in scored]
        return _greedy_fill_by_budget(ranked, budget)

    raise ValueError(f"Unknown sentence selector: {selector!r}, expected one of {_VALID_SENTENCE_SELECTORS}")


def _flatten(spans: List[Tuple[int, int]]) -> List[int]:
    out = []
    for s, e in spans:
        out.extend(range(s, e))
    return out


# ---------------------------------------------------------------------------
# Sentence-level patched generation: one vector per selected sentence,
# broadcast to every position that sentence occupies in the reconstructed
# short sequence.
# ---------------------------------------------------------------------------

def _sentence_summary_vectors(full_resid: torch.Tensor, spans: List[Tuple[int, int]], mode: str) -> List[torch.Tensor]:
    """full_resid: [n_layers_patched, seq_len, hidden_dim] (absolute
    positions, as produced by collect_residual_stream / load_precomputed_full_resid).
    Returns one [n_layers_patched, hidden_dim] tensor per span.
    """
    vectors = []
    for (s, e) in spans:
        if mode == "pooled":
            vec = full_resid[:, s:e, :].mean(dim=1)
        elif mode == "last_token":
            vec = full_resid[:, e - 1, :]
        else:
            raise ValueError(f"Unknown sentence patch mode {mode!r}")
        vectors.append(vec)
    return vectors


def generate_with_sentence_patching(
    lm,
    num_layers: int,
    short_tokens: torch.Tensor,
    full_resid: torch.Tensor,
    selected_spans: List[Tuple[int, int]],
    patch_offset: int,
    tokenizer,
    mode: str,
    max_new_tokens: int = 10,
    layers_to_patch: Optional[List[int]] = None,
) -> str:
    """Same skeleton as evaluate_entropy_splice.generate_with_patching, but
    every token position belonging to sentence k is patched with the SAME
    per-sentence summary vector (pooled or last_token), instead of each
    position getting its own original per-token activation.
    """
    if layers_to_patch is None:
        layers_to_patch = list(range(num_layers))
    layers = get_model_layers(lm)

    sentence_vectors = _sentence_summary_vectors(full_resid, selected_spans, mode)

    # Map each NEW position (0-indexed within the concatenated selection) to
    # the sentence index it belongs to, so we know which summary vector to
    # apply during patching.
    pos_to_sentence = []
    for sent_idx, (s, e) in enumerate(selected_spans):
        pos_to_sentence.extend([sent_idx] * (e - s))

    generated_tokens = []
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            with torch.no_grad():
                with lm.trace(short_tokens, use_cache=True):
                    for local_idx, L in enumerate(layers_to_patch):
                        block_out = layers[L].output
                        hidden = block_out[0] if isinstance(block_out, tuple) else block_out
                        for i, sent_idx in enumerate(pos_to_sentence):
                            new_pos = patch_offset + i
                            hidden[:, new_pos, :] = sentence_vectors[sent_idx][local_idx].to(hidden.device)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    tail_len: int = 0,
    debug: bool = False,
    suffix_variant: str = _DEFAULT_SUFFIX_VARIANT,
    pool_exhaustion_warn_rate_threshold: float = 0.5,
    filter_gt_answer: bool = False,
    activations_pth_dir: Optional[str] = None,
    patch_layers: Optional[str] = None,
    attn_implementation: Optional[str] = None,
):
    rates = _parse_list_arg(retention_rate, cast=float)
    selectors = _parse_list_arg(selector, cast=str)
    for sel in selectors:
        if sel not in _VALID_SENTENCE_SELECTORS:
            raise ValueError(f"--selector must be one of {_VALID_SENTENCE_SELECTORS}, got {sel!r}")

    traces_path = Path(traces_file)
    with open(traces_path) as f:
        questions = json.load(f)
    if question_offset:
        questions = questions[question_offset:]
    if max_questions is not None:
        questions = questions[:max_questions]

    print(f"Loading model: {model}")
    lm, tokenizer, config = load_lm(model, attn_implementation=attn_implementation)
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

    suffix_ids = _resolve_suffix_ids(tokenizer, suffix_variant)
    suffix_text = _SUFFIX_VARIANTS[suffix_variant]
    print(f"Suffix variant: {suffix_variant!r} -> {suffix_text!r} ({len(suffix_ids)} tokens)")

    num_layers = config["num_hidden_layers"]
    print(f"Model: {num_layers} layers")

    layers_to_patch = parse_patch_layers(patch_layers, num_layers) if not skip_patched else None
    if layers_to_patch is not None:
        print(f"Patch layers: {len(layers_to_patch)}/{num_layers} -- {layers_to_patch}")

    PATCH_MODES = ("pooled", "last_token")
    print(f"Sweep: {len(selectors)} selector(s) x {len(rates)} retention_rate(s) "
          f"= {len(selectors) * len(rates)} combo(s). skip_patched={skip_patched} "
          f"patch_modes={PATCH_MODES if not skip_patched else '-'}")

    combo_keys = [(sel, r) for sel in selectors for r in rates]
    results_questions = {k: [] for k in combo_keys}
    stats = {
        k: {
            "full_correct": 0, "reached_correct": 0, "random_correct": 0, "no_cot_correct": 0,
            "pooled_correct": 0, "last_token_correct": 0,
            "random_pooled_correct": 0, "random_last_token_correct": 0,
            "total": 0, "pool_exhausted_high_rate": 0,
        }
        for k in combo_keys
    }
    truncated_skipped = 0
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
                break

            full_ids_list = prompt_tokens + trace_tokens
            full_entropy = [0.0] * len(prompt_tokens) + list(trace_entropy)

            boundaries = _find_thinking_boundaries(trace_tokens, start_ids, end_ids)
            if boundaries is None:
                print(f"  Q{q_idx} Trace {t_idx}: no end_thinking boundary "
                      f"(truncated at {len(trace_tokens)} tokens), skipping (does not count toward budget)")
                truncated_skipped += 1
                continue

            if orig_extracted is not None and not str(orig_extracted).strip():
                print(f"  Q{q_idx} Trace {t_idx}: original extracted_answers empty (no boxed), "
                      f"skipping (does not count toward budget)")
                continue

            t_start, t_end = boundaries
            start_pos = len(prompt_tokens) + t_start
            end_pos = len(prompt_tokens) + t_end

            tail_start = t_end + len(end_ids)
            tail_ids = trace_tokens[tail_start:tail_start + tail_len] if tail_len > 0 else []

            sentences = _segment_sentences(full_ids_list, start_pos, end_pos, tokenizer)
            if debug:
                print(f"    [DEBUG] segmented {len(sentences)} sentence(s) in thinking region "
                      f"[{start_pos}, {end_pos})")

            trace_seed_base = seed + q_idx * 1000 + t_idx
            print(f"  Q{q_idx} T{t_idx}: {end_pos - start_pos} thinking tokens, {len(sentences)} sentences, "
                  f"starting generations... "
                  f"({n_good_traces_this_question}/{max_traces if max_traces is not None else '?'} "
                  f"good traces so far for this question)")

            try:
                full_seq_ids = torch.tensor(
                    [_build_sequence(prompt_tokens, start_ids, trace_tokens[t_start:t_end], end_ids, tail_ids, suffix_ids)],
                    device=device,
                )
                gen_ids = greedy_generate(lm, full_seq_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                full_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                full_correct = check_correct(suffix_text + full_gen_text, gt_answer)

                no_cot_ids = torch.tensor(
                    [_build_no_cot(prompt_tokens, start_ids, end_ids, tail_ids, suffix_ids)], device=device
                )
                gen_ids = greedy_generate(lm, no_cot_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                no_cot_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                no_cot_correct = check_correct(suffix_text + no_cot_gen_text, gt_answer)

                full_resid = None
                if not skip_patched:
                    if activations_pth_dir is not None:
                        full_resid = load_precomputed_full_resid(
                            activations_pth_dir, q_idx, t_idx, trace_tokens,
                            prompt_len=len(prompt_tokens), num_layers=num_layers,
                            device=device, debug=debug, layers_to_patch=layers_to_patch,
                        )
                    if full_resid is None:
                        full_ids = torch.tensor([full_ids_list], device=device)
                        full_resid = collect_residual_stream(
                            lm, num_layers, full_ids, layers_to_collect=layers_to_patch
                        )

                random_cache = {}
                for rate in rates:
                    budget = _budget(start_pos, end_pos, rate)
                    random_spans = select_sentences(
                        "random", sentences, full_ids_list, full_entropy,
                        start_pos, end_pos, rate, tokenizer, seed=trace_seed_base,
                        filter_out=(gt_answer if filter_gt_answer else None),
                    )
                    random_positions = _flatten(random_spans)
                    if not random_positions:
                        random_cache[rate] = None
                        continue
                    random_ids = torch.tensor(
                        [_build_sequence(
                            prompt_tokens, start_ids,
                            [full_ids_list[p] for p in random_positions],
                            end_ids, tail_ids, suffix_ids,
                        )],
                        device=device,
                    )
                    gen_ids = greedy_generate(lm, random_ids, max_new_tokens=max_new_tokens,
                                               eos_token_id=tokenizer.eos_token_id)
                    random_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    random_correct = check_correct(suffix_text + random_gen_text, gt_answer)

                    random_pooled_correct = None
                    random_last_token_correct = None
                    if not skip_patched:
                        for mode, key_name in (("pooled", "random_pooled_correct"), ("last_token", "random_last_token_correct")):
                            gen_text = generate_with_sentence_patching(
                                lm, num_layers, random_ids, full_resid, random_spans,
                                patch_offset=len(prompt_tokens) + patch_base_offset_extra,
                                tokenizer=tokenizer, mode=mode,
                                max_new_tokens=max_new_tokens, layers_to_patch=layers_to_patch,
                            )
                            correct = check_correct(suffix_text + gen_text, gt_answer)
                            if mode == "pooled":
                                random_pooled_correct = correct
                            else:
                                random_last_token_correct = correct

                    random_cache[rate] = {
                        "spans": random_spans,
                        "gen_text": random_gen_text,
                        "correct": random_correct,
                        "pooled_correct": random_pooled_correct,
                        "last_token_correct": random_last_token_correct,
                    }

                for sel in selectors:
                    for rate in rates:
                        key = (sel, rate)
                        budget_for_sel = _budget(start_pos, end_pos, rate)
                        selected_spans = select_sentences(
                            sel, sentences, full_ids_list, full_entropy,
                            start_pos, end_pos, rate, tokenizer, seed=trace_seed_base,
                            filter_out=(gt_answer if filter_gt_answer else None),
                        )
                        reached_positions = _flatten(selected_spans)

                        n_selected_tokens = len(reached_positions)
                        if n_selected_tokens < budget_for_sel:
                            stats[key]["pool_exhausted_high_rate"] += 1
                            if rate >= pool_exhaustion_warn_rate_threshold:
                                print(
                                    f"  Q{q_idx} T{t_idx}: [WARNING] selector={sel} rate={rate:.2f}: "
                                    f"pool exhausted -- only {n_selected_tokens}/{budget_for_sel} "
                                    f"tokens available across {len(selected_spans)} sentence(s). "
                                    f"Use n_tokens_reached/n_tokens_thinking, not retention_rate, "
                                    f"for actual compression on this trace."
                                )

                        if not reached_positions:
                            if debug:
                                print(f"    [DEBUG] selector={sel} rate={rate:.2f}: 0 sentences matched -- "
                                      f"SKIPPING this trace for this combo")
                            continue

                        assert reached_positions == sorted(reached_positions), (
                            f"selector={sel} returned out-of-order positions -- "
                            f"sentence spans must be flattened in original order"
                        )

                        short_ids = torch.tensor(
                            [_build_sequence(
                                prompt_tokens, start_ids,
                                [full_ids_list[p] for p in reached_positions],
                                end_ids, tail_ids, suffix_ids,
                            )],
                            device=device,
                        )
                        gen_ids = greedy_generate(lm, short_ids, max_new_tokens=max_new_tokens,
                                                   eos_token_id=tokenizer.eos_token_id)
                        reached_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        reached_correct = check_correct(suffix_text + reached_gen_text, gt_answer)

                        pooled_correct = None
                        last_token_correct = None
                        if not skip_patched:
                            for mode, var in (("pooled", "pooled_correct"), ("last_token", "last_token_correct")):
                                gen_text = generate_with_sentence_patching(
                                    lm, num_layers, short_ids, full_resid, selected_spans,
                                    patch_offset=len(prompt_tokens) + patch_base_offset_extra,
                                    tokenizer=tokenizer, mode=mode,
                                    max_new_tokens=max_new_tokens, layers_to_patch=layers_to_patch,
                                )
                                correct = check_correct(suffix_text + gen_text, gt_answer)
                                if mode == "pooled":
                                    pooled_correct = correct
                                else:
                                    last_token_correct = correct

                        rc = random_cache.get(rate)
                        trace_result = {
                            "trace_index": t_idx,
                            "selector": sel,
                            "retention_rate": rate,
                            "n_sentences_total": len(sentences),
                            "n_sentences_selected": len(selected_spans),
                            "n_tokens_thinking": end_pos - start_pos,
                            "n_tokens_reached": n_selected_tokens,
                            "full_sequence": {"generated_answer": full_gen_text, "correct": full_correct},
                            "reached_only": {"generated_answer": reached_gen_text, "correct": reached_correct},
                            "random_only": {
                                "generated_answer": rc["gen_text"] if rc else None,
                                "correct": rc["correct"] if rc else None,
                            },
                            "no_cot": {"generated_answer": no_cot_gen_text, "correct": no_cot_correct},
                        }
                        if not skip_patched:
                            trace_result["reached_pooled"] = {"correct": pooled_correct}
                            trace_result["reached_last_token"] = {"correct": last_token_correct}
                            trace_result["random_pooled"] = {"correct": rc["pooled_correct"] if rc else None}
                            trace_result["random_last_token"] = {"correct": rc["last_token_correct"] if rc else None}

                        s = stats[key]
                        s["total"] += 1
                        if full_correct:
                            s["full_correct"] += 1
                        if reached_correct:
                            s["reached_correct"] += 1
                        if rc and rc["correct"]:
                            s["random_correct"] += 1
                        if no_cot_correct:
                            s["no_cot_correct"] += 1
                        if not skip_patched:
                            if pooled_correct:
                                s["pooled_correct"] += 1
                            if last_token_correct:
                                s["last_token_correct"] += 1
                            if rc and rc["pooled_correct"]:
                                s["random_pooled_correct"] += 1
                            if rc and rc["last_token_correct"]:
                                s["random_last_token_correct"] += 1

                        results_questions[key].append({
                            "question_id": q_idx,
                            "GT_answer": gt_answer,
                            "trace": trace_result,
                        })

                        print(
                            f"  Q{q_idx} T{t_idx} sel={sel:24s} rate={rate:.2f}  "
                            f"full={_fmt_bool(full_correct)}  reached={_fmt_bool(reached_correct)}  "
                            f"pooled={_fmt_bool(pooled_correct) if pooled_correct is not None else '  -  '}  "
                            f"last_tok={_fmt_bool(last_token_correct) if last_token_correct is not None else '  -  '}  "
                            f"no_cot={_fmt_bool(no_cot_correct)}"
                        )

                del full_resid
                n_good_traces_this_question += 1

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                    print(f"  Q{q_idx} Trace {t_idx}: [OOM], skipping (does not count toward budget, trying next trace)")
                else:
                    raise
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if (q_idx + 1) % 5 == 0 or q_idx == len(questions) - 1:
            print(f"  ...processed {q_idx + 1}/{len(questions)} questions")

    print()
    header = (
        f"{'selector':24s} {'rate':>6s} {'total':>6s} {'full':>7s} {'reached':>8s} "
        f"{'pooled':>8s} {'last_tok':>9s} {'random':>7s} {'no_cot':>7s} {'pool_exh':>9s}"
    )
    print(header)
    print("-" * len(header))

    for sel, rate in combo_keys:
        s = stats[(sel, rate)]
        total = s["total"]

        def _pct(k):
            return f"{s[k]}/{total}" if total else "  n/a"

        print(
            f"{sel:24s} {rate:>6.2f} {total:>6d} "
            f"{_pct('full_correct'):>7s} {_pct('reached_correct'):>8s} "
            f"{_pct('pooled_correct'):>8s} {_pct('last_token_correct'):>9s} "
            f"{_pct('random_correct'):>7s} {_pct('no_cot_correct'):>7s} "
            f"{s['pool_exhausted_high_rate']:>9d}"
        )

        summary = {
            "total_traces": total,
            "full_sequence_accuracy": s["full_correct"] / total if total else 0,
            "reached_only_accuracy": s["reached_correct"] / total if total else 0,
            "random_only_accuracy": s["random_correct"] / total if total else 0,
            "no_cot_accuracy": s["no_cot_correct"] / total if total else 0,
            "pool_exhausted_count": s["pool_exhausted_high_rate"],
        }
        if not skip_patched:
            summary["reached_pooled_accuracy"] = s["pooled_correct"] / total if total else 0
            summary["reached_last_token_accuracy"] = s["last_token_correct"] / total if total else 0
            summary["random_pooled_accuracy"] = s["random_pooled_correct"] / total if total else 0
            summary["random_last_token_accuracy"] = s["random_last_token_correct"] / total if total else 0

        suffix = "_nopatch" if skip_patched else ""
        if not skip_patched and patch_layers is not None:
            layers_tag = str(patch_layers).replace(":", "-").replace(",", "_")
            suffix += f"_layers{layers_tag}"
        combo_output_file = str(
            traces_path.with_name(
                traces_path.stem + f"_eval_sentence_{sel}_r{rate}{suffix}.json"
            )
        )

        out = {
            "model": model,
            "traces_file": str(traces_path),
            "granularity": "sentence",
            "selector": sel,
            "retention_rate": rate,
            "skip_patched": skip_patched,
            "seed": seed,
            "suffix_variant": suffix_variant,
            "filter_gt_answer": filter_gt_answer,
            "suffix_text": _SUFFIX_VARIANTS[suffix_variant],
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
