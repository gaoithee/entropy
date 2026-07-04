#!/usr/bin/env python3
"""
Evaluate token-selection strategies (inside CoT) and the no-CoT baseline
on teacher_traces.json.

FIX 2 (this pass): full_sequence / reached_only / random_only / no_cot were
all being truncated right after `end_ids` (the <|end|> that closes the
"analysis" channel), with no re-opening of the "final" channel
(<|start|>assistant<|channel|>final<|message|>) that follows it in the real
teacher trace. Greedy-generating from that truncation point forced the model
to *generate* the channel-switch boilerplate itself, inside the same
max_new_tokens budget as the actual answer -- which is why retention_rate=1.0
("keep the whole CoT") did not reproduce baseline accuracy: the model was
being asked to do something it never had to do in the original trace.

Fix: every sequence-construction site now appends `tail_ids` -- the tokens
that actually followed `end_ids` in the original trace (i.e.
`trace_tokens[t_end:]`, the real channel-switch + any leading response
tokens) -- via a single shared `_build_sequence` helper. This exactly
mirrors what evaluate_attribution.py does when it uses the untouched
`full_ids` for its baseline condition.
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
    if (
        hasattr(m, "model")
        and hasattr(m.model, "language_model")
        and hasattr(m.model.language_model, "layers")
    ):
        return m.model.language_model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        return m.gpt_neox.layers
    raise ValueError("Unknown model architecture")


def parse_patch_layers(patch_layers, num_layers: int) -> List[int]:
    """Parse --patch_layers into a concrete list of layer indices to patch.

    Accepts:
      - None (default): all layers, i.e. list(range(num_layers)) -- identical
        to the behavior before this option existed.
      - "a:b" (slice syntax, end-exclusive like Python slicing): e.g. "18:24"
        -> [18, 19, 20, 21, 22, 23]. Useful for windowed ablations ("does the
        last 25% of layers carry the useful signal").
      - "a,b,c" (explicit comma-separated indices): e.g. "0,6,12,18" for
        sparse/equidistant points -- supported, but see the caveat in the
        module docs: activation patching literature (e.g. ROME-style causal
        tracing) generally finds SPARSE isolated-layer patches weaker/noisier
        than CONTIGUOUS windows, because adjacent untouched layers can route
        around a single patched point. Prefer "a:b" windows for a claim you
        intend to report.
    """
    if patch_layers is None:
        return list(range(num_layers))
    if isinstance(patch_layers, (list, tuple)):
        # Fire's CLI parsing quirk: a bare comma-separated value like
        # "0,4,8,12,16,20" gets auto-coerced into a Python tuple BEFORE it
        # ever reaches this function, even when quoted on the shell side --
        # confirmed by the crash this caused: str((0,4,8,...)) produces
        # "(0, 4, 8, ...)" (with parens and spaces), and splitting that on
        # "," yields fragments like "(0" that int() can't parse. Handle the
        # already-a-sequence case directly instead of assuming a string,
        # exactly like _parse_list_arg does elsewhere in this file for the
        # same underlying Fire behavior (--retention_rate, --selector).
        layers = [int(v) for v in patch_layers]
    else:
        s = str(patch_layers).strip()
        if ":" in s:
            parts = s.split(":")
            if len(parts) != 2:
                raise ValueError(f"--patch_layers slice syntax must be 'a:b', got {patch_layers!r}")
            start = int(parts[0]) if parts[0] != "" else 0
            end = int(parts[1]) if parts[1] != "" else num_layers
            layers = list(range(start, end))
        else:
            layers = [int(v.strip()) for v in s.split(",") if v.strip() != ""]
    invalid = [L for L in layers if L < 0 or L >= num_layers]
    if invalid:
        raise ValueError(f"--patch_layers contains out-of-range indices {invalid} "
                          f"for a {num_layers}-layer model")
    if not layers:
        raise ValueError(f"--patch_layers={patch_layers!r} resolved to an empty layer list")
    return layers


def greedy_generate(
    lm: LanguageModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 10,
    eos_token_id: Optional[int] = None,
) -> List[int]:
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


def collect_residual_stream(
    lm: LanguageModel, num_layers: int, tokens: torch.Tensor, layers_to_collect: Optional[List[int]] = None
) -> torch.Tensor:
    """Collect residual stream activations.

    layers_to_collect: if given, only hook/save these layer indices instead
        of all num_layers -- this is the actual compute/memory saving (the
        forward hook itself is only registered on the layers you care about),
        not just a post-hoc slice of an already-fully-collected tensor. Useful
        when a targeted ablation (e.g. --patch_layers "18:24") has already
        established which layers matter, so future activation collection
        (e.g. for Qwen3/Gemma-4, which don't have precomputed .pth files like
        gpt-oss-20b does) doesn't need to save all layers to disk.
        Default None: collect all layers, identical to behavior before this
        parameter existed.

    Returns a tensor of shape [len(layers_to_collect), seq_len, d_model] --
    NOTE the first dimension indexes into layers_to_collect, NOT into the
    model's absolute layer numbering. Callers that need to map back to
    absolute layer indices (e.g. generate_with_patching) must be given the
    same layers_to_collect list explicitly, not assume range(num_layers).
    """
    layers_to_collect = list(range(num_layers)) if layers_to_collect is None else layers_to_collect
    layers = get_model_layers(lm)
    saves = {}
    with torch.no_grad():
        with lm.trace(tokens):
            for L in layers_to_collect:
                block_out = layers[L].output
                hidden = block_out[0] if isinstance(block_out, tuple) else block_out
                saves[L] = hidden.save()

    out = []
    for L in layers_to_collect:
        val = saves[L]
        if val.dim() == 2:
            val = val.unsqueeze(0)
        val = val[0]
        if out and val.device != out[0].device:
            val = val.to(out[0].device)
        out.append(val)
    return torch.stack(out, dim=0)


def load_precomputed_full_resid(
    pth_dir: str,
    q_idx: int,
    t_idx: int,
    expected_trace_tokens: List[int],
    prompt_len: int,
    num_layers: int,
    device,
    debug: bool = False,
    layers_to_patch: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    """Load precomputed activations from a CompressionBase-style question_XXXX.pth
    file, and reshape them into the [len(layers_to_patch), prompt_len + trace_len,
    hidden_dim] layout that generate_with_patching() expects (first dim aligned
    to layers_to_patch, not to range(num_layers) -- see generate_with_patching
    docstring), so the rest of the patching pipeline needs no further changes
    to consume either a live or precomputed source.

    layers_to_patch: if given, only these absolute layer indices are sliced out
        of the stored (always full-layer, for gpt-oss-20b) activations tensor
        before returning -- e.g. for a windowed ablation ("--patch_layers
        18:24") on data that was collected with ALL layers saved. This does
        NOT reduce the on-disk storage of an already-saved .pth (that
        requires re-collecting with collect_residual_stream's own
        layers_to_collect for NEW data, e.g. Qwen3/Gemma-4 that don't have
        precomputed activations yet) -- it only reduces what's loaded into
        memory and what generate_with_patching ends up touching.
        Default None: use all num_layers, identical to behavior before this
        parameter existed.

    CompressionBase .pth format (question_{q_idx:04d}.pth):
        {"prompt_tokens": [...], "GT_answer": ..., "traces": [
            {"tokens": [...trace tokens only, NOT including prompt...],
             "activations": Tensor[num_trace_tokens, num_layers, hidden_dim],
             "entropies_hf": [...]},
            ...
        ]}

    IMPORTANT: the stored activations are indexed against `trace["tokens"]`
    (trace-only, no prompt prepended) -- NOT against full_ids_list
    (prompt_tokens + trace_tokens) like our own full_resid. This function pads
    the prompt-region with zeros (safe: patching only ever touches positions
    inside the thinking region, i.e. always >= prompt_len, so the zero-filled
    prompt region of the returned tensor is never read) and places the loaded
    activations at the correct absolute offset.

    Returns None (falls back to live collect_residual_stream in the caller) if:
      - the file/trace doesn't exist,
      - the stored trace tokens don't match expected_trace_tokens (different
        temperature/sampling run than the one in teacher_traces.json would
        silently misalign every position index otherwise -- this is checked
        explicitly rather than assumed, since a silent mismatch would corrupt
        every downstream patching result without any visible error),
      - the layer/hidden_dim shape looks inconsistent with what the live model
        would have produced (best-effort check only; a genuine mismatch here
        usually means wrong model/checkpoint).
    """
    layers_to_patch = list(range(num_layers)) if layers_to_patch is None else layers_to_patch
    pth_path = Path(pth_dir) / f"question_{q_idx:04d}.pth"
    if not pth_path.exists():
        if debug:
            print(f"    [DEBUG] precomputed activations: {pth_path} not found, "
                  f"falling back to live collect_residual_stream")
        return None

    try:
        question_data = torch.load(pth_path, weights_only=False)
    except Exception as e:
        print(f"  [WARNING] failed to load {pth_path}: {e} -- falling back to live compute")
        return None

    traces = question_data.get("traces")
    if traces is None or t_idx >= len(traces):
        if debug:
            print(f"    [DEBUG] precomputed activations: trace {t_idx} not found in "
                  f"{pth_path} ({len(traces) if traces else 0} traces available), "
                  f"falling back to live collect_residual_stream")
        return None

    trace = traces[t_idx]
    stored_tokens = trace.get("tokens")
    stored_activations = trace.get("activations")
    if stored_tokens is None or stored_activations is None:
        if debug:
            print(f"    [DEBUG] precomputed activations: trace {t_idx} missing "
                  f"'tokens'/'activations' keys, falling back to live compute")
        return None

    # CRITICAL validation: positions in reached_positions/random_positions are
    # absolute indices into full_ids_list = prompt_tokens + trace_tokens, built
    # from teacher_traces.json. If the .pth trace_tokens don't match EXACTLY
    # (e.g. it came from a different generation run, different seed/sampling),
    # every patched position would silently pull the WRONG activation -- with
    # no crash, no error, just quietly wrong results. Check before trusting it.
    if list(stored_tokens) != list(expected_trace_tokens):
        print(f"  [WARNING] Q{q_idx} T{t_idx}: precomputed .pth trace_tokens do NOT "
              f"match teacher_traces.json trace_tokens for this question/trace "
              f"(len {len(stored_tokens)} vs {len(expected_trace_tokens)}, or content "
              f"differs) -- these are likely from different generation runs. "
              f"Falling back to live collect_residual_stream to avoid silently "
              f"misaligned patching.")
        return None

    # stored_activations: [num_trace_tokens, num_layers, hidden_dim] -- always
    # full-layer for gpt-oss-20b's precomputed dataset, we slice down below.
    if stored_activations.dim() != 3 or stored_activations.shape[1] != num_layers:
        print(f"  [WARNING] Q{q_idx} T{t_idx}: precomputed activations shape "
              f"{tuple(stored_activations.shape)} inconsistent with expected "
              f"num_layers={num_layers} at dim 1 -- falling back to live compute "
              f"(wrong model/checkpoint?)")
        return None

    hidden_dim = stored_activations.shape[2]
    trace_len = stored_activations.shape[0]

    # Slice to just the requested layers BEFORE the permute, so downstream
    # memory usage/copies scale with len(layers_to_patch), not num_layers.
    sliced_activations = stored_activations[:, layers_to_patch, :]

    # Reshape [trace_len, len(layers_to_patch), hidden_dim] -> [len(layers_to_patch), trace_len, hidden_dim]
    trace_resid = sliced_activations.permute(1, 0, 2).to(device)

    # Zero-pad the prompt region -- never read by generate_with_patching since
    # patched positions are always >= prompt_len (inside the thinking region).
    prompt_pad = torch.zeros((len(layers_to_patch), prompt_len, hidden_dim), device=device, dtype=trace_resid.dtype)
    full_resid = torch.cat([prompt_pad, trace_resid], dim=1)

    if debug:
        print(f"    [DEBUG] precomputed activations: loaded from {pth_path}, "
              f"trace_len={trace_len}, patching {len(layers_to_patch)}/{num_layers} layers "
              f"({layers_to_patch}), hidden_dim={hidden_dim} -- SKIPPING live forward pass "
              f"for this trace")

    return full_resid


def generate_with_patching(
    lm: LanguageModel,
    num_layers: int,
    short_tokens: torch.Tensor,
    full_resid: torch.Tensor,
    reached_positions: List[int],
    patch_offset: int,
    tokenizer,
    max_new_tokens: int = 10,
    layers_to_patch: Optional[List[int]] = None,
) -> str:
    """
    layers_to_patch: absolute layer indices (into the model's own layer
        numbering) to patch. If given, ONLY these layers have their block
        output overwritten with full_resid; all other layers compute
        normally from whatever token embedding is at that position (real,
        unpatched forward computation) -- this is a windowed/partial patch,
        not a full residual-stream override.
        `full_resid`'s first dimension must correspond 1:1, in order, to
        `layers_to_patch` (this is exactly what collect_residual_stream /
        load_precomputed_full_resid produce when given the same
        layers_to_patch) -- NOT to range(num_layers). Passing a full_resid
        collected for all layers while patching only a subset would silently
        misalign which stored activation gets applied to which layer.
        Default None: patch all layers (list(range(num_layers))), identical
        to the behavior before this parameter existed.
    """
    if layers_to_patch is None:
        layers_to_patch = list(range(num_layers))
    layers = get_model_layers(lm)
    n_patch = len(reached_positions)
    generated_tokens = []
    past_key_values = None

    for step in range(max_new_tokens):
        if step == 0:
            with torch.no_grad():
                with lm.trace(short_tokens, use_cache=True):
                    for local_idx, L in enumerate(layers_to_patch):
                        block_out = layers[L].output
                        hidden = block_out[0] if isinstance(block_out, tuple) else block_out
                        for i in range(n_patch):
                            new_pos = patch_offset + i
                            orig_pos = reached_positions[i]
                            hidden[:, new_pos, :] = full_resid[local_idx, orig_pos, :].to(hidden.device)
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


def load_lm(model_name: str, quantization: str | None = None, attn_implementation: str | None = None):
    from entropy.core.model_loader import load_model_and_tokenizer

    m = model_name.lower()

    # Auto-select attention implementation per model family when not explicitly
    # given. Rationale (verified empirically on A100/Leonardo, see README):
    #   - gpt-oss: uses attention sinks, which SDPA cannot expose (no access to
    #     pre-softmax logits). The only flash-attn variant that supports sinks
    #     (vllm-flash-attn3) requires Hopper GPUs, unavailable here -> eager.
    #   - everything else: SDPA cuts OOM rate drastically on long traces vs
    #     eager's O(seq_len^2) memory growth, with no known correctness issue
    #     for this codebase (only collects layer *outputs*, never attention
    #     weights). Passing --attn_implementation explicitly always overrides
    #     this default, e.g. for debugging or one-off comparisons.
    if attn_implementation is None:
        attn_implementation = "eager" if "gpt-oss" in m else "sdpa"
        print(f"[attn] auto-selected attn_implementation={attn_implementation!r} for {model_name}")
    else:
        print(f"[attn] using explicit attn_implementation={attn_implementation!r} for {model_name}")

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
# NEW: shared sequence builder (the actual fix)
# ---------------------------------------------------------------------------

def _build_sequence(
    prompt_tokens: List[int],
    start_ids: List[int],
    content_ids: List[int],
    end_ids: List[int],
    tail_ids: List[int],
    suffix_ids: List[int] = None,
) -> List[int]:
    """Build a full input sequence with the channel-switch boilerplate intact.

    prompt + <|start|>...analysis<|message|> + content + <|end|> + tail + suffix

    `tail_ids` is the slice of the ORIGINAL trace that followed end_ids
    (trace_tokens[t_end:]) -- i.e. the real <|start|>assistant<|channel|>
    final<|message|> marker (and possibly a few leading response tokens if
    trace_offset/truncation logic ever hands us more than just the marker).
    Without this, greedy_generate has to *generate* the channel switch
    itself inside max_new_tokens, which silently eats the token budget
    before the model ever gets to the actual answer -- this was the root
    cause of retention_rate=1.0 not matching baseline accuracy.

    `suffix_ids` is an engineered, dataset/experiment-chosen continuation
    (e.g. "Therefore, the answer is \\boxed{") appended after the tail. This
    is NOT part of the original trace -- it's a forced commitment point,
    deliberately mirroring the pattern used in
    attributions/nnsight_sentence_causal.py (get_answer_suffix). Its purpose
    is methodological, not cosmetic: it prevents the model from continuing
    to reason in the post-thinking free-generation space, which would let
    it silently reconstruct information that was supposed to have been cut
    by the retention_rate/selector -- defeating the point of measuring how
    much the compressed CoT alone carries. See team discussion: this must
    stay in place as a core part of the compression-verification setup, not
    be swapped for a dynamic/unbounded post-thinking budget.
    """
    if suffix_ids is None:
        suffix_ids = []
    return prompt_tokens + start_ids + content_ids + end_ids + tail_ids + suffix_ids


# ---------------------------------------------------------------------------
# Answer-suffix variants ("outside CoT" forced commitment point)
# ---------------------------------------------------------------------------
# All variants force the model to commit to an answer immediately after
# end_thinking (+ tail), instead of letting it keep reasoning in free-form
# text. This is intentional: it's what makes the retention_rate/selector
# experiment actually measure what survives INSIDE the (possibly
# compressed) thinking region, rather than measuring "can the model
# re-derive everything anyway if given enough tokens after the cut".
# Default is 'therefore_boxed' -- do not change the default without team
# agreement (see chat discussion 2026-07-03).

_SUFFIX_VARIANTS = {
    "therefore_boxed": "\n\nTherefore, the answer is \\boxed{",
    "boxed_only": "\n\n\\boxed{",
    "based_only_on_above": "\n\nBased only on the above, the best answer I can determine is \\boxed{",
    "one_sentence_boxed": "\n\nGiven the reasoning above, in one sentence, the answer is \\boxed{",
}

_DEFAULT_SUFFIX_VARIANT = "therefore_boxed"


def _resolve_suffix_ids(tokenizer, suffix_variant: str) -> List[int]:
    if suffix_variant not in _SUFFIX_VARIANTS:
        raise ValueError(
            f"--suffix_variant must be one of {sorted(_SUFFIX_VARIANTS)}, got {suffix_variant!r}"
        )
    suffix_text = _SUFFIX_VARIANTS[suffix_variant]
    return tokenizer.encode(suffix_text, add_special_tokens=False)


# ---------------------------------------------------------------------------
# Selection criteria ("inside CoT") -- unchanged from before
# ---------------------------------------------------------------------------

def _budget(start_pos: int, end_pos: int, retention_rate: float) -> int:
    thinking_len = end_pos - start_pos
    raw = int(retention_rate * thinking_len)
    return max(1, min(raw, thinking_len))


def _reached_from_entropy(entropies, start_pos, end_pos, retention_rate, mode):
    num_peaks = _budget(start_pos, end_pos, retention_rate)
    indexed = [(entropies[i], i) for i in range(start_pos, end_pos)]
    indexed.sort(key=lambda x: x[0], reverse=(mode == "high"))
    return sorted(i for _, i in indexed[:num_peaks])


_NUMERIC_RE = re.compile(r"\d")


def _reached_numbers_only(tokens, start_pos, end_pos, tokenizer, retention_rate, seed=0, filter_out=None):
    budget = _budget(start_pos, end_pos, retention_rate)
    # ALIGNED with CompressionPatching._select_numbers (compression_patching.py):
    # was previously `pattern.fullmatch(r"\s*[+-]?\d[\d.,]*\s*$")`, which required
    # the ENTIRE decoded token to be a bare digit sequence -- tokens like "17_b",
    # "b+7", "(28" (digit + attached math/LaTeX context, common in this dataset)
    # never matched and were silently dropped, leaving only denuded digits with
    # no surrounding structure (confirmed via --debug: reconstructed content was
    # an unreadable digit blob like '17177979799979...'). CompressionPatching's
    # `_select_numbers` uses a much looser `_NUMERIC_RE.search(r"\d")` -- any
    # token containing a digit anywhere qualifies, keeping attached context.
    matched = [
        i for i in range(start_pos, end_pos)
        if _NUMERIC_RE.search(tokenizer.decode([tokens[i]]))
    ]
    if filter_out is not None:
        # Mirrors CompressionPooling's filter_out (compression_pooling.py):
        # drop any matched position whose decoded token literally contains
        # the ground-truth answer string, to avoid the selector trivially
        # leaking the final answer through selection rather than through
        # genuine compressed reasoning. OFF by default (filter_out=None) --
        # only applied when the caller explicitly opts in via
        # --filter_gt_answer, so existing runs/results are unaffected.
        filter_str = str(filter_out)
        matched = [i for i in matched if filter_str not in tokenizer.decode([tokens[i]])]
    if len(matched) > budget:
        # Random subsample among the matches, not the first `budget` in
        # position order -- taking the first N systematically concentrated
        # selections near the START of the thinking region (since matched
        # is already position-sorted), ignoring any numbers/newlines/
        # sentence-ends later in the CoT regardless of how informative they
        # were. Random sampling preserves the pattern's natural spread.
        rng = random.Random(seed)
        matched = sorted(rng.sample(matched, budget))
    return matched


def _reached_newlines_only(tokens, start_pos, end_pos, tokenizer, retention_rate, seed=0, filter_out=None):
    budget = _budget(start_pos, end_pos, retention_rate)
    # ALIGNED with CompressionPatching._select_newline: criterion was already
    # equivalent ("\n" in decoded token text), aligning implementation exactly
    # (precomputed set of newline-containing token ids over the full `tokens`
    # vocabulary in range, not per-position redecoding) for parity.
    newline_token_ids = {
        tok for tok in tokens
        if "\n" in tokenizer.decode([tok], skip_special_tokens=False)
    }
    matched = [
        i for i in range(start_pos, end_pos)
        if tokens[i] in newline_token_ids
    ]
    if filter_out is not None:
        filter_str = str(filter_out)
        matched = [i for i in matched if filter_str not in tokenizer.decode([tokens[i]])]
    if len(matched) > budget:
        rng = random.Random(seed)
        matched = sorted(rng.sample(matched, budget))
    return matched


def _reached_end_of_sentence_only(tokens, start_pos, end_pos, tokenizer, retention_rate, seed=0, filter_out=None):
    budget = _budget(start_pos, end_pos, retention_rate)
    # ALIGNED with CompressionPatching._select_end_of_sentence -- this is a
    # real semantic change, not just an implementation detail:
    # - Previously: built a cumulative running text over the whole thinking
    #   region and matched `[.!?]\s` (period/exclaim/question + whitespace),
    #   then mapped each match to the first token whose char-start was AT OR
    #   AFTER the boundary -- which is actually the token that STARTS the
    #   NEXT sentence, not the one ending the current one.
    # - Now: token-pair based, matches ONLY '.' (not '!'/'?', matching their
    #   scope exactly), and selects the token that itself CONTAINS the
    #   sentence-ending period (either ".\n"/". " within one token, or a
    #   token ending in "." whose immediate next token starts with "\n"/" ").
    #   This keeps the period-bearing token itself in the compressed CoT,
    #   which is the token that actually carries the "sentence just ended"
    #   signal, rather than an arbitrary token from the following sentence.
    decode_end = min(end_pos + 1, len(tokens))
    decoded: dict = {}
    for tok_id in set(tokens[start_pos:decode_end]):
        decoded[tok_id] = tokenizer.decode([tok_id], skip_special_tokens=False)

    matched_set = set()
    for i in range(start_pos, min(end_pos, len(tokens))):
        text = decoded.get(tokens[i], "")
        if ".\n" in text or ". " in text:
            matched_set.add(i)
        if text.endswith(".") and i + 1 < len(tokens):
            next_text = decoded.get(tokens[i + 1], "")
            if next_text.startswith("\n") or next_text.startswith(" "):
                matched_set.add(i)

    matched = sorted(matched_set)
    if filter_out is not None:
        filter_str = str(filter_out)
        matched = [i for i in matched if filter_str not in tokenizer.decode([tokens[i]])]
    if len(matched) > budget:
        rng = random.Random(seed)
        matched = sorted(rng.sample(matched, budget))
    return matched


def _sample_random_positions(lo: int, hi: int, n_sample: int, exclude: List[int], seed: int) -> List[int]:
    rng = random.Random(seed)
    candidates = [i for i in range(lo, hi) if i not in set(exclude)]
    if len(candidates) <= n_sample:
        return sorted(candidates)
    return sorted(rng.sample(candidates, n_sample))


_VALID_SELECTORS = ("low_entropy", "high_entropy", "numbers", "newlines", "end_of_sentence", "random")


def select_positions(selector, tokens, entropies, start_pos, end_pos, retention_rate, tokenizer, seed, filter_out=None):
    if selector == "low_entropy":
        return _reached_from_entropy(entropies, start_pos, end_pos, retention_rate, "low")
    elif selector == "high_entropy":
        return _reached_from_entropy(entropies, start_pos, end_pos, retention_rate, "high")
    elif selector == "numbers":
        return _reached_numbers_only(tokens, start_pos, end_pos, tokenizer, retention_rate, seed=seed, filter_out=filter_out)
    elif selector == "newlines":
        return _reached_newlines_only(tokens, start_pos, end_pos, tokenizer, retention_rate, seed=seed, filter_out=filter_out)
    elif selector == "end_of_sentence":
        return _reached_end_of_sentence_only(tokens, start_pos, end_pos, tokenizer, retention_rate, seed=seed, filter_out=filter_out)
    elif selector == "random":
        budget = _budget(start_pos, end_pos, retention_rate)
        return _sample_random_positions(start_pos, end_pos, budget, exclude=[], seed=seed)
    else:
        raise ValueError(f"Unknown selector: {selector!r}, expected one of {_VALID_SELECTORS}")


# ---------------------------------------------------------------------------
# No-CoT baseline ("outside CoT") -- FIXED to include tail_ids
# ---------------------------------------------------------------------------

def _build_no_cot(
    prompt_tokens: List[int],
    start_ids: List[int],
    end_ids: List[int],
    tail_ids: List[int],
    suffix_ids: List[int] = None,
) -> List[int]:
    """prompt + start_think + end_think + tail (channel switch) + suffix -- zero thinking content."""
    return _build_sequence(prompt_tokens, start_ids, [], end_ids, tail_ids, suffix_ids)


_COL_TRACE = 10
_COL_INFO = 48


def _fmt_bool(v: bool) -> str:
    return " True" if v else "False"


def _parse_list_arg(value, cast=float) -> List:
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
    tail_len: int = 0,
    debug: bool = False,
    suffix_variant: str = _DEFAULT_SUFFIX_VARIANT,
    pool_exhaustion_warn_rate_threshold: float = 0.5,
    filter_gt_answer: bool = False,
    activations_pth_dir: Optional[str] = None,
    patch_layers: Optional[str] = None,
    attn_implementation: Optional[str] = None,
):
    """
    suffix_variant: which forced-commitment suffix to append after
        end_thinking (+ tail), before letting the model generate its
        answer. One of: therefore_boxed, boxed_only, based_only_on_above,
        one_sentence_boxed (see _SUFFIX_VARIANTS for exact text). Default is
        'therefore_boxed' -- keep this as the default; it's a deliberate
        methodological choice (forces the model to answer using only what
        survived inside the thinking region, instead of letting it keep
        reasoning in the post-thinking free-generation space and silently
        reconstruct information that retention_rate/selector were supposed
        to have cut). See _SUFFIX_VARIANTS for the other options, meant for
        ablation, not for casual switching.
    tail_len: IMPORTANT -- for gpt-oss (and likely other harmony-format
        models), `end_ids` returned by entropy.models.registry.get_thinking_tokens
        already bundles the FULL channel-switch marker, e.g.
        '<|end|><|start|>assistant<|channel|>final<|message|>' as a single
        end_ids sequence -- confirmed via --debug: decoding end_ids for
        openai/gpt-oss-20b shows exactly this. That means the original
        `... + end_ids` construction (without any tail) was ALREADY correct
        for this model: no extra tail is needed, since end_ids itself puts
        the model right at the start of the "final" channel, ready to
        generate the answer.
        Appending a tail on top of that (tail_len > 0) grabs tokens that
        come AFTER end_ids in the original trace -- which at that point is
        the actual answer content (e.g. '\\boxed{basketball}<|return|>').
        That leaks the answer into the model's input, so it immediately
        emits EOS and generates 0 tokens -- exactly the empty-generation bug
        seen when tail_len=8 was used as default.
        Keep tail_len=0 unless you've verified via --debug that, for your
        specific model/registry config, end_ids does NOT already include a
        channel-switch marker (i.e. decoded end_ids is just something like
        '<|end|>' with nothing else) -- only then does a nonzero tail make
        sense.
    activations_pth_dir: optional path to a directory of CompressionBase-style
        question_XXXX.pth files (the same format used by
        compression_patching.py/compression_pooling.py: {"prompt_tokens":...,
        "GT_answer":..., "traces": [{"tokens":..., "activations":
        Tensor[trace_len, num_layers, hidden_dim], "entropies_hf":...}]}).
        Only used when --skip_patched False. If provided, patching sources
        activations from these precomputed files instead of running a live
        forward pass (collect_residual_stream) -- useful when you already
        have activations collected separately and want to avoid recomputing
        them. Falls back automatically (with a printed warning) to the live
        forward pass if: the file/trace doesn't exist, or the stored
        trace_tokens don't match the trace_tokens from --traces_file (e.g.
        a different generation run) -- this fallback is silent-safe by
        design, never silently patches from mismatched activations.
        Default None: behaves exactly as before this option existed (always
        live forward pass when --skip_patched False).
    patch_layers: which layers to patch, only used when --skip_patched False.
        See parse_patch_layers() for accepted syntax ("18:24" windowed slice,
        "0,6,12" explicit list, or None for all layers). Windows are
        preferred over sparse points for anything you intend to report --
        causal-tracing literature (e.g. ROME-style) generally finds isolated
        single-layer patches weaker/noisier than contiguous windows, since
        adjacent untouched layers can route information around a single
        patched point. This also cuts memory/compute proportionally when
        combined with fresh activation collection (not just when slicing an
        already-saved full-layer .pth) -- collect_residual_stream only hooks
        the requested layers, so for models without precomputed activations
        (e.g. Qwen3, Gemma-4) this is a real cost reduction, not just a
        post-hoc slice.
        Default None: patch all layers, identical to behavior before this
        option existed.
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
    print(f"Suffix variant: {suffix_variant!r} -> {suffix_text!r} "
          f"({len(suffix_ids)} tokens)")

    num_layers = config["num_hidden_layers"]
    print(f"Model: {num_layers} layers")

    layers_to_patch = parse_patch_layers(patch_layers, num_layers) if not skip_patched else None
    if layers_to_patch is not None:
        print(f"Patch layers: {len(layers_to_patch)}/{num_layers} -- {layers_to_patch}")

    print(f"Sweep: {len(selectors)} selector(s) x {len(rates)} retention_rate(s) "
          f"= {len(selectors) * len(rates)} combo(s). skip_patched={skip_patched}")

    combo_keys = [(sel, r) for sel in selectors for r in rates]
    results_questions = {k: [] for k in combo_keys}
    stats = {
        k: {
            "full_correct": 0, "reached_correct": 0, "patched_correct": 0,
            "random_correct": 0, "random_patched_correct": 0, "no_cot_correct": 0,
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

            # NEW: real continuation from the trace, right after end_thinking.
            # This is the channel-switch marker (+ maybe a few leading answer
            # tokens) that was previously being dropped.
            # NOTE: t_end is the position where end_ids *begins* (see
            # _find_thinking_boundaries), so we must skip past end_ids itself
            # before slicing the tail -- otherwise end_ids gets duplicated
            # (once from _build_sequence's own `+ end_ids`, once again here).
            tail_start = t_end + len(end_ids)
            tail_ids = trace_tokens[tail_start:tail_start + tail_len] if tail_len > 0 else []

            teacher_answer_len = len(trace_tokens) - tail_start

            if tail_len > 0 and debug:
                decoded_end = tokenizer.decode(end_ids)
                if "<|message|>" in decoded_end or "channel" in decoded_end:
                    print(f"    [DEBUG] WARNING: end_ids already decodes to a full channel-switch "
                          f"({decoded_end!r}) -- a nonzero tail_len={tail_len} will likely leak the "
                          f"real answer content into the input. Consider --tail_len 0.")

            if debug:
                print(f"    [DEBUG] gt_answer = {gt_answer!r}")
                print(f"    [DEBUG] orig_extracted (teacher's own answer at gen time) = {orig_extracted!r}")
                print(f"    [DEBUG] len(trace_tokens) = {len(trace_tokens)}, t_start={t_start}, t_end={t_end}")
                print(f"    [DEBUG] start_ids = {start_ids} -> decoded: {tokenizer.decode(start_ids)!r}")
                print(f"    [DEBUG] end_ids   = {end_ids} -> decoded: {tokenizer.decode(end_ids)!r}")
                print(f"    [DEBUG] tail_ids ({len(tail_ids)} tokens) = {tail_ids}")
                print(f"    [DEBUG] tail_ids decoded: {tokenizer.decode(tail_ids)!r}")
                print(f"    [DEBUG] teacher's own answer took {teacher_answer_len} tokens "
                      f"(from end of thinking to end of trace) -- "
                      f"max_new_tokens={max_new_tokens} is "
                      f"{'SUFFICIENT' if max_new_tokens >= teacher_answer_len else 'LIKELY TOO LOW'} "
                      f"for this trace")
                # sanity: cosa c'era DAVVERO nel trace originale subito dopo end_ids,
                # con una finestra più larga di tail_len, per capire se tail_len basta
                wide_tail = trace_tokens[tail_start:tail_start + 30]
                print(f"    [DEBUG] wide window (30 tok) after end_ids, decoded: "
                      f"{tokenizer.decode(wide_tail)!r}")

            trace_seed_base = seed + q_idx * 1000 + t_idx
            print(f"  Q{q_idx} T{t_idx}: {end_pos - start_pos} thinking tokens, starting generations... "
                  f"({n_good_traces_this_question}/{max_traces if max_traces is not None else '?'} "
                  f"good traces so far for this question)")

            try:
                # ---- Conditions independent of (selector, rate) ----
                full_seq_ids = torch.tensor(
                    [_build_sequence(prompt_tokens, start_ids, trace_tokens[t_start:t_end], end_ids, tail_ids, suffix_ids)],
                    device=device,
                )
                gen_ids = greedy_generate(lm, full_seq_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                full_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                full_correct = check_correct(suffix_text + full_gen_text, gt_answer)

                # UNIVERSAL sanity check (not gated by --debug): an empty
                # generation on the full_sequence condition -- especially at
                # retention_rate=1.0 -- is the exact symptom of the
                # answer-leaked-into-input bug found on gpt-oss-20b (end_ids
                # already containing the full channel-switch marker, plus a
                # nonzero tail duplicating the real answer into the prompt).
                # Different models (Qwen3, Gemma-4) may structure end_ids
                # differently, so re-check this on every new model/config.
                if len(gen_ids) == 0:
                    print(f"  Q{q_idx} T{t_idx}: [WARNING] full_sequence generated 0 tokens "
                          f"(immediate EOS) -- likely the answer is already leaked into the "
                          f"input via end_ids/tail_ids for this model. Run with --debug True "
                          f"and check 'wide window after end_ids' vs 'end_ids decoded'.")

                if debug:
                    print(f"    [DEBUG] full_sequence n_input_tokens = {full_seq_ids.shape[1]}, "
                          f"n_generated = {len(gen_ids)} (max_new_tokens={max_new_tokens})")
                    print(f"    [DEBUG] full_sequence gen_ids (raw, incl special) = {gen_ids}")
                    print(f"    [DEBUG] full_sequence generated_answer = {full_gen_text!r}")
                    print(f"    [DEBUG] full_sequence correct = {full_correct}  "
                          f"(checking against suffix_text + generated = {suffix_text + full_gen_text!r})")
                    if len(gen_ids) == max_new_tokens:
                        print(f"    [DEBUG] WARNING: hit max_new_tokens budget without EOS -- "
                              f"answer may be truncated before completion, try raising max_new_tokens")

                no_cot_ids = torch.tensor(
                    [_build_no_cot(prompt_tokens, start_ids, end_ids, tail_ids, suffix_ids)], device=device
                )
                gen_ids = greedy_generate(lm, no_cot_ids, max_new_tokens=max_new_tokens,
                                           eos_token_id=tokenizer.eos_token_id)
                no_cot_gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                no_cot_correct = check_correct(suffix_text + no_cot_gen_text, gt_answer)

                if debug:
                    print(f"    [DEBUG] no_cot generated_answer = {no_cot_gen_text!r}, correct={no_cot_correct}")

                full_resid = None
                if not skip_patched:
                    if activations_pth_dir is not None:
                        full_resid = load_precomputed_full_resid(
                            activations_pth_dir, q_idx, t_idx, trace_tokens,
                            prompt_len=len(prompt_tokens), num_layers=num_layers,
                            device=device, debug=debug, layers_to_patch=layers_to_patch,
                        )
                    if full_resid is None:
                        # Either --activations_pth_dir wasn't given, or the
                        # precomputed file/trace was missing or failed
                        # validation (see load_precomputed_full_resid) -- in
                        # both cases fall back to the original live forward
                        # pass, exactly as before this feature was added.
                        full_ids = torch.tensor([full_ids_list], device=device)
                        full_resid = collect_residual_stream(
                            lm, num_layers, full_ids, layers_to_collect=layers_to_patch
                        )

                random_cache = {}

                for rate in rates:
                    budget = _budget(start_pos, end_pos, rate)
                    random_positions = _sample_random_positions(
                        start_pos, end_pos, budget, exclude=[], seed=trace_seed_base,
                    )
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

                    random_patched_correct = None
                    if not skip_patched:
                        gen_text = generate_with_patching(
                            lm, num_layers, random_ids, full_resid, random_positions,
                            patch_offset=len(prompt_tokens) + patch_base_offset_extra,
                            tokenizer=tokenizer,
                            max_new_tokens=max_new_tokens,
                            layers_to_patch=layers_to_patch,
                        )
                        random_patched_correct = check_correct(suffix_text + gen_text, gt_answer)

                        if debug:
                            print(f"    [DEBUG] rate={rate:.2f}: random_PATCHED "
                                  f"generated_answer = {gen_text!r}, correct={random_patched_correct}")

                    random_cache[rate] = {
                        "positions": random_positions,
                        "gen_text": random_gen_text,
                        "correct": random_correct,
                        "patched_correct": random_patched_correct,
                    }

                for sel in selectors:
                    for rate in rates:
                        key = (sel, rate)
                        budget_for_sel = _budget(start_pos, end_pos, rate)
                        reached_positions = select_positions(
                            sel, full_ids_list, full_entropy, start_pos, end_pos,
                            rate, tokenizer, seed=trace_seed_base,
                            filter_out=(gt_answer if filter_gt_answer else None),
                        )
                        # INVARIANT: the compressed CoT must preserve the ORIGINAL
                        # token order, even though numbers/newlines/end_of_sentence/
                        # random now sample randomly WITHIN their candidate pool.
                        # Randomness applies to WHICH positions are kept, never to
                        # the order they're stitched back together in -- shuffling
                        # would turn "compressed CoT" into "scrambled token bag",
                        # which is a different (and much weirder) experiment.
                        assert reached_positions == sorted(reached_positions), (
                            f"selector={sel} returned out-of-order positions: "
                            f"{reached_positions} -- the reconstructed CoT would not "
                            f"preserve original token order"
                        )

                        # UNCONDITIONAL (not gated by --debug) warning: pool
                        # exhaustion at a HIGH nominal retention_rate is the
                        # most misleading case -- it silently produces a
                        # selector that looks "rate-independent" in the
                        # summary table (identical results at e.g. rate=0.5
                        # and rate=1.0), which is easy to misread as "this
                        # selector doesn't benefit from more budget" instead
                        # of what it actually is: "this trace simply doesn't
                        # have enough numbers/newlines/sentence-ends to fill
                        # even a 50%+ budget". Counted into the final summary
                        # table (see pool_exhausted_high_rate column) so it's
                        # visible without re-running with --debug.
                        if (
                            sel in ("numbers", "newlines", "end_of_sentence")
                            and rate >= pool_exhaustion_warn_rate_threshold
                            and len(reached_positions) < budget_for_sel
                        ):
                            stats[key]["pool_exhausted_high_rate"] += 1
                            print(
                                f"  Q{q_idx} T{t_idx}: [WARNING] selector={sel} rate={rate:.2f} "
                                f"(>= {pool_exhaustion_warn_rate_threshold:.2f} threshold): "
                                f"pool exhausted -- only {len(reached_positions)}/{budget_for_sel} "
                                f"tokens available, not the nominal budget. This selector's "
                                f"'retention_rate' does not reflect actual compression for this "
                                f"trace; use n_tokens_reached/n_tokens_thinking instead."
                            )

                        if debug and sel in ("numbers", "newlines", "end_of_sentence"):
                            # These are pattern-based selectors: unlike low_entropy/random,
                            # they take the FIRST budget-many matches in position order
                            # (see _reached_*_only truncation logic), so at low
                            # retention_rate they systematically sample only from the
                            # START of the thinking region -- never from the middle/end,
                            # regardless of where the informative content actually is.
                            # Also, matched count can fall short of budget entirely if the
                            # trace has few numbers/newlines/sentence-ends, which silently
                            # `continue`s (skips this trace for this combo) below --
                            # making `total` in the final summary NOT directly comparable
                            # across selectors on the same sweep.
                            print(f"    [DEBUG] selector={sel} rate={rate:.2f}: "
                                  f"budget={budget_for_sel}, matched={len(reached_positions)} "
                                  f"({'FULL' if len(reached_positions) >= budget_for_sel else 'UNDER budget -- trace may be skipped below'})")
                            if reached_positions:
                                first_pos_frac = (reached_positions[0] - start_pos) / max(1, end_pos - start_pos)
                                last_pos_frac = (reached_positions[-1] - start_pos) / max(1, end_pos - start_pos)
                                print(f"    [DEBUG]   selected positions span "
                                      f"{first_pos_frac:.2%}-{last_pos_frac:.2%} of the thinking region "
                                      f"(low % = concentrated near the START, as expected for this selector)")

                        if debug and reached_positions:
                            # SHARED across ALL selectors (extended from
                            # numbers/newlines/end_of_sentence-only per user
                            # request), to directly compare how the same
                            # raw-concatenation reconstruction looks for
                            # entropy-based vs pattern-based selection on the
                            # same trace. The model itself never "reads" this
                            # as squashed text -- it receives raw token ID
                            # embeddings in this order -- but the readability
                            # (or lack thereof) of the decoded string is still
                            # informative about how out-of-distribution the
                            # reconstructed sequence is.
                            individual_tokens = [tokenizer.decode([full_ids_list[p]]) for p in reached_positions]
                            n_show = min(40, len(individual_tokens))
                            print(f"    [DEBUG] selector={sel} rate={rate:.2f}: "
                                  f"first {n_show} individually decoded selected tokens: "
                                  f"{individual_tokens[:n_show]!r}")

                            # Leak-risk visibility even when --filter_gt_answer is
                            # OFF (default): how many of the selected tokens contain
                            # the ground-truth answer substring? If this is > 0 while
                            # filter_gt_answer=False, `reached_correct=True` results
                            # for this selector/trace may be trivial leakage rather
                            # than genuine compressed reasoning -- worth knowing
                            # regardless of whether filtering is turned on.
                            n_leak_risk = sum(
                                1 for p in reached_positions
                                if gt_answer in tokenizer.decode([full_ids_list[p]])
                            )
                            if n_leak_risk > 0:
                                print(f"    [DEBUG]   [LEAK RISK] {n_leak_risk}/{len(reached_positions)} "
                                      f"selected tokens literally contain gt_answer={gt_answer!r} "
                                      f"(filter_gt_answer={filter_gt_answer})")

                            # Full (untruncated) reconstructed content -- the previous
                            # preview was capped at 300 chars, which for numbers-heavy
                            # traces can hide most of the actual content.
                            full_reconstructed = tokenizer.decode(
                                [full_ids_list[p] for p in reached_positions],
                                skip_special_tokens=True,
                            )
                            print(f"    [DEBUG]   FULL reconstructed content "
                                  f"({len(full_reconstructed)} chars): {full_reconstructed!r}")

                        if not reached_positions:
                            if debug:
                                print(f"    [DEBUG] selector={sel} rate={rate:.2f}: 0 positions matched -- "
                                      f"SKIPPING this trace for this combo (won't count toward 'total' "
                                      f"in the summary, unlike low_entropy/random which always match)")
                            continue

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

                        if debug:
                            print(f"    [DEBUG] selector={sel} rate={rate:.2f}: reached_only "
                                  f"generated_answer = {reached_gen_text!r}, correct={reached_correct}")

                        patched_correct = None
                        if not skip_patched:
                            gen_text = generate_with_patching(
                                lm, num_layers, short_ids, full_resid, reached_positions,
                                patch_offset=len(prompt_tokens) + patch_base_offset_extra,
                                tokenizer=tokenizer,
                                max_new_tokens=max_new_tokens,
                                layers_to_patch=layers_to_patch,
                            )
                            patched_correct = check_correct(suffix_text + gen_text, gt_answer)

                            if debug:
                                # This was NEVER printed before -- patched_correct was
                                # computed and saved to the output JSON, but with
                                # --skip_patched False you had no way to see the
                                # patched result during the run itself, only
                                # reached_only (surface, unpatched). This is exactly
                                # the number needed to answer "does patching save
                                # numbers/end_of_sentence at high compression".
                                print(f"    [DEBUG] selector={sel} rate={rate:.2f}: reached_PATCHED "
                                      f"generated_answer = {gen_text!r}, correct={patched_correct}")

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

    print()
    if skip_patched:
        header = (
            f"{'selector':16s} {'rate':>6s} {'total':>6s} {'full':>7s} "
            f"{'reached':>8s} {'random':>7s} {'no_cot':>7s} {'pool_exh':>9s}"
        )
    else:
        header = (
            f"{'selector':16s} {'rate':>6s} {'total':>6s} {'full':>7s} "
            f"{'reached':>8s} {'patched':>8s} {'random':>7s} {'rand_pat':>9s} "
            f"{'no_cot':>7s} {'pool_exh':>9s}"
        )
    print(header)
    print("-" * len(header))

    for sel, rate in combo_keys:
        s = stats[(sel, rate)]
        total = s["total"]

        def _pct(k):
            return f"{s[k]}/{total}" if total else "  n/a"

        pool_exh_str = str(s["pool_exhausted_high_rate"]) if sel in ("numbers", "newlines", "end_of_sentence") else "-"
        if skip_patched:
            print(
                f"{sel:16s} {rate:>6.2f} {total:>6d} "
                f"{_pct('full_correct'):>7s} {_pct('reached_correct'):>8s} "
                f"{_pct('random_correct'):>7s} {_pct('no_cot_correct'):>7s} {pool_exh_str:>9s}"
            )
        else:
            print(
                f"{sel:16s} {rate:>6.2f} {total:>6d} "
                f"{_pct('full_correct'):>7s} {_pct('reached_correct'):>8s} "
                f"{_pct('patched_correct'):>8s} {_pct('random_correct'):>7s} "
                f"{_pct('random_patched_correct'):>9s} {_pct('no_cot_correct'):>7s} {pool_exh_str:>9s}"
            )

        summary = {
            "total_traces": total,
            "full_sequence_accuracy": s["full_correct"] / total if total else 0,
            "reached_only_accuracy": s["reached_correct"] / total if total else 0,
            "random_only_accuracy": s["random_correct"] / total if total else 0,
            "no_cot_accuracy": s["no_cot_correct"] / total if total else 0,
            "pool_exhausted_high_rate_count": s["pool_exhausted_high_rate"],
            "pool_exhaustion_warn_rate_threshold": pool_exhaustion_warn_rate_threshold,
        }
        if not skip_patched:
            summary["reached_patched_accuracy"] = s["patched_correct"] / total if total else 0
            summary["random_patched_accuracy"] = s["random_patched_correct"] / total if total else 0

        combo_output_file = output_file
        if combo_output_file is None or len(combo_keys) > 1:
            suffix = "_nopatch" if skip_patched else ""
            if not skip_patched and patch_layers is not None:
                # Without this, sweeping multiple --patch_layers windows on the
                # same selector/rate would silently overwrite the same output
                # file each time -- no error, just quietly lost results.
                layers_tag = str(patch_layers).replace(":", "-").replace(",", "_")
                suffix += f"_layers{layers_tag}"
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