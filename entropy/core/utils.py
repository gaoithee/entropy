"""General utilities. Ported from neurohike.core."""
from __future__ import annotations

import re
from typing import Any, Optional


def extract_boxed_answer(text: str) -> str:
    r"""Extract the last \boxed{...} answer from a reasoning trace.

    Handles nested braces.  Returns empty string if nothing found.
    """
    pattern = r"\\boxed\{"
    matches = [m.start() for m in re.finditer(pattern, text)]
    if not matches:
        return ""
    start = matches[-1] + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else ""


# ---------------------------------------------------------------------------
# Answer normalization + equivalence checking.
# Ported verbatim from attributions/vllm_sentence_causal.py (the
# _normalize_answer_text / _normalize_mcq_answer / _math_parse_variants /
# _math_verify_with_variants / _verify_answer_equivalence chain), so that
# correctness evaluation here matches the attribution pipeline's criterion
# exactly instead of a raw substring match.
# ---------------------------------------------------------------------------

def _normalize_answer_text(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""

    text = text.strip("$").strip()
    text = re.sub(r"^\\text\{(.+)\}$", r"\1", text).strip()
    if re.fullmatch(r"\(?[A-Za-z]\)?\.?", text):
        text = text.strip("().").upper()
    return re.sub(r"\s+", "", text).casefold()


def _normalize_mcq_answer(answer: Optional[str]) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if not text:
        return ""
    text = text.strip("$").strip()
    text = re.sub(r"^\\text\{(.+)\}$", r"\1", text).strip()
    match = re.match(r"^\(?\s*([A-Za-z])\s*\)?\s*(?:[.:)\]]|\s|$)", text)
    if match:
        return match.group(1).casefold()
    return _normalize_answer_text(text)


def _math_parse_variants(answer: str) -> list[str]:
    text = str(answer).strip()
    if not text:
        return []

    normalized = (
        text
        .replace(r"\tfrac", r"\frac")
        .replace(r"\dfrac", r"\frac")
        .replace(r"\left", "")
        .replace(r"\right", "")
    )
    variants = [text, f"${text}$", normalized, f"${normalized}$"]

    seen = set()
    unique = []
    for variant in variants:
        if variant and variant not in seen:
            unique.append(variant)
            seen.add(variant)
    return unique


def _math_verify_with_variants(reference: str, candidate: str) -> dict[str, Any]:
    try:
        from math_verify import parse, verify

        last_error = None
        for ref_variant in _math_parse_variants(reference):
            try:
                reference_parsed = parse(ref_variant)
            except Exception as exc:  # noqa: BLE001 - try the next variant
                last_error = str(exc)
                continue
            if not reference_parsed:
                continue

            for candidate_variant in _math_parse_variants(candidate):
                try:
                    candidate_parsed = parse(candidate_variant)
                except Exception as exc:  # noqa: BLE001 - try the next variant
                    last_error = str(exc)
                    continue
                if candidate_parsed and verify(reference_parsed, candidate_parsed):
                    return {
                        "equivalent": True,
                        "method": "math_verify",
                        "error": None,
                        "reference_variant": ref_variant,
                        "candidate_variant": candidate_variant,
                    }

        return {
            "equivalent": False,
            "method": "math_verify",
            "error": last_error,
            "reference_variant": None,
            "candidate_variant": None,
        }
    except Exception as exc:  # noqa: BLE001 - verifier failures should not stop evaluation
        return {
            "equivalent": False,
            "method": "verification_error",
            "error": str(exc),
            "reference_variant": None,
            "candidate_variant": None,
        }


def verify_answer_equivalence(reference: str, candidate: str, answer_domain: str = "math") -> dict[str, Any]:
    """Full equivalence result (method used, error if any, matched variants).
    answer_domain: 'math' (default), 'mcq', or 'string'.
    """
    if answer_domain == "mcq":
        ref_norm = _normalize_mcq_answer(reference)
        cand_norm = _normalize_mcq_answer(candidate)
        if not cand_norm:
            return {"equivalent": False, "method": "no_guess", "error": None}
        if not ref_norm:
            return {"equivalent": False, "method": "missing_reference", "error": None}
        return {"equivalent": cand_norm == ref_norm, "method": "mcq_exact", "error": None}

    if answer_domain == "string":
        ref_norm = _normalize_answer_text(reference)
        cand_norm = _normalize_answer_text(candidate)
        if not cand_norm:
            return {"equivalent": False, "method": "no_guess", "error": None}
        if not ref_norm:
            return {"equivalent": False, "method": "missing_reference", "error": None}
        return {"equivalent": cand_norm == ref_norm, "method": "normalized_exact", "error": None}

    # math (default)
    ref_norm = _normalize_answer_text(reference)
    cand_norm = _normalize_answer_text(candidate)
    if not cand_norm:
        return {"equivalent": False, "method": "no_guess", "error": None}
    if not ref_norm:
        return {"equivalent": False, "method": "missing_reference", "error": None}
    if cand_norm == ref_norm:
        return {"equivalent": True, "method": "normalized_exact", "error": None}
    return _math_verify_with_variants(reference, candidate)


def check_correct(
    generated: str,
    gt_answer: str,
    is_code: bool = False,
    answer_domain: str = "math",
    already_extracted: bool = False,
) -> bool:
    """Correctness check used throughout entropy/. By default extracts the
    last \\boxed{...} answer from `generated` first, then verifies
    equivalence against `gt_answer` via math_verify (normalized-exact as a
    fast path before falling back to math_verify's variant expansion).

    already_extracted: set True when `generated` is already an extracted
    answer (e.g. from a dataset's precomputed `extracted_answers` field) —
    skips extract_boxed_answer and verifies `generated` directly. Use this
    whenever the caller has extracted answers available; it avoids a
    redundant/lossy re-extraction pass on raw text that may not even
    contain \\boxed{} anymore.

    is_code: math_verify does not evaluate code; kept as a non-empty-string
    placeholder for parity with upstream scripts (real eval deferred to
    EvalPlus, not implemented here).

    answer_domain: 'math' (default), 'mcq', or 'string' — passed through to
    verify_answer_equivalence.
    """
    if is_code:
        return len(generated.strip()) > 0
    extracted = generated if already_extracted else extract_boxed_answer(generated)
    if not extracted:
        return False
    result = verify_answer_equivalence(gt_answer, extracted, answer_domain)
    return bool(result["equivalent"])