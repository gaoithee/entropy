"""General utilities. Ported from neurohike.core."""
from __future__ import annotations
import re


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
