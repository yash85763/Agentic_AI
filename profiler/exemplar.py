"""
utils/exemplars.py — Exemplar sampling for column values.

Exemplars are real observed values shown to the LLM so it can recognise
what user queries might be referring to.  We sample strategically to
maximise diversity within a fixed token budget.
"""

from __future__ import annotations

import random
from typing import Any

EXEMPLAR_LIMIT = 25  # max unique values stored verbatim


def sample_exemplars(unique_values: list[Any], limit: int = EXEMPLAR_LIMIT) -> tuple[list[str], bool]:
    """
    Given a list of unique values from a column, return a representative
    sample capped at `limit`, plus a bool indicating whether truncation occurred.

    Sampling strategy (when values exceed limit):
      - First 10 (sorted) — covers low-end / early alphabetical values
      - Last 5  (sorted) — covers high-end values
      - 10 random from middle — covers the distribution interior

    Returns:
        (exemplars: list[str], truncated: bool)
    """
    str_vals = [str(v) for v in unique_values if v is not None and str(v).strip() != ""]

    # Sort for determinism
    try:
        sorted_vals = sorted(str_vals)
    except TypeError:
        sorted_vals = sorted(str_vals, key=str)

    if len(sorted_vals) <= limit:
        return sorted_vals, False

    # Truncated sampling
    head = sorted_vals[:10]
    tail = sorted_vals[-5:]
    middle = sorted_vals[10:-5]

    if len(middle) > 10:
        random.seed(42)
        middle_sample = random.sample(middle, 10)
    else:
        middle_sample = middle

    sampled = head + middle_sample + tail
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for v in sampled:
        if v not in seen:
            seen.add(v)
            deduped.append(v)

    return deduped, True
