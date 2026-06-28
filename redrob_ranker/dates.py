"""Shared date helpers and the pinned "now" reference for the dataset.

The released pool is a frozen snapshot whose last_active dates top out mid-2026,
so we pin "now" to 2026-06 instead of reading the wall clock. This keeps the
honeypot timeline checks and behavioral-recency scoring deterministic and
reproducible no matter when the ranker runs. Both the feature extractor and the
honeypot detector import these, so the constant and the parser live in exactly
one place.
"""

from __future__ import annotations

# Pinned reference month for the frozen dataset, as an absolute month index.
NOW_MONTHS = 2026 * 12 + 6


def ym(date_str: str) -> int | None:
    """Convert a 'YYYY-MM-DD' string to an absolute month index (year*12 + month)."""
    if not date_str:
        return None
    try:
        parts = date_str.split("-")
        return int(parts[0]) * 12 + int(parts[1])
    except (ValueError, IndexError):
        return None
