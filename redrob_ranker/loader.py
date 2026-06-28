"""Streaming candidate loader. Handles plain or gzipped JSONL, and uses orjson
when available for speed (falls back to the stdlib json module)."""

from __future__ import annotations

import gzip
import io
from typing import Iterator, Dict, Any

try:
    import orjson  # type: ignore

    def _loads(line: bytes) -> Dict[str, Any]:
        return orjson.loads(line)
except Exception:  # pragma: no cover - orjson is optional
    import json

    def _loads(line: bytes) -> Dict[str, Any]:
        return json.loads(line)


def _open_maybe_gzip(path: str) -> io.BufferedReader:
    """Open *path* for binary reading, transparently decompressing gzip."""
    if path.endswith(".gz"):
        return gzip.open(path, "rb")  # type: ignore[return-value]
    fh = open(path, "rb")
    magic = fh.read(2)
    fh.seek(0)
    if magic == b"\x1f\x8b":  # gzip magic even without .gz suffix
        fh.close()
        return gzip.open(path, "rb")  # type: ignore[return-value]
    return fh


def iter_candidates(path: str) -> Iterator[Dict[str, Any]]:
    """Yield candidate records one at a time from a (possibly gzipped) JSONL file."""
    with _open_maybe_gzip(path) as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield _loads(raw)
            except Exception:
                # Skip a malformed line rather than aborting the whole run.
                continue
