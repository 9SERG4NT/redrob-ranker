"""End-to-end ranking pipeline: load -> featurize -> TF-IDF -> score -> rank -> reason."""

from __future__ import annotations

import csv
import sys
import time
from typing import Dict, List

from . import features as F
from . import reasoning
from . import scoring
from .loader import iter_candidates
from .text import compute_tfidf_scores


def _log(msg: str) -> None:
    print(f"[redrob-ranker] {msg}", file=sys.stderr, flush=True)


def rank_all(candidates_path: str) -> List[Dict]:
    """Featurize and score every candidate; return them fully sorted (best first).

    Each returned feature dict carries its components, gate/modifier breakdown, and a
    normalized ``display_score``. This is the shared core used by both the CSV writer
    and the dashboard (which needs the whole pool, not just the top 100)."""
    t0 = time.time()

    # --- Pass 1: stream + featurize (we never hold the full raw records) -------
    feats: List[Dict] = []
    docs: List[str] = []
    for cand in iter_candidates(candidates_path):
        if not cand.get("candidate_id"):
            continue
        feat = F.extract(cand)
        docs.append(feat.pop("tfidf_doc"))  # move heavy text out of the feature dict
        feats.append(feat)
    _log(f"featurized {len(feats):,} candidates in {time.time()-t0:.1f}s")

    # --- TF-IDF semantic backbone --------------------------------------------
    t1 = time.time()
    tfidf_scores = compute_tfidf_scores(docs)
    del docs
    _log(f"TF-IDF scored in {time.time()-t1:.1f}s")

    # --- Finalize scores ------------------------------------------------------
    for feat, tf in zip(feats, tfidf_scores):
        scoring.finalize(feat, tf)

    # --- Normalize for display, then sort deterministically -------------------
    max_score = max((f["score"] for f in feats), default=1.0) or 1.0
    for f in feats:
        f["display_score"] = round(f["score"] / max_score, 6)

    # Round BEFORE sorting so any ties are ordered by candidate_id ascending,
    # which is exactly what the validator requires for equal scores.
    feats.sort(key=lambda f: (-f["display_score"], f["candidate_id"]))
    _log(f"ranked {len(feats):,} candidates | total {time.time()-t0:.1f}s")
    return feats


def write_submission(feats: List[Dict], out_path: str, top_n: int = 100) -> List[Dict]:
    """Write the top-N ranking CSV from already-ranked feats; return the top-N slice."""
    top = feats[:top_n]
    rows = [
        {
            "candidate_id": feat["candidate_id"],
            "rank": i,
            "score": f"{feat['display_score']:.6f}",
            "reasoning": reasoning.generate(feat, i),
        }
        for i, feat in enumerate(top, start=1)
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
    _log(f"wrote {len(rows)} rows to {out_path}")
    return top


def rank(candidates_path: str, out_path: str, top_n: int = 100) -> List[Dict]:
    """Convenience end-to-end: rank everything and write the top-N CSV."""
    feats = rank_all(candidates_path)
    return write_submission(feats, out_path, top_n=top_n)
