"""TF-IDF semantic backbone (the "hybrid retrieval" half of the ranker).

We fit a TF-IDF vector space over the candidate corpus and the JD query, then score
each candidate by cosine similarity to the JD. This recovers strong-but-plain-language
candidates whose prose matches the JD's intent even when they avoid buzzwords, and it
provides a smooth, graded relevance signal that complements the discrete rule scores.

If scikit-learn is unavailable, every candidate receives a neutral 0.5 and the ranker
degrades gracefully to lexicon-only narrative scoring.
"""

from __future__ import annotations

from typing import List

from . import config as C


def compute_tfidf_scores(docs: List[str]) -> List[float]:
    """Return per-document cosine similarity to the JD, scaled to [0,1] by rank."""
    if not docs:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel
        import numpy as np
    except Exception:  # pragma: no cover - optional dependency
        return [0.5] * len(docs)

    # The corpus the vectorizer sees includes the JD query. On the full 100K pool
    # min_df=3/max_df=0.6 prunes noise, but on a tiny sandbox upload those bounds
    # can leave an empty vocabulary (max_df < min_df), so relax them for small N.
    total = len(docs) + 1
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3 if total >= 20 else 1,
        max_df=0.6 if total >= 10 else 1.0,
        max_features=60000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    # Fit on the JD + corpus so JD terms are in-vocabulary, then score. A degenerate
    # corpus (e.g. only stop words) can still empty the vocabulary; fall back to the
    # same neutral score the missing-sklearn path uses rather than crash the caller.
    try:
        matrix = vectorizer.fit_transform([C.JD_QUERY_TEXT] + docs)
    except ValueError:
        return [0.5] * len(docs)
    jd_vec = matrix[0]
    cand_matrix = matrix[1:]
    sims = linear_kernel(jd_vec, cand_matrix).ravel()

    # Rank-percentile scaling -> robust to the long tail and keeps the signal in [0,1].
    order = sims.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(sims)
    ranks[order] = np.arange(n)
    scaled = ranks / max(n - 1, 1)
    return scaled.tolist()
