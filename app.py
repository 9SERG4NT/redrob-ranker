"""Streamlit sandbox for the Redrob ranker.

A lightweight hosted demo that satisfies the submission-spec sandbox requirement:
accept a small candidate sample (<=100), run the full ranking system end-to-end on
CPU, and display the ranked table with reasoning. Deploy free on Streamlit Cloud or
HuggingFace Spaces.

    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os

import streamlit as st

from redrob_ranker import features as F
from redrob_ranker import reasoning, scoring
from redrob_ranker.text import compute_tfidf_scores

st.set_page_config(page_title="Redrob Candidate Ranker", layout="wide")
st.title("Redrob — Senior AI Engineer Candidate Ranker")
st.caption(
    "Hybrid rule-based + TF-IDF ranker. Upload a JSONL sample (<=100 candidates) "
    "or use the bundled sample. CPU-only, no network."
)

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_candidates.jsonl")


def _read_jsonl(text: str):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _rank(candidates):
    feats, docs = [], []
    for c in candidates:
        if not c.get("candidate_id"):
            continue
        f = F.extract(c)
        docs.append(f.pop("tfidf_doc"))
        feats.append(f)
    tfidf = compute_tfidf_scores(docs)
    for f, tf in zip(feats, tfidf):
        scoring.finalize(f, tf)
    mx = max((f["score"] for f in feats), default=1.0) or 1.0
    for f in feats:
        f["display_score"] = round(f["score"] / mx, 6)
    feats.sort(key=lambda f: (-f["display_score"], f["candidate_id"]))
    return feats


col1, col2 = st.columns([2, 1])
with col2:
    use_sample = st.button("Use bundled sample (50 candidates)")
    uploaded = st.file_uploader("…or upload candidates.jsonl", type=["jsonl", "json", "txt"])
    top_n = st.slider("Top N", 5, 100, 25)

candidates = None
if use_sample and os.path.exists(SAMPLE_PATH):
    with open(SAMPLE_PATH, "r", encoding="utf-8") as fh:
        candidates = _read_jsonl(fh.read())
elif uploaded is not None:
    candidates = _read_jsonl(io.TextIOWrapper(uploaded, encoding="utf-8").read())

if candidates:
    candidates = candidates[:100]
    with st.spinner(f"Ranking {len(candidates)} candidates…"):
        ranked = _rank(candidates)[:top_n]
    rows = []
    for i, f in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": i,
                "candidate_id": f["candidate_id"],
                "title": f["raw"]["title"],
                "yoe": f["raw"]["yoe"],
                "location": f["raw"]["location"],
                "score": f["display_score"],
                "reasoning": reasoning.generate(f, i),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    csv_lines = ["candidate_id,rank,score,reasoning"]
    for r in rows:
        rsn = '"' + r["reasoning"].replace('"', '""') + '"'
        csv_lines.append(f'{r["candidate_id"]},{r["rank"]},{r["score"]:.6f},{rsn}')
    st.download_button(
        "Download ranking CSV", "\n".join(csv_lines), file_name="submission_sample.csv"
    )
else:
    st.info("Click **Use bundled sample** or upload a JSONL file to run the ranker.")
