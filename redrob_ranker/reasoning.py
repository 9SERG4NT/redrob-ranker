"""Fact-grounded reasoning generation.

Stage-4 manual review checks each reasoning for: specific facts, JD connection,
honest concerns, no hallucination, variation, and rank-consistency. Every sentence
here is assembled only from values actually present in the candidate's record, so
nothing can be hallucinated. Tone and the strength/concern balance scale with the
candidate's rank so the reasoning stays consistent with where we placed them.
"""

from __future__ import annotations

from typing import Dict, List

_BUCKET_LABEL = {
    "embeddings_retrieval": "embeddings/retrieval",
    "vector_db": "vector search",
    "ranking_ir": "ranking & IR",
    "recsys": "recommendation systems",
    "python_ml": "Python/ML",
    "nlp_llm": "NLP/LLM",
    "mlops": "MLOps",
}


def _strengths(feat: Dict) -> List[str]:
    raw = feat["raw"]
    out: List[str] = []

    title = raw.get("title") or "candidate"
    yoe = raw.get("yoe")
    if isinstance(yoe, (int, float)):
        out.append(f"{title} with {yoe:.1f} yrs experience")
    else:
        out.append(str(title))

    # Top trust-weighted competency buckets actually covered.
    strong_buckets = [
        _BUCKET_LABEL[b]
        for b, t in sorted(feat["bucket_trust"].items(), key=lambda x: x[1], reverse=True)
        if t > 0.8
    ][:3]
    if strong_buckets:
        out.append("demonstrated depth in " + ", ".join(strong_buckets))

    # Named skills with credible trust.
    named = [s[0] for s in feat.get("matched_skills", [])][:3]
    if named:
        out.append("skills incl. " + ", ".join(named))

    # Career-narrative evidence (plain-language proof of the JD's exact work).
    evidence = [h for h in feat.get("lex_hits", []) if h in (
        "recommendation system", "learning to rank", "search relevance",
        "ranking and retrieval", "ranking system", "retrieval system",
        "semantic search", "vector search", "hybrid search", "reranking",
        "re-ranking", "information retrieval", "personalization", "matching layer",
        "offline to online", "a/b test", "ndcg", "at scale", "shipped", "in production",
    )]
    if evidence:
        out.append("career shows " + ", ".join(evidence[:2]))

    # Location relative to the JD's Pune/Noida preference.
    if feat["loc_kind"] == "preferred":
        out.append(f"based in {raw.get('location')} (JD-preferred)")
    elif feat["loc_kind"] == "tier1":
        out.append(f"in Tier-1 city {raw.get('location')}")
    elif feat["loc_kind"] == "india_other" and raw.get("willing_to_relocate"):
        out.append("willing to relocate")

    # Availability signals.
    rr = raw.get("response_rate")
    if isinstance(rr, (int, float)) and rr >= 0.6:
        out.append(f"responsive to recruiters ({rr:.0%})")
    if raw.get("open_to_work") and (raw.get("months_inactive") or 9) <= 3:
        out.append("open to work and recently active")

    return out


def _concerns(feat: Dict) -> List[str]:
    raw = feat["raw"]
    out: List[str] = []

    if feat["loc_kind"] == "abroad":
        out.append(f"based outside India ({raw.get('country')}; no visa sponsorship)")
    rr = raw.get("response_rate")
    if isinstance(rr, (int, float)) and rr < 0.25:
        out.append(f"low recruiter response rate ({rr:.0%})")
    mi = raw.get("months_inactive")
    if isinstance(mi, (int, float)) and mi >= 6:
        out.append(f"inactive ~{int(mi)} months")
    npd = raw.get("notice_period_days")
    if isinstance(npd, (int, float)) and npd >= 90:
        out.append(f"long notice period ({int(npd)}d)")
    yoe = raw.get("yoe")
    if isinstance(yoe, (int, float)):
        if yoe < 5:
            out.append(f"below the 5-9y band ({yoe:.1f}y)")
        elif yoe > 11:
            out.append(f"above the 5-9y band ({yoe:.1f}y)")
    if feat["services_only"]:
        out.append("entire career at services/consulting firms")
    if feat["cv_dominant"]:
        out.append("CV/speech-leaning, light on NLP/IR")
    if feat["pure_research"]:
        out.append("research framing with limited production evidence")
    if feat["title_chaser"]:
        out.append("short average tenure (job-hopping risk)")
    if not feat.get("matched_skills") and feat["lexicon"] < 0.2:
        out.append("AI/IR fit is inferred and thin")
    return out


def generate(feat: Dict, rank: int) -> str:
    """Produce a 1-2 sentence, rank-consistent justification."""
    if feat["is_honeypot"]:
        why = feat["honeypot_reasons"][0] if feat["honeypot_reasons"] else "internal profile inconsistency"
        return f"Flagged as inconsistent ({why}); ranked at the bottom and excluded from genuine fits."

    strengths = _strengths(feat)
    concerns = _concerns(feat)

    # Lead clause varies with rank so tone matches placement.
    head = strengths[0] if strengths else "Candidate"
    rest = strengths[1:]

    if rank <= 10:
        lead = f"Top fit: {head}"
        body = "; ".join(rest[:3])
        s = f"{lead}" + (f"; {body}." if body else ".")
        if concerns:
            s += f" Minor caveat: {concerns[0]}."
    elif rank <= 50:
        lead = f"Strong fit: {head}"
        body = "; ".join(rest[:2])
        s = f"{lead}" + (f"; {body}." if body else ".")
        if concerns:
            s += f" Concern: {concerns[0]}" + (f"; {concerns[1]}." if len(concerns) > 1 else ".")
    else:
        lead = f"Borderline fit: {head}"
        body = "; ".join(rest[:1])
        s = f"{lead}" + (f"; {body}." if body else ".")
        if concerns:
            s += " Concerns: " + "; ".join(concerns[:2]) + "."
        else:
            s += " Included as filler near the cutoff."

    # CSV-safety: keep it single-line and bounded.
    return " ".join(s.split())[:300]
