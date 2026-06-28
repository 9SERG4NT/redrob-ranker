"""Combine per-candidate components into a final fit score.

    base   = weighted sum of [role, skills, narrative, product, experience,
                              location, education]                       in [0,1]
    score  = base x role_gate x behavioral_modifier x disqualifier_penalties
                                                     x honeypot_gate

The additive base captures *how well the candidate matches the JD*; the multiplicative
modifiers capture *whether they are actually hireable/available* and hard-gate the
traps (off-target roles, honeypots, JD disqualifiers).
"""

from __future__ import annotations

from typing import Dict

from . import config as C


def _role_gate(role_score: float) -> float:
    for threshold, mult in C.ROLE_GATE:
        if role_score < threshold:
            return mult
    return 1.0


def _behavioral_modifier(quality: float) -> float:
    return C.B_MIN + (C.B_MAX - C.B_MIN) * quality


def _disqualifier_penalty(feat: Dict) -> float:
    mult = 1.0
    if feat["services_only"]:
        mult *= C.PENALTY_SERVICES_ONLY
    if feat["pure_research"]:
        mult *= C.PENALTY_PURE_RESEARCH
    if feat["cv_dominant"]:
        mult *= C.PENALTY_CV_SPEECH_ONLY
    if feat["title_chaser"]:
        mult *= C.PENALTY_TITLE_CHASER
    return mult


def finalize(feat: Dict, tfidf_score: float) -> Dict:
    """Attach 'base', 'score', and the modifier breakdown to *feat* and return it."""
    narrative = (
        C.NARRATIVE_LEXICON_SHARE * feat["lexicon"]
        + C.NARRATIVE_TFIDF_SHARE * tfidf_score
    )

    w = C.COMPONENT_WEIGHTS
    base = (
        w["role"] * feat["role"]
        + w["skills"] * feat["skills"]
        + w["narrative"] * narrative
        + w["product"] * feat["product"]
        + w["experience"] * feat["experience"]
        + w["location"] * feat["location"]
        + w["education"] * feat["education"]
    )

    role_gate = _role_gate(feat["role"])
    beh_mod = _behavioral_modifier(feat["behavioral_quality"])
    dq = _disqualifier_penalty(feat)

    if feat["is_honeypot"]:
        score = C.HONEYPOT_SCORE * base  # crush to the very bottom, keep sortable
    else:
        score = base * role_gate * beh_mod * dq

    feat["tfidf"] = tfidf_score
    feat["narrative"] = narrative
    feat["base"] = base
    feat["role_gate"] = role_gate
    feat["behavioral_modifier"] = beh_mod
    feat["disqualifier_penalty"] = dq
    feat["score"] = score
    return feat
