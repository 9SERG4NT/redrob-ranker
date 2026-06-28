"""Feature extraction: turn a raw candidate record into a compact dict of scored
components plus the few raw facts the reasoning generator needs. Designed to run in
a single streaming pass so we never hold all 100K full records in memory at once.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from . import config as C
from . import honeypots
from .dates import NOW_MONTHS, ym

_PROF_W = {"beginner": 0.35, "intermediate": 0.60, "advanced": 0.85, "expert": 1.0}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _title_score(title: str) -> float:
    t = (title or "").lower()
    for sub, sc in C.TITLE_SCORES:
        if sub in t:
            return sc
    return C.DEFAULT_TITLE_SCORE


def _norm_company(name: str) -> str:
    return (name or "").strip().lower()


# --------------------------------------------------------------------------- #
# Component scorers
# --------------------------------------------------------------------------- #
def _role_component(profile: Dict, career: List[Dict]) -> float:
    """Best role relevance across current and historical titles."""
    cur = _title_score(profile.get("current_title", ""))
    best_hist = max((_title_score(r.get("title", "")) for r in career), default=0.0)
    # Current title is the strongest signal; historical AI roles count slightly less.
    return max(cur, 0.92 * best_hist)


def _skill_components(skills: List[Dict], assessments: Dict[str, float]):
    """Trust-weighted competency coverage across JD skill buckets.

    trust = proficiency x duration-credit x endorsement-credit x assessment-credit.
    Keyword stuffers list many AI skills but with 0 months, 0 endorsements and low
    assessment scores, so their trust collapses toward zero.
    """
    assess_lc = { (k or "").lower(): v for k, v in (assessments or {}).items() }
    bucket_trust = {b: 0.0 for b in C.SKILL_BUCKETS}
    cv_speech_trust = 0.0
    matched: List[tuple] = []  # (skill_name, bucket, trust) for reasoning

    for s in skills:
        name = (s.get("name") or "").lower()
        prof_w = _PROF_W.get(s.get("proficiency"), 0.5)
        dur = s.get("duration_months", 0) or 0
        dur_factor = min(1.0, dur / 24.0)
        endo = s.get("endorsements", 0) or 0
        endo_factor = 0.55 + 0.45 * min(1.0, endo / 20.0)
        if name in assess_lc:
            assess_factor = 0.45 + 0.55 * (assess_lc[name] / 100.0)
        else:
            assess_factor = 0.82  # not assessed -> mildly neutral
        trust = prof_w * (0.35 + 0.65 * dur_factor) * endo_factor * assess_factor

        for bname, bdef in C.SKILL_BUCKETS.items():
            if name in bdef["skills"]:
                bucket_trust[bname] += trust
                matched.append((s.get("name"), bname, trust))
                break
        if name in C.CV_SPEECH_SKILLS:
            cv_speech_trust += trust

    # Saturate each bucket independently, then weight by JD priority.
    skill_score = 0.0
    relevant_trust = 0.0
    for bname, bdef in C.SKILL_BUCKETS.items():
        sat = 1.0 - math.exp(-bucket_trust[bname] / 1.4)
        skill_score += bdef["weight"] * sat
        relevant_trust += bucket_trust[bname]

    # Coverage bonus: rewards breadth across the JD "absolutely need" buckets
    # rather than stacking a single bucket.
    must_have = ("embeddings_retrieval", "vector_db", "ranking_ir", "python_ml")
    covered = sum(1 for b in must_have if bucket_trust[b] > 0.8)
    coverage_bonus = 0.12 * (covered / len(must_have))
    skill_score = min(1.0, skill_score + coverage_bonus)

    cv_dominant = (
        cv_speech_trust > 1.5
        and relevant_trust < 0.6 * cv_speech_trust
        and bucket_trust["nlp_llm"] < 0.8
        and bucket_trust["ranking_ir"] < 0.5
    )
    matched.sort(key=lambda x: x[2], reverse=True)
    return skill_score, bucket_trust, cv_dominant, matched


def _narrative_lexicon(text: str) -> tuple[float, List[str]]:
    """Weighted phrase evidence from the candidate's free text."""
    total = 0.0
    hits: List[str] = []
    for phrase, w in C.CAREER_PHRASES:
        if phrase in text:
            total += w
            hits.append(phrase)
    score = 1.0 - math.exp(-total / C.CAREER_TAU)
    return score, hits


def _experience_component(yoe: float) -> float:
    """Soft band peaking at 5-9 years (JD: a range, not a hard requirement)."""
    if yoe is None:
        return 0.4
    if 5.0 <= yoe <= 9.0:
        return 1.0
    if yoe < 5.0:
        # Steeper ramp: a 3y "junior" is a clear notch below the 5-9y band.
        return max(0.08, 0.08 + 0.92 * ((yoe - 1.0) / 4.0))
    # yoe > 9
    return max(0.38, 1.0 - 0.07 * (yoe - 9.0))


def _location_component(profile: Dict, willing_to_relocate: bool, work_mode: str) -> tuple[float, str]:
    loc = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    city = loc.split(",")[0].strip()
    if country == "india":
        if city in C.PREFERRED_CITIES:
            return 1.0, "preferred"
        if city in C.TIER1_CITIES:
            return 0.94, "tier1"
        # other Indian city
        return (0.82 if willing_to_relocate else 0.72), "india_other"
    # Outside India: case-by-case, no visa sponsorship (JD strongly prefers India).
    base = 0.34
    if willing_to_relocate:
        base = 0.52
    if work_mode in ("remote", "flexible"):
        base += 0.04
    return min(base, 0.56), "abroad"


def _education_component(education: List[Dict]) -> float:
    if not education:
        return 0.5
    best = 0.0
    for e in education:
        tier_sc = C.EDU_TIER_SCORE.get(e.get("tier", "unknown"), 0.5)
        field = (e.get("field_of_study") or "").lower()
        rel = any(f in field for f in C.RELEVANT_FIELDS)
        best = max(best, min(1.0, tier_sc + (0.08 if rel else 0.0)))
    return best


def _product_component(career: List[Dict]) -> tuple[float, bool, bool]:
    """Fraction of career at product companies + flags."""
    if not career:
        return 0.5, False, False
    months_product = 0.0
    months_services = 0.0
    months_total = 0.0
    any_product = False
    for r in career:
        comp = _norm_company(r.get("company", ""))
        dur = r.get("duration_months", 0) or 0
        months_total += dur
        if comp in C.PRODUCT_COMPANIES:
            months_product += dur
            any_product = True
        elif comp in C.SERVICES_COMPANIES:
            months_services += dur
    if months_total <= 0:
        return 0.5, False, False
    frac_product = months_product / months_total
    frac_services = months_services / months_total
    # Neutral baseline 0.5; product pulls up, all-services pulls down.
    score = 0.5 + 0.5 * frac_product - 0.25 * frac_services
    score = max(0.0, min(1.0, score))
    services_only = (frac_services > 0.99) and not any_product
    return score, services_only, any_product


def _behavioral_quality(sig: Dict) -> tuple[float, Dict]:
    """Availability/engagement quality in [0,1] + raw facts for reasoning."""
    resp = sig.get("recruiter_response_rate", 0.0) or 0.0
    last_active = ym(sig.get("last_active_date", ""))
    months_inactive = (NOW_MONTHS - last_active) if last_active else 9
    open_to_work = bool(sig.get("open_to_work_flag", False))
    icr = sig.get("interview_completion_rate", 0.0) or 0.0
    saved = sig.get("saved_by_recruiters_30d", 0) or 0
    completeness = sig.get("profile_completeness_score", 0.0) or 0.0
    notice = sig.get("notice_period_days")
    verified = sum(
        bool(sig.get(k)) for k in ("verified_email", "verified_phone", "linkedin_connected")
    )

    # Sub-scores in [0,1].
    f_resp = resp                                   # respond to recruiters
    f_active = max(0.0, 1.0 - months_inactive / 9.0)  # recency
    f_open = 1.0 if open_to_work else 0.55
    f_icr = icr
    f_saved = min(1.0, saved / 15.0)                # recruiter demand
    f_complete = completeness / 100.0
    f_verified = verified / 3.0
    # Notice period (JD: "we'd love sub-30-day notice ... 30+ day candidates are
    # still in scope but the bar gets higher"). <=30d is ideal; 90d (the pool
    # median) is middling; 180d is a real availability drag, never zero.
    if notice is None:
        f_notice = 0.6
    elif notice <= 30:
        f_notice = 1.0
    elif notice <= 90:
        f_notice = 1.0 - 0.4 * ((notice - 30) / 60.0)
    else:
        f_notice = max(0.3, 0.6 - 0.3 * ((notice - 90) / 90.0))

    quality = (
        0.27 * f_resp
        + 0.20 * f_active
        + 0.12 * f_notice
        + 0.12 * f_open
        + 0.09 * f_icr
        + 0.09 * f_saved
        + 0.06 * f_complete
        + 0.05 * f_verified
    )
    facts = {
        "response_rate": resp,
        "months_inactive": months_inactive,
        "open_to_work": open_to_work,
        "saved_by_recruiters_30d": saved,
        "notice_period_days": sig.get("notice_period_days"),
        "github_activity_score": sig.get("github_activity_score"),
    }
    return max(0.0, min(1.0, quality)), facts


def _title_chaser(career: List[Dict]) -> bool:
    # Reserve the penalty for clear, consistent hopping (JD: "every 1.5 years").
    # A couple of short stints in an otherwise stable career should not trigger it.
    durs = [r.get("duration_months", 0) for r in career]
    if len(durs) >= 5 and (sum(durs) / len(durs)) < 18.0:
        return True
    if len(durs) >= 4 and (sum(durs) / len(durs)) < 15.0:
        return True
    return False


def _pure_research(profile: Dict, text: str, product_any: bool) -> bool:
    title = (profile.get("current_title") or "").lower()
    research_title = "research" in title
    research_hits = sum(1 for p in C.RESEARCH_ONLY_PHRASES if p in text)
    production_hits = any(
        p in text for p in ("in production", "to production", "shipped", "real users", "at scale")
    )
    return research_title and research_hits >= 1 and not production_hits and not product_any


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compute every component + raw reasoning facts for one candidate."""
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    sig = candidate.get("redrob_signals", {}) or {}

    # Free-text corpus for lexicon + TF-IDF.
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    parts += [r.get("description", "") for r in career]
    parts += [r.get("title", "") for r in career]
    text = " \n ".join(p for p in parts if p)
    text_lc = text.lower()
    # Skill names appended (lower weight) so TF-IDF still sees them, but the
    # lexicon scoring above runs only on real prose.
    tfidf_doc = text + " " + " ".join(s.get("name", "") for s in skills)

    role = _role_component(profile, career)
    skill_score, bucket_trust, cv_dominant, matched_skills = _skill_components(
        skills, sig.get("skill_assessment_scores", {})
    )
    lex_score, lex_hits = _narrative_lexicon(text_lc)
    yoe = profile.get("years_of_experience")
    exp = _experience_component(yoe)
    willing = bool(sig.get("willing_to_relocate", False))
    work_mode = sig.get("preferred_work_mode", "")
    loc, loc_kind = _location_component(profile, willing, work_mode)
    edu = _education_component(education)
    product, services_only, product_any = _product_component(career)
    beh_q, beh_facts = _behavioral_quality(sig)

    is_honeypot, hp_reasons = honeypots.detect(candidate)
    title_chaser = _title_chaser(career)
    pure_research = _pure_research(profile, text_lc, product_any)

    return {
        "candidate_id": candidate.get("candidate_id"),
        # components in [0,1]
        "role": role,
        "skills": skill_score,
        "lexicon": lex_score,
        "experience": exp,
        "location": loc,
        "education": edu,
        "product": product,
        "behavioral_quality": beh_q,
        # flags
        "cv_dominant": cv_dominant,
        "services_only": services_only,
        "title_chaser": title_chaser,
        "pure_research": pure_research,
        "is_honeypot": is_honeypot,
        "honeypot_reasons": hp_reasons,
        "bucket_trust": bucket_trust,
        # raw facts for reasoning
        "tfidf_doc": tfidf_doc,
        "loc_kind": loc_kind,
        "matched_skills": matched_skills[:6],
        "lex_hits": lex_hits,
        "raw": {
            "name": profile.get("anonymized_name"),
            "title": profile.get("current_title"),
            "company": profile.get("current_company"),
            "company_size": profile.get("current_company_size"),
            "industry": profile.get("current_industry"),
            "yoe": yoe,
            "location": profile.get("location"),
            "country": profile.get("country"),
            "education": education[:1],
            "willing_to_relocate": willing,
            "work_mode": work_mode,
            **beh_facts,
        },
    }
