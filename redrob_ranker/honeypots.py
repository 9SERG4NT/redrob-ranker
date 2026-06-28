"""Honeypot detection.

The dataset embeds ~80 "subtly impossible" profiles forced to relevance tier 0;
ranking them in the top 100 risks disqualification (>10% honeypot rate). We do not
special-case identities - we detect the *internal contradictions* that make a
profile impossible, exactly the inconsistencies described in the submission spec:

  * "8 years of experience at a company founded 3 years ago"
        -> years_of_experience can't be supported by the career timeline, or a
           single role's duration_months exceeds its own start->end span.
  * "'expert' proficiency in 10 skills with 0 years used"
        -> several advanced/expert skills with duration_months == 0.

A candidate is gated to ~zero score if ANY hard contradiction is found.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .dates import NOW_MONTHS, ym


def detect(candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Return (is_honeypot, reasons)."""
    reasons: List[str] = []
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", []) or []

    # --- 1. Role duration exceeds its own calendar span -----------------------
    for role in career:
        start = ym(role.get("start_date", ""))
        end_raw = role.get("end_date")
        end = ym(end_raw) if end_raw else NOW_MONTHS
        dur = role.get("duration_months")
        if start is not None and end is not None and isinstance(dur, int):
            span = end - start
            if dur - span > 9:  # claims far longer tenure than dates allow
                reasons.append(
                    f"role at {role.get('company','?')} claims {dur} months but "
                    f"its dates span only {max(span,0)}"
                )
            if start > NOW_MONTHS + 2:
                reasons.append(f"role at {role.get('company','?')} starts in the future")
            if end_raw and ym(end_raw) is not None and ym(end_raw) < start:
                reasons.append(f"role at {role.get('company','?')} ends before it starts")

    # --- 2. Stated experience can't be supported by the career timeline -------
    yoe = profile.get("years_of_experience")
    if isinstance(yoe, (int, float)) and career:
        total_career_years = sum(
            r.get("duration_months", 0) for r in career
        ) / 12.0
        # Career history caps at 10 roles; only flag when the gap is large AND the
        # candidate is NOT capped (so the gap can't be explained by hidden roles).
        if len(career) < 10 and (yoe - total_career_years) > 5.0:
            reasons.append(
                f"claims {yoe:.0f}y experience but career history totals only "
                f"{total_career_years:.1f}y across {len(career)} roles"
            )
        # Earliest start vs claimed experience: starting 16y ago but claiming 25y.
        starts = [s for s in (ym(r.get("start_date", "")) for r in career) if s]
        if starts:
            earliest_years = (NOW_MONTHS - min(starts)) / 12.0
            if yoe - earliest_years > 6.0:
                reasons.append(
                    f"claims {yoe:.0f}y experience but first role began only "
                    f"{earliest_years:.1f}y ago"
                )

    # --- 3. Advanced/expert proficiency with zero months of use ---------------
    skills = candidate.get("skills", []) or []
    zero_dur_expert = [
        s.get("name", "?")
        for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 1) == 0
    ]
    if len(zero_dur_expert) >= 3:
        reasons.append(
            f"{len(zero_dur_expert)} advanced/expert skills with 0 months of use"
        )

    return (len(reasons) > 0, reasons)
