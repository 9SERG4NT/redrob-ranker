"""Honeypot detector should fire on internal contradictions and pass clean profiles."""

import copy

from redrob_ranker.honeypots import detect


def test_clean_profile_is_not_a_honeypot(genuine_ai_engineer):
    is_hp, reasons = detect(genuine_ai_engineer)
    assert is_hp is False, reasons


def test_experience_exceeds_timeline(honeypot_candidate):
    is_hp, reasons = detect(honeypot_candidate)
    assert is_hp is True
    assert any("experience" in r for r in reasons)


def test_role_duration_exceeds_span(genuine_ai_engineer):
    c = copy.deepcopy(genuine_ai_engineer)
    # 200 months of tenure in a role whose dates span ~5 years.
    c["career_history"][0]["duration_months"] = 200
    is_hp, reasons = detect(c)
    assert is_hp is True
    assert any("span" in r for r in reasons)


def test_expert_skills_with_zero_months(genuine_ai_engineer):
    c = copy.deepcopy(genuine_ai_engineer)
    for s in c["skills"]:
        s["proficiency"] = "expert"
        s["duration_months"] = 0
    is_hp, reasons = detect(c)
    assert is_hp is True
    assert any("0 months" in r for r in reasons)
