"""End-to-end scoring behaviour: genuine fit ranks above stuffer; honeypot is crushed."""

from redrob_ranker import features as F
from redrob_ranker import scoring


def _score(candidate, tfidf=0.8):
    feat = F.extract(candidate)
    feat.pop("tfidf_doc")
    return scoring.finalize(feat, tfidf)


def test_genuine_beats_keyword_stuffer(genuine_ai_engineer, keyword_stuffer):
    good = _score(genuine_ai_engineer)
    bad = _score(keyword_stuffer)
    assert good["score"] > 5 * bad["score"], (good["score"], bad["score"])


def test_keyword_stuffer_is_role_gated(keyword_stuffer):
    bad = _score(keyword_stuffer)
    # Off-target title -> heavy role gate and near-zero skill trust.
    assert bad["role"] < 0.2
    assert bad["role_gate"] <= 0.12
    assert bad["skills"] < 0.1


def test_skill_trust_collapses_for_zero_month_skills(keyword_stuffer, genuine_ai_engineer):
    bad = _score(keyword_stuffer)
    good = _score(genuine_ai_engineer)
    # Same buzzwords appear in both, but trust weighting separates them.
    assert good["skills"] > 0.4 > bad["skills"]


def test_honeypot_is_crushed(honeypot_candidate, genuine_ai_engineer):
    hp = _score(honeypot_candidate)
    good = _score(genuine_ai_engineer)
    assert hp["is_honeypot"] is True
    assert hp["score"] < 0.05 * good["score"]


def test_components_are_bounded(genuine_ai_engineer):
    f = _score(genuine_ai_engineer)
    for key in ("role", "skills", "lexicon", "experience", "location", "education", "product"):
        assert 0.0 <= f[key] <= 1.0, (key, f[key])
