"""Shared fixtures: minimal candidate records exercising each scoring path."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _signals(**overrides):
    base = {
        "profile_completeness_score": 80.0,
        "signup_date": "2023-01-01",
        "last_active_date": "2026-05-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 20,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.7,
        "avg_response_time_hours": 10.0,
        "skill_assessment_scores": {},
        "connection_count": 300,
        "endorsements_received": 50,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20, "max": 40},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 40,
        "search_appearance_30d": 100,
        "saved_by_recruiters_30d": 10,
        "interview_completion_rate": 0.8,
        "offer_acceptance_rate": 0.6,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
def genuine_ai_engineer():
    """Strong, consistent Senior AI Engineer in Pune at a product company."""
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test One",
            "headline": "Senior AI Engineer | ranking & retrieval at scale",
            "summary": "Built ranking and retrieval systems in production; shipped "
            "recommendation system improvements with offline-to-online A/B testing.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Senior AI Engineer",
            "current_company": "Razorpay",
            "current_company_size": "1001-5000",
            "current_industry": "Fintech",
        },
        "career_history": [
            {
                "company": "Razorpay",
                "title": "Senior AI Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 65,
                "is_current": True,
                "industry": "Fintech",
                "company_size": "1001-5000",
                "description": "Owned the search relevance and ranking system; "
                "semantic search, learning to rank, A/B testing, in production at scale.",
            }
        ],
        "education": [
            {
                "institution": "IIT Bombay",
                "degree": "B.Tech",
                "field_of_study": "Computer Science",
                "start_year": 2014,
                "end_year": 2018,
                "grade": "8.5 CGPA",
                "tier": "tier_1",
            }
        ],
        "skills": [
            {"name": "Embeddings", "proficiency": "expert", "endorsements": 40, "duration_months": 60},
            {"name": "Learning to Rank", "proficiency": "expert", "endorsements": 30, "duration_months": 50},
            {"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 40},
            {"name": "Python", "proficiency": "expert", "endorsements": 50, "duration_months": 80},
            {"name": "Information Retrieval", "proficiency": "advanced", "endorsements": 25, "duration_months": 45},
        ],
        "redrob_signals": _signals(),
    }


@pytest.fixture
def keyword_stuffer():
    """Marketing Manager with a stuffed AI skill list, all 0-month/0-endorsement."""
    return {
        "candidate_id": "CAND_0000002",
        "profile": {
            "anonymized_name": "Test Two",
            "headline": "Marketing Manager",
            "summary": "Led marketing campaigns and brand strategy across regions.",
            "location": "Pune, Maharashtra",
            "country": "India",
            "years_of_experience": 7.0,
            "current_title": "Marketing Manager",
            "current_company": "Acme Corp",
            "current_company_size": "1001-5000",
            "current_industry": "Manufacturing",
        },
        "career_history": [
            {
                "company": "Acme Corp",
                "title": "Marketing Manager",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 65,
                "is_current": True,
                "industry": "Manufacturing",
                "company_size": "1001-5000",
                "description": "Managed campaigns, SEO, content calendars and budgets.",
            }
        ],
        "education": [],
        "skills": [
            {"name": "RAG", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Pinecone", "proficiency": "expert", "endorsements": 0, "duration_months": 0},
            {"name": "Embeddings", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
            {"name": "LLMs", "proficiency": "expert", "endorsements": 1, "duration_months": 0},
            {"name": "Vector Search", "proficiency": "advanced", "endorsements": 0, "duration_months": 0},
        ],
        "redrob_signals": _signals(),
    }


@pytest.fixture
def honeypot_candidate(genuine_ai_engineer):
    """Impossible: claims 14y experience over a ~5y timeline."""
    c = copy.deepcopy(genuine_ai_engineer)
    c["candidate_id"] = "CAND_0000003"
    c["profile"]["years_of_experience"] = 14.0
    return c
