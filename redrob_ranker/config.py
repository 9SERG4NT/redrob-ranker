"""Domain knowledge, lexicons, and tunable weights for the Redrob ranker.

Everything that encodes *what the JD actually wants* lives here so it can be read,
audited, and defended in one place. The job description for this challenge is the
"Senior AI Engineer - Founding Team" role: production embeddings/retrieval/ranking,
vector search, strong Python, rigorous ranking evaluation, shipped at a product
company, India Tier-1 located, and genuinely available.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Job-description query text (used for the TF-IDF semantic backbone).
# Hand-distilled from job_description.md to emphasise the competencies that
# actually decide fit, and to de-emphasise the keyword soup that stuffer
# candidates copy verbatim.
# ---------------------------------------------------------------------------
JD_QUERY_TEXT = """
senior ai engineer founding team owning the intelligence layer ranking retrieval
and matching systems for a talent platform. production experience with embeddings
based retrieval systems sentence transformers bge e5 deployed to real users.
embedding drift index refresh retrieval quality regression. vector databases and
hybrid search infrastructure pinecone weaviate qdrant milvus opensearch
elasticsearch faiss. learning to rank reranking relevance. recommendation systems
personalization search and discovery semantic search. designing evaluation
frameworks for ranking ndcg mrr map offline to online correlation a b testing.
strong python code quality. shipped an end to end ranking search or recommendation
system to real users at meaningful scale at a product company. nlp and information
retrieval. llm fine tuning lora qlora peft. applied machine learning in production
not pure research. scrappy product engineering ship a working ranker quickly and
iterate from real user feedback.
"""

# ---------------------------------------------------------------------------
# Title -> role-relevance score. Matched case-insensitively as substrings against
# both the current title and every title in the candidate's career history; the
# candidate's role score is the best (current title slightly preferred).
# This is the decisive signal against keyword-stuffer traps: a "Marketing Manager"
# with a perfect AI skill list scores ~0 here and is gated out.
# ---------------------------------------------------------------------------
TITLE_SCORES = [
    # (substring, score)  -- evaluated in order; first match wins
    ("recommendation systems engineer", 1.00),
    ("applied ml engineer", 1.00),
    ("applied machine learning", 1.00),
    ("machine learning engineer", 1.00),
    ("senior ai engineer", 1.00),
    ("lead ai engineer", 1.00),
    ("staff machine learning", 1.00),
    ("ml engineer", 0.98),
    ("ai engineer", 0.98),
    ("nlp engineer", 0.96),
    ("search engineer", 0.96),
    ("applied scientist", 0.95),
    ("software engineer (ml)", 0.95),
    ("ai specialist", 0.80),
    ("research engineer", 0.72),   # research-leaning; production redeems via narrative
    ("research scientist", 0.68),
    ("data scientist", 0.88),
    ("computer vision engineer", 0.66),  # CV-only is explicitly down-weighted by JD
    ("data engineer", 0.64),
    ("analytics engineer", 0.58),
    ("backend engineer", 0.56),
    ("full stack", 0.52),
    ("software engineer", 0.55),
    ("cloud engineer", 0.46),
    ("devops engineer", 0.44),
    ("frontend engineer", 0.40),
    ("mobile developer", 0.40),
    ("java developer", 0.42),
    (".net developer", 0.40),
    ("qa engineer", 0.38),
    ("data analyst", 0.48),
]
# Anything not matched above is treated as a non-technical / off-target role.
DEFAULT_TITLE_SCORE = 0.06

# ---------------------------------------------------------------------------
# Skill competency buckets. Buzzwords AND their plain-language equivalents map to
# the same underlying competency, so a "Tier-5" candidate who writes "Ranking
# Systems / Text Encoders" instead of "Learning to Rank / Embeddings" still gets
# full credit. Bucket weights reflect the JD's "absolutely need" list.
# ---------------------------------------------------------------------------
SKILL_BUCKETS = {
    "embeddings_retrieval": {
        "weight": 0.22,
        "skills": {
            "embeddings", "sentence transformers", "text encoders",
            "vector representations", "semantic search", "rag", "vector search",
            "haystack", "llamaindex",
        },
    },
    "vector_db": {
        "weight": 0.18,
        "skills": {
            "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
            "elasticsearch", "faiss", "pgvector", "search backend",
            "search infrastructure", "indexing algorithms",
        },
    },
    "ranking_ir": {
        "weight": 0.20,
        "skills": {
            "information retrieval", "information retrieval systems",
            "learning to rank", "bm25", "ranking systems", "search & discovery",
            "search and discovery", "ranking", "search",
        },
    },
    "recsys": {
        "weight": 0.10,
        "skills": {"recommendation systems", "content matching"},
    },
    "python_ml": {
        "weight": 0.16,
        "skills": {
            "python", "pytorch", "tensorflow", "scikit-learn", "machine learning",
            "deep learning", "feature engineering", "data science",
            "statistical modeling", "open-source ml libraries",
        },
    },
    "nlp_llm": {
        "weight": 0.09,
        "skills": {
            "nlp", "natural language processing", "llms", "llm",
            "hugging face transformers", "fine-tuning llms", "prompt engineering",
            "lora", "qlora", "peft", "model adaptation",
        },
    },
    "mlops": {
        "weight": 0.05,
        "skills": {
            "mlops", "mlflow", "kubeflow", "weights & biases",
            "workflow orchestration", "bentoml", "airflow",
        },
    },
}

# Skills that signal a computer-vision / speech / robotics focus. The JD does not
# want CV/speech specialists *without* NLP/IR exposure.
CV_SPEECH_SKILLS = {
    "computer vision", "opencv", "yolo", "image classification",
    "object detection", "speech recognition", "asr", "tts",
    "diffusion models", "gans", "cnn",
}

# ---------------------------------------------------------------------------
# Career-history narrative lexicon. Weighted phrases searched (lowercased) across
# headline + summary + every role description. This is how plain-language fits are
# recovered and how skill claims are corroborated by real work.
# ---------------------------------------------------------------------------
CAREER_PHRASES = [
    # decisive evidence of the exact work the JD wants
    ("recommendation system", 3.0),
    ("learning to rank", 3.0),
    ("search and discovery", 2.6),
    ("search & discovery", 2.6),
    ("ranking and retrieval", 3.0),
    ("ranking system", 2.6),
    ("retrieval system", 2.6),
    ("semantic search", 2.4),
    ("vector search", 2.4),
    ("hybrid search", 2.6),
    ("search relevance", 2.6),
    ("reranking", 2.4),
    ("re-ranking", 2.4),
    ("information retrieval", 2.6),
    ("personalization", 2.0),
    ("matching layer", 2.4),
    ("relevance", 1.4),
    ("embedding", 2.0),
    ("embeddings", 2.0),
    # rigorous evaluation (a JD "absolutely need")
    ("a/b test", 2.0),
    ("a b test", 1.6),
    ("offline metric", 2.2),
    ("online metric", 1.8),
    ("offline to online", 2.4),
    ("ndcg", 2.4),
    ("mrr", 2.0),
    ("offline experimentation", 2.0),
    ("evaluation methodology", 2.0),
    # production / shipping posture (JD wants shippers, not researchers)
    ("in production", 1.6),
    ("to production", 1.4),
    ("shipped", 1.6),
    ("at scale", 1.6),
    ("end-to-end", 1.4),
    ("end to end", 1.4),
    ("real users", 1.6),
    ("recruiter", 1.0),
    # general ML competency
    ("machine learning", 1.0),
    ("deep learning", 0.8),
    ("nlp", 1.0),
    ("natural language", 1.0),
    ("large language model", 1.2),
    ("fine-tun", 1.0),
    ("fine tun", 1.0),
    ("feature pipeline", 1.0),
    ("feature engineering", 0.8),
    ("vector database", 2.0),
    ("drift", 1.2),
    ("index refresh", 1.6),
    ("retrieval quality", 2.0),
]
# Saturation constant: career score = 1 - exp(-total_weight / CAREER_TAU)
CAREER_TAU = 9.0

# Phrases that indicate pure-research framing without production (mild negative).
RESEARCH_ONLY_PHRASES = [
    "published", "publication", "peer-reviewed", "research lab", "phd thesis",
    "academic", "state-of-the-art benchmark", "novel architecture",
]

# ---------------------------------------------------------------------------
# Companies. Services/consulting houses are penalised when they make up a
# candidate's *entire* career (JD-explicit). Product companies are rewarded.
# Fictional placeholder companies in the dataset are treated as neutral.
# ---------------------------------------------------------------------------
SERVICES_COMPANIES = {
    "infosys", "wipro", "tcs", "tata consultancy services", "accenture",
    "capgemini", "cognizant", "hcl", "tech mahindra", "mphasis", "mindtree",
    "ltimindtree", "genpact",
}
PRODUCT_COMPANIES = {
    # Indian product companies / startups
    "swiggy", "zomato", "cred", "razorpay", "flipkart", "meesho", "inmobi",
    "nykaa", "zoho", "freshworks", "vedantu", "ola", "phonepe", "paytm",
    "sharechat", "glance", "dunzo", "urban company", "unacademy", "byju's",
    "groww", "dream11", "rephrase.ai", "haptik", "observe.ai", "gupshup",
    # global product / tech
    "google", "meta", "facebook", "amazon", "microsoft", "apple", "netflix",
    "uber", "linkedin", "airbnb", "stripe", "nvidia", "openai", "anthropic",
    "spotify", "pinterest", "twitter", "snap", "adobe", "salesforce",
}

# ---------------------------------------------------------------------------
# Location scoring. JD: Pune/Noida preferred; Tier-1 Indian cities welcome;
# outside India case-by-case with no visa sponsorship.
# ---------------------------------------------------------------------------
PREFERRED_CITIES = {"noida", "pune"}              # JD office locations
TIER1_CITIES = {
    "bangalore", "bengaluru", "hyderabad", "mumbai", "delhi", "gurgaon",
    "gurugram", "chennai", "new delhi",
}

# ---------------------------------------------------------------------------
# Education tiers.
# ---------------------------------------------------------------------------
EDU_TIER_SCORE = {
    "tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.62, "tier_4": 0.48, "unknown": 0.5,
}
RELEVANT_FIELDS = {
    "computer science", "machine learning", "artificial intelligence",
    "data science", "statistics", "mathematics", "information technology",
    "electronics", "software engineering", "computational",
}

# ---------------------------------------------------------------------------
# Component weights for the additive base score (must sum to ~1.0).
# ---------------------------------------------------------------------------
COMPONENT_WEIGHTS = {
    "role": 0.19,
    "skills": 0.22,
    "narrative": 0.19,   # career lexicon + TF-IDF semantic backbone
    "product": 0.09,
    "experience": 0.13,
    "location": 0.13,
    "education": 0.05,
}

# Split of the "narrative" component between the hand lexicon and TF-IDF cosine.
NARRATIVE_LEXICON_SHARE = 0.6
NARRATIVE_TFIDF_SHARE = 0.4

# Behavioral-modifier multiplier range. behavioral_quality in [0,1] is mapped
# linearly into [B_MIN, B_MAX].
B_MIN = 0.66
B_MAX = 1.12

# Role gate: crushes off-target roles (keyword stuffers) regardless of other
# components. Keyed on thresholds of the best role score.
ROLE_GATE = [
    (0.20, 0.10),   # best role score < 0.20  -> x0.10  (non-technical)
    (0.40, 0.34),   # < 0.40 -> x0.34
    (0.55, 0.78),   # < 0.55 -> x0.78         (generic tech, leans on narrative)
    (1.01, 1.00),   # otherwise -> x1.00
]

# Disqualifier multipliers (JD "things we explicitly do NOT want").
PENALTY_SERVICES_ONLY = 0.42      # entire career at consulting/services houses
PENALTY_PURE_RESEARCH = 0.55      # research framing, no production evidence
PENALTY_CV_SPEECH_ONLY = 0.50     # CV/speech specialist without NLP/IR
PENALTY_TITLE_CHASER = 0.74       # >=4 jobs, avg tenure < 18 months
PENALTY_RECENT_LLM_ONLY = 0.80    # only <12mo of LangChain/LLM-wrapper work

# Honeypot gate: near-zero (kept > 0 so the row is still sortable / explainable).
HONEYPOT_SCORE = 0.02
