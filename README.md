# Redrob — Intelligent Candidate Discovery & Ranking

A transparent, CPU-only hybrid ranking system that selects the **top 100 candidates**
from a **100,000-candidate pool** for a single job description, with a fact-grounded
reason for every pick — plus an interactive HTML dashboard to explore and stress-test
the results.

Ranks the full pool in **~60 seconds** on a laptop (no GPU, no network), comfortably
inside the competition's 5-minute / 16 GB budget.

---

## 1. Problem statement

Given 100,000 synthetic candidate profiles, rank the 100 best fits for the
*"Senior AI Engineer — Founding Team"* role and output a CSV
(`candidate_id, rank, score, reasoning`). Submissions are scored against a hidden
relevance ground truth:

```
composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
```

The dataset is deliberately adversarial. The right answer is **not** "whoever lists the
most AI keywords." It rewards reading the profile and reasoning about real fit:

| Trap in the data | What it looks like | What a good system must do |
|---|---|---|
| **Keyword stuffers** (~70K) | Non-technical titles (Marketing Manager, HR) padded with AI buzzwords | Recognise the title/career mismatch and exclude them |
| **Plain-language "Tier-5" gems** | Genuine ranking/retrieval engineers who never write "RAG" or "Pinecone" | Recover them from career evidence, not buzzwords |
| **Honeypots** (~80) | Internally impossible profiles (14y experience over a 5y timeline; expert skills with 0 months used) | Detect the contradiction and floor them (>10% in top 100 = disqualified) |
| **Stale / unavailable** | Perfect on paper but inactive 6 months, 5% recruiter response | Down-weight by real availability signals |

Hard constraints: CPU only, no GPU, no network/LLM calls during ranking, ≤ 5 min, ≤ 16 GB.

---

## 2. Solution architecture

Each candidate's score is an **additive base** (how well they match the JD) multiplied by
a chain of **gates and modifiers** (whether they are genuinely hireable and real):

```
base  = 0.19·role + 0.22·skills + 0.19·narrative + 0.09·product
      + 0.13·experience + 0.13·location + 0.05·education          # every term in [0,1]

score = base × role_gate × behavioral_modifier × disqualifier_penalties × honeypot_gate
```

```
 candidates.jsonl(.gz)
        │  stream + parse (orjson)
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ feature extraction  (redrob_ranker/features.py)               │
 │   role · skills(trust-weighted) · narrative · product ·       │
 │   experience · location · education · behavioral · flags      │
 └──────────────────────────────────────────────────────────────┘
        │                                  │
        ▼                                  ▼
 TF-IDF semantic backbone          honeypot detector
 (text.py, cosine vs JD)           (honeypots.py)
        │                                  │
        └───────────────┬──────────────────┘
                        ▼
              scoring.finalize (scoring.py)
        base × gates × modifiers → final score
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
 submission.csv (top 100)       dashboard.html (explore + simulate)
   + reasoning.py text
```

### Component reference

| Component | Captures | Why it defeats the traps |
|---|---|---|
| **role** | Best title across current + past roles, mapped to AI/ML/Search/Recsys relevance | Decisive anti-stuffer signal — an off-target title is gated to ~0 (`role_gate`) |
| **skills** | Trust-weighted coverage of the JD's must-have buckets (embeddings/retrieval, vector DB, ranking/IR, Python/ML, …) | **trust = proficiency × duration × endorsements × on-platform assessment** — lazily stuffed skills (0 months, 0 endorsements, low assessment) collapse to ~0 |
| **narrative** | Career-prose lexicon (60%) + TF-IDF cosine to the JD (40%) | Recovers plain-language fits who describe the work without buzzwords |
| **product** | Share of tenure at product vs services/consulting companies | Rewards "shipped at a product company"; penalises services-only careers |
| **experience** | Soft band peaking at 5–9 years | A range, not a hard cut |
| **location** | India Tier-1 (Pune/Noida preferred), relocation, work-mode | Encodes the JD's strong India preference |
| **education** | Institution tier + field relevance | Minor tie-breaker |
| **behavioral_modifier** | Recruiter response, recency, open-to-work, interview completion, recruiter saves, notice period, verification | Down-weights "perfect on paper but unavailable" |
| **disqualifier_penalties** | Services-only, pure-research, CV/speech-without-NLP, title-chasing | The JD's explicit "do NOT want" list |
| **honeypot_gate** | Internal-contradiction detection | Floors impossible profiles → top-100 honeypot rate of **0** |

All domain knowledge (weights, lexicons, company lists, skill buckets) lives in one
auditable file: [`redrob_ranker/config.py`](redrob_ranker/config.py).

---

## 3. Repository layout

```
rank.py                     CLI entry point (the reproduce command)
build_dashboard.py          generates the interactive dashboard.html
redrob_ranker/
  config.py                 JD query, lexicons, company lists, weights  (all knobs)
  loader.py                 streaming JSONL / gzip reader (orjson if available)
  dates.py                  shared date helpers + pinned "now"
  features.py               per-candidate component extraction
  honeypots.py              internal-contradiction detection
  text.py                   TF-IDF semantic backbone
  scoring.py                base + gates/modifiers -> final score
  reasoning.py              fact-grounded, rank-consistent justifications
  pipeline.py               load -> featurize -> score -> rank -> write
app.py                      Streamlit sandbox (runs on a small uploaded sample)
dashboard.html              self-contained interactive dashboard (no server needed)
tests/                      unit tests for honeypots and scoring
Dockerfile                  self-contained reproduction / sandbox image
requirements.txt
submission_metadata.yaml
```

---

## 4. Quick start

```bash
pip install -r requirements.txt

# Produce the ranking (plain or gzipped JSONL both work):
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Validate against the competition spec:
python validate_submission.py submission.csv

# Build the interactive dashboard:
python build_dashboard.py --candidates ./candidates.jsonl --out ./dashboard.html

# Run the tests:
python -m pytest tests/ -q
```

No pre-computation step is required; everything (including TF-IDF) is built inside the
single ranking run.

---

## 5. Interactive dashboard

`dashboard.html` is a **single self-contained file** (open it directly in any browser —
no server, no network). Three sections:

1. **Ranking** — the top-100 table with per-candidate score breakdowns (click a row),
   red-flag chips, reasoning, and distribution charts.
2. **Dataset analysis** — the full 100K story: role composition, the
   `100K → genuine AI roles → India → 5–9y band → top 100` selection funnel, and the
   pool-wide score distribution.
3. **Simulation** — retune the ranker live. One-click **scenario presets**
   (Skills-only, Role-blind, Location-blind, Equal weights, …) and **gate toggles**
   (role gate / availability / penalties / honeypot gate). The top 100 re-ranks
   instantly and the KPIs (India %, off-target %, honeypots, "kept vs production") update
   so you can *see* each mechanism do its job — e.g. turning the honeypot gate off makes
   honeypots flood back in.

---

## 6. Results on the released pool

*(sanity checks against the JD's intent — not the hidden competition score)*

- Top-100: **98% India-based**, **0 honeypots**, experience **4–9 years** (mean ~6.4).
- Every top-100 title is a genuine AI/ML/Search/Recsys/Data-Science role — no
  keyword-stuffer survives the role gate.
- The plain-language "Tier-5" archetype (a Senior AI Engineer who built ranking/retrieval
  across product companies) lands at **rank 1**.
- Runs end-to-end in **~60 s** on CPU; the in-dashboard simulator at production weights
  reproduces the CSV ranking exactly.

---

## 7. Compute & reproducibility

| Constraint | This system |
|---|---|
| Runtime | ~60 s for 100K candidates |
| Memory | well under 16 GB (single streaming pass; heavy text dropped after TF-IDF) |
| Compute | CPU only |
| Network | none during ranking |
| Dependencies | `numpy`, `scikit-learn`, optional `orjson` (graceful fallback) |

Reproduce command: `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`
