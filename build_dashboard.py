#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard visualizing the top-100 ranking.

Reuses the production ranker, then embeds the ranked candidates (with their full
score breakdowns and reasoning) plus aggregate stats into a single static HTML file
that opens offline in any browser - no server, no network, no dependencies.

    python build_dashboard.py --candidates ./candidates.jsonl --out ./dashboard.html
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from redrob_ranker import config as C
from redrob_ranker import reasoning
from redrob_ranker.pipeline import rank_all, write_submission

_COMPONENTS = ["role", "skills", "narrative", "product", "experience", "location", "education"]
_SIM_POOL = 1200  # candidates exposed to the in-browser weight simulator
_BUCKETS = ["embeddings_retrieval", "vector_db", "ranking_ir", "recsys", "python_ml", "nlp_llm", "mlops"]


def _records(top):
    out = []
    for i, f in enumerate(top, start=1):
        raw = f["raw"]
        out.append(
            {
                "rank": i,
                "id": f["candidate_id"],
                "name": raw.get("name"),
                "title": raw.get("title"),
                "company": raw.get("company"),
                "yoe": raw.get("yoe"),
                "location": raw.get("location"),
                "country": raw.get("country"),
                "score": f["display_score"],
                "base": round(f["base"], 4),
                "components": {c: round(float(f[c]), 4) for c in _COMPONENTS},
                "gates": {
                    "role_gate": round(f["role_gate"], 3),
                    "behavioral_modifier": round(f["behavioral_modifier"], 3),
                    "disqualifier_penalty": round(f["disqualifier_penalty"], 3),
                },
                "behavioral_quality": round(f["behavioral_quality"], 3),
                "buckets": {b: round(f["bucket_trust"].get(b, 0.0), 2) for b in _BUCKETS},
                "flags": {
                    "cv_dominant": f["cv_dominant"],
                    "services_only": f["services_only"],
                    "title_chaser": f["title_chaser"],
                    "pure_research": f["pure_research"],
                    "is_honeypot": f["is_honeypot"],
                },
                "facts": {
                    "response_rate": raw.get("response_rate"),
                    "months_inactive": raw.get("months_inactive"),
                    "notice_period_days": raw.get("notice_period_days"),
                    "open_to_work": raw.get("open_to_work"),
                    "saved_by_recruiters_30d": raw.get("saved_by_recruiters_30d"),
                    "github_activity_score": raw.get("github_activity_score"),
                    "willing_to_relocate": raw.get("willing_to_relocate"),
                    "work_mode": raw.get("work_mode"),
                },
                "reasoning": reasoning.generate(f, i),
            }
        )
    return out


def _aggregates(records):
    n = len(records) or 1
    india = sum(1 for r in records if (r["country"] or "") == "India")
    honey = sum(1 for r in records if r["flags"]["is_honeypot"])
    yoes = [r["yoe"] for r in records if isinstance(r["yoe"], (int, float))]
    notices = [r["facts"]["notice_period_days"] for r in records if isinstance(r["facts"]["notice_period_days"], (int, float))]
    titles = Counter(r["title"] for r in records)
    cities = Counter((r["location"] or "?").split(",")[0].strip() for r in records)

    def hist(vals, edges):
        labels, counts = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            labels.append(f"{a}-{b}")
            counts.append(sum(1 for v in vals if a <= v < b))
        return {"labels": labels, "counts": counts}

    return {
        "total": len(records),
        "india_pct": round(100 * india / n),
        "honeypots": honey,
        "avg_yoe": round(sum(yoes) / len(yoes), 1) if yoes else None,
        "avg_notice": round(sum(notices) / len(notices)) if notices else None,
        "titles": titles.most_common(12),
        "cities": cities.most_common(12),
        "exp_hist": hist(yoes, [0, 2, 4, 5, 7, 9, 11, 13, 20]),
        "notice_hist": hist(notices, [0, 15, 30, 45, 60, 90, 120, 181]),
        "scores": [r["score"] for r in records],
    }


def _pool_analysis(feats):
    """Aggregate the *whole* pool to tell the dataset story (traps -> top 100)."""
    n = len(feats) or 1
    india = sum(1 for f in feats if (f["raw"].get("country") or "") == "India")
    honey = sum(1 for f in feats if f["is_honeypot"])

    # Role classes: how the 100K pool splits into genuine AI roles vs traps.
    ai_core = strong_tech = weak_tech = off_target = 0
    for f in feats:
        r = f["role"]
        if r >= 0.9:
            ai_core += 1
        elif r >= 0.55:
            strong_tech += 1
        elif r >= 0.4:
            weak_tech += 1
        else:
            off_target += 1

    def in_band(f):
        y = f["raw"].get("yoe")
        return isinstance(y, (int, float)) and 5.0 <= y <= 9.0

    # Funnel: each stage is a strict subset of the one above it.
    s_ai = [f for f in feats if f["role"] >= 0.9 and not f["is_honeypot"]]
    s_india = [f for f in s_ai if (f["raw"].get("country") or "") == "India"]
    s_band = [f for f in s_india if in_band(f)]
    s_clean = [f for f in s_band if not (f["services_only"] or f["cv_dominant"] or f["pure_research"] or f["title_chaser"])]
    funnel = [
        ["All candidates", n],
        ["Genuine AI/ML role", len(s_ai)],
        ["...based in India", len(s_india)],
        ["...in the 5-9y band", len(s_band)],
        ["...no JD disqualifier", len(s_clean)],
        ["Selected top 100", min(100, n)],
    ]

    # Score distribution across the pool (display_score is normalized to [0,1]).
    edges = [i / 20 for i in range(21)]
    counts = [0] * 20
    for f in feats:
        b = min(19, int(f["display_score"] * 20))
        counts[b] += 1
    score_hist = {"labels": [f"{int(edges[i]*100)}" for i in range(20)], "counts": counts}

    return {
        "n_total": n,
        "india_total": india,
        "india_pct": round(100 * india / n),
        "honeypots_total": honey,
        "role_classes": [
            ["Genuine AI/ML", ai_core],
            ["Strong tech (SWE/Data)", strong_tech],
            ["Weak/other tech", weak_tech],
            ["Off-target / non-tech", off_target],
        ],
        "funnel": funnel,
        "score_hist": score_hist,
    }


def _sim_pool(feats, k):
    """Compact per-candidate features for the in-browser weight simulator.

    The pool is the union of the top candidates by *final* score (so the real
    top-100 is always present and the default scenario reproduces production) and
    the top candidates by *base* score (so candidates the gates currently suppress
    - honeypots, off-target profiles - are included and visibly flood back in when
    you toggle a gate off)."""
    final_rank = {f["candidate_id"]: i for i, f in enumerate(feats)}
    by_base = sorted(feats, key=lambda f: -f["base"])
    chosen, seen = [], set()
    for f in feats[: k // 2] + by_base[:k]:
        cid = f["candidate_id"]
        if cid in seen:
            continue
        seen.add(cid)
        chosen.append(f)
        if len(chosen) >= k:
            break

    pool = []
    for f in chosen:
        idx = final_rank[f["candidate_id"]]
        raw = f["raw"]
        pool.append({
            # base components (short keys to keep the file small)
            "role": round(f["role"], 4), "skills": round(f["skills"], 4),
            "narrative": round(f["narrative"], 4), "product": round(f["product"], 4),
            "experience": round(f["experience"], 4), "location": round(f["location"], 4),
            "education": round(f["education"], 4),
            # multipliers (fixed; independent of component weights)
            "g": round(f["role_gate"], 4), "b": round(f["behavioral_modifier"], 4),
            "d": round(f["disqualifier_penalty"], 4), "h": 1 if f["is_honeypot"] else 0,
            # facts for live KPI recompute
            "t": raw.get("title"), "c": (raw.get("location") or "?").split(",")[0].strip(),
            "co": raw.get("country"), "y": raw.get("yoe"),
            "no": raw.get("notice_period_days"), "orig": idx,  # original rank index
        })
    return pool


def build(candidates_path: str, out_html: str, top_n: int = 100) -> None:
    csv_path = os.path.join(os.path.dirname(out_html) or ".", "submission.csv")
    feats = rank_all(candidates_path)
    write_submission(feats, csv_path, top_n=top_n)
    top = feats[:top_n]

    payload = {
        "records": _records(top),
        "agg": _aggregates(_records(top)),
        "pool": _pool_analysis(feats),
        "sim": _sim_pool(feats, _SIM_POOL),
        "weights": {k: C.COMPONENT_WEIGHTS[k] for k in _COMPONENTS},
        "honeypot_score": C.HONEYPOT_SCORE,
    }
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = _TEMPLATE.replace("/*__DATA__*/", blob)
    with open(out_html, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[dashboard] wrote {out_html} ({len(top)} top, {len(payload['sim'])} sim pool, {len(html)//1024} KB)")


# --------------------------------------------------------------------------- #
# Single-file HTML template (vanilla JS + inline SVG; no external dependencies)
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Redrob - Candidate Ranking Dashboard</title>
<style>
  /* Light theme: warm off-white canvas, clay-coral accent palette. */
  :root{
    --bg:#F0EEE6; --panel:#FBFAF7; --panel2:#F2EFE7; --line:#E3DFD3;
    --txt:#1F1E1C; --mut:#7C7669; --accent:#C9603F; --accent2:#D97757;
    --good:#3E7D5A; --warn:#B8862F; --bad:#BF4D43;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font:14px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased}
  .serif{font-family:Georgia,"Times New Roman",ui-serif,serif}
  header{padding:26px 30px 20px;border-bottom:1px solid var(--line);display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;background:var(--panel)}
  header h1{font-family:Georgia,"Times New Roman",ui-serif,serif;font-size:25px;margin:0;font-weight:600;letter-spacing:-.2px}
  header .dot{color:var(--accent2)}
  header .sub{color:var(--mut);font-size:13.5px}
  .wrap{padding:22px 30px 60px;max-width:1500px;margin:0 auto}
  .kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:18px}
  .kpi{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:15px 17px;box-shadow:0 1px 2px rgba(60,50,30,.04)}
  .kpi .v{font-size:24px;font-weight:680;letter-spacing:-.3px}
  .kpi .l{color:var(--mut);font-size:12px;margin-top:3px}
  .kpi .v.good{color:var(--good)} .kpi .v.bad{color:var(--bad)}
  .grid{display:grid;grid-template-columns:1.15fr 1fr;gap:16px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin-bottom:16px;box-shadow:0 1px 3px rgba(60,50,30,.05)}
  .card h2{font-size:12px;margin:0 0 14px;color:var(--mut);font-weight:650;letter-spacing:.6px;text-transform:uppercase}
  table{width:100%;border-collapse:collapse}
  th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);font-size:13px;white-space:nowrap}
  th{color:var(--mut);font-weight:650;position:sticky;top:0;background:var(--panel)}
  tbody tr{cursor:pointer}
  tbody tr:hover{background:var(--panel2)}
  tbody tr.sel{background:#F6E4DA;outline:1px solid var(--accent2)}
  .tablebox{max-height:560px;overflow:auto}
  .bar{height:8px;background:var(--panel2);border-radius:6px;overflow:hidden;min-width:70px;border:1px solid var(--line)}
  .bar>span{display:block;height:100%;background:linear-gradient(90deg,#D97757,#C9603F)}
  .rank{color:var(--mut);width:30px}
  .pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:11px;border:1px solid var(--line);color:var(--mut);background:var(--panel2)}
  .pill.good{color:var(--good);border-color:#bcd8c7;background:#eef6f0}
  .pill.bad{color:var(--bad);border-color:#e8c4c0;background:#fbeeec}
  .comp{display:flex;align-items:center;gap:10px;margin:7px 0}
  .comp .nm{width:96px;color:var(--mut);font-size:12px;text-transform:capitalize}
  .comp .bar{flex:1}
  .comp .pc{width:42px;text-align:right;color:var(--txt);font-variant-numeric:tabular-nums}
  .muted{color:var(--mut)}
  .chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
  .reason{background:#F6EFE3;border-left:3px solid var(--accent2);padding:11px 13px;border-radius:8px;margin-top:12px;color:#3a352c}
  .facts{display:grid;grid-template-columns:1fr 1fr;gap:5px 16px;margin-top:12px;font-size:12px}
  .facts div span{color:var(--mut)}
  .chart .row{display:flex;align-items:center;gap:10px;margin:5px 0}
  .chart .lab{width:150px;color:var(--mut);font-size:12px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .chart .hb{flex:1;height:15px;background:var(--panel2);border-radius:5px;overflow:hidden;border:1px solid var(--line)}
  .chart .hb>span{display:block;height:100%;background:linear-gradient(90deg,#D9A15B,#C9603F)}
  .chart .ct{width:34px;color:var(--txt);font-size:12px;font-variant-numeric:tabular-nums}
  .two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .gate{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--line);font-size:12px}
  .gate b{font-variant-numeric:tabular-nums}
  .hist{display:flex;align-items:flex-end;gap:2px;height:120px;padding-top:6px}
  .hist .col{flex:1;background:linear-gradient(180deg,#D97757,#C9603F);border-radius:3px 3px 0 0;min-height:1px;transition:filter .1s}
  .hist .col:hover{filter:brightness(1.08)}
  button{background:var(--accent);color:#fff;border:0;padding:8px 14px;border-radius:9px;cursor:pointer;font-size:13px;font-weight:550}
  button:hover{background:var(--accent2)}
  .btnrow{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
  .btn2{background:var(--panel);color:var(--txt);border:1px solid var(--line);padding:7px 12px;border-radius:20px;cursor:pointer;font-size:12.5px;font-weight:500}
  .btn2:hover{border-color:var(--accent2);color:var(--accent)}
  .btn2.on{background:var(--accent);color:#fff;border-color:var(--accent)}
  .slider{display:flex;align-items:center;gap:10px;margin:9px 0}
  .slider .nm{width:90px;color:var(--mut);text-transform:capitalize;font-size:12px}
  .slider input{flex:1}
  .slider .val{width:38px;text-align:right;font-variant-numeric:tabular-nums}
  input[type=range]{accent-color:var(--accent2)}
  @media(max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}.two{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>Redrob<span class="dot">.</span> Candidate Ranking</h1>
  <span class="sub">Top-100 for the &ldquo;Senior AI Engineer &mdash; Founding Team&rdquo; JD &middot; click any row for the score breakdown</span>
</header>
<div class="wrap">
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div>
      <div class="card">
        <h2>Ranked candidates</h2>
        <div class="tablebox">
          <table>
            <thead><tr><th>#</th><th>Title</th><th>Exp</th><th>Location</th><th style="width:120px">Score</th></tr></thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </div>
    </div>
    <div>
      <div class="card" id="detail"><h2>Candidate detail</h2><div class="muted">Select a candidate from the list.</div></div>
    </div>
  </div>

  <div class="card">
    <h2>Distributions across the top 100</h2>
    <div class="two">
      <div><div class="muted" style="margin-bottom:6px">Current title</div><div class="chart" id="ch_titles"></div></div>
      <div><div class="muted" style="margin-bottom:6px">Location (city)</div><div class="chart" id="ch_cities"></div></div>
    </div>
    <div class="two" style="margin-top:14px">
      <div><div class="muted" style="margin-bottom:6px">Years of experience</div><div class="chart" id="ch_exp"></div></div>
      <div><div class="muted" style="margin-bottom:6px">Notice period (days)</div><div class="chart" id="ch_notice"></div></div>
    </div>
  </div>

  <div class="card">
    <h2>Dataset analysis &mdash; the full 100K pool</h2>
    <div class="kpis" id="pkpis" style="grid-template-columns:repeat(4,1fr)"></div>
    <div class="two" style="margin-top:14px">
      <div><div class="muted" style="margin-bottom:6px">Selection funnel (100K &rarr; 100)</div><div class="chart" id="ch_funnel"></div></div>
      <div><div class="muted" style="margin-bottom:6px">Role composition of the pool</div><div class="chart" id="ch_roles"></div></div>
    </div>
    <div style="margin-top:16px">
      <div class="muted" style="margin-bottom:6px">Final-score distribution across the pool (normalized 0&ndash;100, log-scaled height)</div>
      <div id="ch_scores" class="hist"></div>
    </div>
  </div>

  <div class="card">
    <h2>Simulation &mdash; tune the ranker, watch it re-rank live</h2>
    <div class="muted" style="margin-bottom:12px">Click a scenario or drag the weights. The top 100 of the <span id="simPool"></span> strongest contenders recomputes instantly:
      score = (&Sigma; weight&middot;component) &times; role-gate &times; availability &times; penalties &times; honeypot-gate.</div>
    <div class="muted" style="font-size:12px;margin-bottom:6px">Scenarios (weight presets)</div>
    <div class="btnrow" id="presets"></div>
    <div class="muted" style="font-size:12px;margin-bottom:6px">Toggle the gates &amp; modifiers (off = set multiplier to 1)</div>
    <div class="btnrow" id="toggles"></div>
    <div class="two">
      <div id="sliders"></div>
      <div>
        <div class="kpis" id="simkpis" style="grid-template-columns:repeat(3,1fr)"></div>
        <div style="margin-top:12px"><button id="reset">Reset to production defaults</button></div>
      </div>
    </div>
    <div class="two" style="margin-top:16px">
      <div><div class="muted" style="margin-bottom:6px">New top-100 titles</div><div class="chart" id="sim_titles"></div></div>
      <div><div class="muted" style="margin-bottom:6px">New top 20</div>
        <div class="tablebox" style="max-height:320px"><table><thead><tr><th>#</th><th>Title</th><th>Exp</th><th>City</th></tr></thead><tbody id="sim_rows"></tbody></table></div>
      </div>
    </div>
  </div>
</div>

<script>
const PAYLOAD = /*__DATA__*/;
const R = PAYLOAD.records, A = PAYLOAD.agg;
const esc = s => (s==null?"":String(s)).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
const pct = v => Math.round((v||0)*100);

// KPIs
const kpis = [
  ["Candidates", A.total, ""],
  ["India", A.india_pct + "%", "good"],
  ["Honeypots in top 100", A.honeypots, A.honeypots===0?"good":"bad"],
  ["Avg experience", (A.avg_yoe??"-") + " y", ""],
  ["Avg notice", (A.avg_notice??"-") + " d", ""],
  ["Top title", (A.titles[0]?A.titles[0][0]:"-"), ""],
];
document.getElementById("kpis").innerHTML = kpis.map(k=>
  `<div class="kpi"><div class="v ${k[2]}">${esc(k[1])}</div><div class="l">${esc(k[0])}</div></div>`).join("");

// Table
document.getElementById("rows").innerHTML = R.map(r=>`
  <tr data-r="${r.rank}">
    <td class="rank">${r.rank}</td>
    <td>${esc(r.title)}</td>
    <td>${r.yoe==null?"-":r.yoe.toFixed(1)}</td>
    <td class="muted">${esc((r.location||"").split(",")[0])}</td>
    <td><div style="display:flex;align-items:center;gap:8px">
      <div class="bar" style="flex:1"><span style="width:${pct(r.score)}%"></span></div>
      <span style="width:42px;text-align:right;font-variant-numeric:tabular-nums">${r.score.toFixed(3)}</span>
    </div></td>
  </tr>`).join("");

function flagPills(f){
  const out=[];
  if(f.is_honeypot) out.push('<span class="pill bad">honeypot</span>');
  if(f.services_only) out.push('<span class="pill bad">services-only</span>');
  if(f.cv_dominant) out.push('<span class="pill bad">CV/speech-heavy</span>');
  if(f.pure_research) out.push('<span class="pill bad">research-only</span>');
  if(f.title_chaser) out.push('<span class="pill bad">job-hopper</span>');
  if(out.length===0) out.push('<span class="pill good">no red flags</span>');
  return out.join(" ");
}

function showDetail(rank){
  const r = R.find(x=>x.rank===rank); if(!r) return;
  document.querySelectorAll("#rows tr").forEach(tr=>tr.classList.toggle("sel", +tr.dataset.r===rank));
  const comp = Object.entries(r.components).map(([k,v])=>`
    <div class="comp"><div class="nm">${esc(k)}</div>
      <div class="bar"><span style="width:${pct(v)}%"></span></div>
      <div class="pc">${pct(v)}</div></div>`).join("");
  const fc = r.facts;
  const detail = `
    <h2>#${r.rank} &middot; ${esc(r.title)}</h2>
    <div style="font-size:16px;font-weight:650">${esc(r.name||r.id)} <span class="muted" style="font-weight:400">&middot; ${esc(r.id)}</span></div>
    <div class="muted">${esc(r.company||"")} &middot; ${esc(r.location||"")}, ${esc(r.country||"")} &middot; ${r.yoe==null?"-":r.yoe.toFixed(1)} yrs &middot; score ${r.score.toFixed(3)}</div>
    <div class="chips">${flagPills(r.flags)}</div>
    <div class="reason">${esc(r.reasoning)}</div>
    <h2 style="margin-top:16px">Base components</h2>
    ${comp}
    <h2 style="margin-top:16px">Multipliers</h2>
    <div class="gate"><span>Base score</span><b>${r.base.toFixed(3)}</b></div>
    <div class="gate"><span>Role gate</span><b>&times;${r.gates.role_gate}</b></div>
    <div class="gate"><span>Behavioral modifier (avail.)</span><b>&times;${r.gates.behavioral_modifier}</b></div>
    <div class="gate"><span>Disqualifier penalty</span><b>&times;${r.gates.disqualifier_penalty}</b></div>
    <div class="gate"><span><b>Final score</b></span><b>${r.score.toFixed(3)}</b></div>
    <div class="facts">
      <div><span>Recruiter response</span> ${fc.response_rate==null?"-":pct(fc.response_rate)+"%"}</div>
      <div><span>Last active</span> ${fc.months_inactive==null?"-":fc.months_inactive+" mo ago"}</div>
      <div><span>Notice period</span> ${fc.notice_period_days==null?"-":fc.notice_period_days+" d"}</div>
      <div><span>Open to work</span> ${fc.open_to_work?"yes":"no"}</div>
      <div><span>Saved by recruiters</span> ${fc.saved_by_recruiters_30d??"-"}</div>
      <div><span>Willing to relocate</span> ${fc.willing_to_relocate?"yes":"no"}</div>
    </div>`;
  document.getElementById("detail").innerHTML = detail;
}
document.getElementById("rows").addEventListener("click", e=>{
  const tr = e.target.closest("tr"); if(tr) showDetail(+tr.dataset.r);
});

// Charts
function barChart(el, pairs){
  const max = Math.max(1, ...pairs.map(p=>p[1]));
  el.innerHTML = pairs.map(p=>`
    <div class="row"><div class="lab" title="${esc(p[0])}">${esc(p[0])}</div>
      <div class="hb"><span style="width:${Math.round(100*p[1]/max)}%"></span></div>
      <div class="ct">${p[1]}</div></div>`).join("");
}
barChart(document.getElementById("ch_titles"), A.titles);
barChart(document.getElementById("ch_cities"), A.cities);
barChart(document.getElementById("ch_exp"), A.exp_hist.labels.map((l,i)=>[l, A.exp_hist.counts[i]]));
barChart(document.getElementById("ch_notice"), A.notice_hist.labels.map((l,i)=>[l, A.notice_hist.counts[i]]));

showDetail(1);

/* ---------- Dataset analysis (full pool) ---------- */
const P = PAYLOAD.pool;
const pk = [
  ["Total candidates", P.n_total.toLocaleString(), ""],
  ["Based in India", P.india_total.toLocaleString()+" ("+P.india_pct+"%)", ""],
  ["Honeypots in pool", P.honeypots_total, ""],
  ["Genuine AI/ML roles", P.role_classes[0][1].toLocaleString(), "good"],
];
document.getElementById("pkpis").innerHTML = pk.map(k=>
  `<div class="kpi"><div class="v ${k[2]}">${esc(k[1])}</div><div class="l">${esc(k[0])}</div></div>`).join("");
barChart(document.getElementById("ch_funnel"), P.funnel);
barChart(document.getElementById("ch_roles"), P.role_classes);
const sh = P.score_hist, smx = Math.max(1, ...sh.counts);
document.getElementById("ch_scores").innerHTML = sh.counts.map((c,i)=>
  `<div class="col" title="score ~${sh.labels[i]}: ${c.toLocaleString()} candidates" style="height:${Math.max(1,Math.round(100*Math.log10(c+1)/Math.log10(smx+1)))}%"></div>`).join("");

/* ---------- Simulation (live re-rank: presets + gate toggles + sliders) ---------- */
const SIM = PAYLOAD.sim, W0 = PAYLOAD.weights, HS = PAYLOAD.honeypot_score;
const COMPS = ["role","skills","narrative","product","experience","location","education"];
let W = Object.assign({}, W0);
const GATES = {gate:true, avail:true, pen:true, honey:true};  // multipliers on/off

// Weight-preset scenarios.
const PRESETS = {
  "Production default": Object.assign({}, W0),
  "Skills-only (stuffer test)": {role:0,skills:0.40,narrative:0,product:0,experience:0,location:0,education:0},
  "Narrative-only": {role:0,skills:0,narrative:0.40,product:0,experience:0,location:0,education:0},
  "Role-blind": Object.assign({}, W0, {role:0}),
  "Location-blind": Object.assign({}, W0, {location:0}),
  "Experience-blind": Object.assign({}, W0, {experience:0}),
  "Equal weights": {role:.14,skills:.14,narrative:.14,product:.14,experience:.15,location:.15,education:.14},
};
const TOGGLE_DEFS = [
  ["gate","Role gate"], ["avail","Availability modifier"],
  ["pen","Disqualifier penalties"], ["honey","Honeypot gate"],
];

const slidersEl = document.getElementById("sliders");
function renderSliders(){
  slidersEl.innerHTML = COMPS.map(c=>`
    <div class="slider"><div class="nm">${c}</div>
      <input type="range" min="0" max="0.4" step="0.01" value="${W[c]}" data-c="${c}">
      <div class="val" id="v_${c}">${W[c].toFixed(2)}</div></div>`).join("");
}
renderSliders();
document.getElementById("simPool").textContent = SIM.length.toLocaleString();

document.getElementById("presets").innerHTML = Object.keys(PRESETS).map((p,i)=>
  `<button class="btn2${i===0?' on':''}" data-preset="${esc(p)}">${esc(p)}</button>`).join("");
document.getElementById("toggles").innerHTML = TOGGLE_DEFS.map(([k,l])=>
  `<button class="btn2 on" data-gate="${k}">${esc(l)}: on</button>`).join("");

function simulate(){
  const N = 100;
  const scored = SIM.map(s=>{
    let base = 0; for(const c of COMPS) base += W[c]*s[c];
    let score;
    if(s.h && GATES.honey){ score = HS*base; }
    else { score = base * (GATES.gate?s.g:1) * (GATES.avail?s.b:1) * (GATES.pen?s.d:1); }
    return {s, score};
  });
  scored.sort((a,b)=>b.score-a.score);
  const top = scored.slice(0, N).map(x=>x.s);
  const avg = a => a.length ? a.reduce((p,q)=>p+q,0)/a.length : 0;
  const india = top.filter(s=>s.co==="India").length;
  const ys = top.filter(s=>typeof s.y==="number").map(s=>s.y);
  const hp = top.filter(s=>s.h).length;
  const off = top.filter(s=>s.role<0.4).length;        // off-target / non-tech infiltration
  const kept = top.filter(s=>s.orig<100).length;
  const tc = {}; top.forEach(s=>{tc[s.t]=(tc[s.t]||0)+1;});
  const titleArr = Object.entries(tc).sort((a,b)=>b[1]-a[1]);
  const kpis = [
    ["India", Math.round(100*india/N)+"%", india>=90?"good":(india>=70?"":"bad")],
    ["Avg exp", avg(ys).toFixed(1)+"y", ""],
    ["Off-target", off+"%", off?"bad":"good"],
    ["Honeypots", hp, hp?"bad":"good"],
    ["Kept vs prod", kept+"%", kept>=90?"good":""],
    ["Top title", titleArr[0]?titleArr[0][0]:"-", ""],
  ];
  document.getElementById("simkpis").innerHTML = kpis.map(k=>
    `<div class="kpi"><div class="v ${k[2]}" style="font-size:17px">${esc(k[1])}</div><div class="l">${esc(k[0])}</div></div>`).join("");
  barChart(document.getElementById("sim_titles"), titleArr.slice(0,8));
  document.getElementById("sim_rows").innerHTML = top.slice(0,20).map((s,i)=>
    `<tr><td class="rank">${i+1}</td><td>${esc(s.t)}</td><td>${s.y==null?"-":(+s.y).toFixed(1)}</td><td class="muted">${esc(s.c)}</td></tr>`).join("");
}

function applyPreset(name){
  W = Object.assign({}, PRESETS[name]);
  renderSliders();
  document.querySelectorAll("#presets .btn2").forEach(b=>b.classList.toggle("on", b.dataset.preset===name));
  simulate();
}
document.getElementById("presets").addEventListener("click", e=>{
  const b = e.target.closest("[data-preset]"); if(b) applyPreset(b.dataset.preset);
});
document.getElementById("toggles").addEventListener("click", e=>{
  const b = e.target.closest("[data-gate]"); if(!b) return;
  const k = b.dataset.gate; GATES[k] = !GATES[k];
  b.classList.toggle("on", GATES[k]);
  b.textContent = TOGGLE_DEFS.find(t=>t[0]===k)[1] + ": " + (GATES[k]?"on":"off");
  simulate();
});
slidersEl.addEventListener("input", e=>{
  const c = e.target.dataset.c; if(!c) return;
  W[c] = +e.target.value; document.getElementById("v_"+c).textContent = W[c].toFixed(2);
  document.querySelectorAll("#presets .btn2").forEach(b=>b.classList.remove("on"));
  simulate();
});
document.getElementById("reset").addEventListener("click", ()=>{
  Object.keys(GATES).forEach(k=>GATES[k]=true);
  document.querySelectorAll("#toggles .btn2").forEach(b=>{b.classList.add("on");
    b.textContent = TOGGLE_DEFS.find(t=>t[0]===b.dataset.gate)[1]+": on";});
  applyPreset("Production default");
});
simulate();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a static HTML dashboard for the ranking.")
    ap.add_argument("--candidates", default="./candidates.jsonl")
    ap.add_argument("--out", default="./dashboard.html")
    ap.add_argument("--top", type=int, default=100)
    args = ap.parse_args()
    build(args.candidates, args.out, top_n=args.top)


if __name__ == "__main__":
    main()
