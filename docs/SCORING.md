# Scoring Reference

Quick reference for every metric in the pipeline: what it is, the exact formula,
and which file/function computes it. Source of truth for all weights/thresholds
is `src/config.py` — if a number here ever disagrees with that file, config.py
wins.

## Top-level chain

```
Sim_hybrid    = W_DENSE·S_dense_norm + W_LEX·S_lex_norm            (hybrid.py)
recruiter_fit = W_STRUCT·S_struct + W_SIM·Sim_hybrid                (recruiter_brain.py)
fit           = (1−CROSS_BLEND)·recruiter_fit + CROSS_BLEND·S_cross (shortlist rows only)
final         = role_gate · fit · behavioral_mult · honeypot_kill
```

| Symbol | Value | Meaning |
|---|---|---|
| `W_DENSE` | 0.60 | dense/semantic weight inside `Sim_hybrid` |
| `W_LEX` | 0.40 | BM25/lexical weight inside `Sim_hybrid` |
| `W_STRUCT` | 0.62 | structured-rubric weight inside `recruiter_fit` |
| `W_SIM` | 0.38 | retrieval-similarity weight inside `recruiter_fit` |
| `SHORTLIST_N` | 300 | candidates (by `recruiter_fit`) that reach the cross-encoder |
| `CROSS_BLEND` | 0.50 | cross-encoder weight inside `fit`, shortlist rows only |

Effective weight of each channel on a shortlisted candidate's **final** score:
`S_struct` 31%, `S_cross` 50%, `S_dense` 11.4%, `S_lex` 7.6%. Below the
shortlist cutoff, `S_cross` never applies and `fit == recruiter_fit`.

---

## 1. `S_struct` — structured "recruiter brain" rubric
**File:** `src/data_processing/features.py: struct_score()`

```
S_struct = clamp( Σ(component_i · weight_i) − Σ(penalties), 0, 1 )
```

| Component | Weight | Measures | Formula |
|---|---|---|---|
| `role_fit` | 0.28 | Title/trajectory seniority & relevance | Branches on `title_class`: nontech→0.05 · ai→`min(1.0, 0.85+0.05·ai_role_count)` · ds→0.50/0.68/0.82 by ML-signal count · adjacent→0.66 if retrieval-build evidence else 0.28/0.42/0.55 · unknown→0.18/0.40 |
| `career_evidence` | 0.24 | What they actually built | `min(1.0, ml_signal_count/14)` |
| `skill_trust` | 0.18 | Trust-weighted JD-critical skills | per matched skill: `prof_w · (0.5+0.5·endorsement_f) · (0.4+0.6·duration_f)`, ×1.2 if assessment score ≥70 backs it, ×0.4 if "expert" @ 0 months; summed, `min(1.0, total/4.0)` |
| `experience_fit` | 0.12 | Soft curve, peak 6–8y | step: <2y→0.20, <4y→0.55, <5y→0.78, 5–9y→1.00, 9–12y→0.85, >12y→0.65 |
| `eval_framework` | 0.08 | NDCG/MRR/MAP/A-B literacy | 1.0 strong terms (ndcg/mrr/learning-to-rank/a-b test) · 0.5 weak terms (map/precision/recall) · 0.0 none |
| `product_company` | 0.10 | Product co. vs consulting | `min(1.0, 0.35+0.22·tech_industry_count)`, ×0.5 if all-consulting/zero-tech |

**Penalties** (subtracted, each capped, from `PENALTIES`):

| Penalty | Amount | Trigger |
|---|---|---|
| `cv_speech_robotics_primary` | 0.25 | ≥3 `WRONG_DOMAIN` hits, no IR/NLP counter-evidence |
| `academic_only_no_prod` | 0.30 | academic keywords present, zero production keywords |
| `langchain_recent_only` | 0.20 | LangChain/LlamaIndex mentioned, no pre-LLM IR fundamentals |
| `title_chaser` | 0.15 | ≥3 jobs averaging <18 months tenure |
| `consulting_only_career` | 0.20 | every job at a firm in `CONSULTING_FIRMS` |
| `architecture_tech_lead_stale` | 0.20 | current job (`is_current=True`) title contains "architect"/"tech lead"/"technical lead" AND `duration_months` ≥ 18 |

**Not implemented (checked, rejected):** job_description.docx's "closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source)" disqualifier has no papers/talks field anywhere in candidate_schema.json, so `github_activity_score` was tried as the open-source-validation proxy. Measured against the real 100K pool: `years_of_experience≥5 AND github_activity_score≤0` fires on 43% of the whole pool, and even restricted to the JD's actual target titles (`ai`/`ds` classes) with the strict "-1 = no GitHub linked" reading, it still fires on 16% of that population — having no GitHub is just common among otherwise-strong candidates in this dataset, not a reliable stand-in for "worked only on closed-source with zero external validation." Shipping it would have penalized a large share of genuinely good candidates on a proxy the data doesn't actually support, so it was left out rather than forced in.

**Helper variables:**
- `ml_signal_count(career)` — raw keyword-hit count of `ML_PRODUCTION_KW` (embedding, vector, retrieval, ranking, recommendation, nlp, rag, fine-tun, semantic, rerank, a/b test, index, production, deployed, scale, latency) across all career-history text
- `title_class(title)` — `nontech` / `ai` / `ds` / `adjacent` / `unknown`, from the taxonomy sets in config (`NON_TECHNICAL_TITLES`, `AI_TITLES`, `DATA_SCIENTIST_TITLES`, `ADJACENT_TITLES`)
- `_has_retrieval_build(career)` — ≥2 distinct `RETRIEVAL_BUILD_KW` hits (retrieval, ranking, recommendation, recsys, embedding, vector, semantic search, search relevance, rerank)

**Not currently wired in:** `trajectory_score()` (`0.45·stability + 0.35·experience_shape + 0.20·seniority_arc`) is computed and attached to diagnostics for reporting, but despite the module docstring's claim, it is never added or subtracted in `struct_score()` — it doesn't affect the score.

---

## 2. `Sim_hybrid` — dual-engine retrieval fusion
**File:** `src/retrieval/hybrid.py: fuse()`

```
Sim_hybrid = 0.60·S_dense_norm + 0.40·S_lex_norm
```

- **`S_dense`** — cosine similarity between the candidate's `profile_doc` embedding and the mean-pooled JD embedding. Real encoder: BGE-small (`sentence-transformers`, 384-d). Fallback: deterministic hash-of-tokens vector (512-d), used automatically whenever `sentence-transformers` fails to import.
- **`S_lex`** — BM25 score of `profile_doc` against curated `JD_BM25_TERMS` (embeddings, retrieval, ranking, vector, search, faiss, pinecone, weaviate, qdrant, elasticsearch, ndcg, mrr, map, recommendation, hybrid, reranking, production, sentence, transformers, semantic, learning, rank, bm25, recsys).
- Both raw scores go through `normalize01()` — robust min-max to `[0,1]` across the whole scoring pool, clipping the top/bottom 0.5% as an outlier guard. **This normalization is pool-relative**: on a tiny pool (e.g. 5 candidates) it stretches extremes to exact 0.0/1.0, which would not happen at full 100K scale.

---

## 3. `role_gate` — title/career gate, blind to skills
**File:** `src/data_processing/features.py: role_gate()`

Purely multiplicative. Skills array is never read here — only title and career text.

| Tier | Value | Trigger |
|---|---|---|
| Core AI title | 1.00 | `title_class` is `ai` or `ds` |
| Adjacent + evidence | 0.95 | adjacent/unknown title, but `_has_retrieval_build()` fires |
| Adjacent + some signal | 0.70 | adjacent title, `ml_signal_count ≥ 2` |
| Generic/weak | 0.30 | adjacent/unknown title, weak signal |
| Non-tech | 0.05 | `title_class == nontech` |

---

## 4. `behavioral_mult` — availability multiplier
**File:** `src/data_processing/behavioral.py`

```
behavioral_mult = clamp(0.85 + Σ adjustments, 0.55, 1.15)
```

| Signal | Adjustment |
|---|---|
| `profile.country`/`location` + `willing_to_relocate` | outside India: −0.08 if not willing to relocate · −0.03 if willing (JD: "we don't sponsor work visas" isn't resolved by willingness alone) · 0 if in India (JD is "flexible" on location within India) |
| `recruiter_response_rate` | `(rr − 0.40) · 0.25` |
| `last_active_date` | +0.08 if <30d · +0.04 if <90d · −0.08 if >180d · 0 otherwise |
| `open_to_work_flag` | +0.06 true · −0.04 false |
| `notice_period_days` | +0.05 if ≤30 · +0.02 if ≤60 · −0.05 if ≥120 |
| `interview_completion_rate` | `(icr − 0.50) · 0.10` |
| `offer_acceptance_rate` | `+ oar · 0.05` (skipped if −1 = no history) |
| `saved_by_recruiters_30d>0` or `profile_views_received_30d≥50` | +0.03 |
| verified email + phone | +0.02 |

`behavioral_mult` now takes `(signals, profile)` — the location/relocation check needs `profile.country`/`profile.location`, which live outside `redrob_signals`.

---

## 5. `honeypot_kill` — hard veto
**File:** `src/data_processing/honeypot.py: is_honeypot()` → `HONEYPOT_KILL = 0.001`

Boolean OR of 6 independent contradiction checks; **any one firing** = killed:

| # | Rule |
|---|---|
| 1 | Job start year predates plausible working age vs. stated `years_of_experience` (gap > yoe+6) |
| 2 | Any "expert" skill with <2 months usage |
| 3 | Structured `years_of_experience` diverges from summed career months by >4 years |
| 4 | Internal date contradiction: end < start, or start date in the future |
| 5 | Summary states a YOE (regex `"with X years"`) contradicting the structured field by >2.5y, backed by career history (i.e. structured field looks tampered) |
| 6 | ≥3 advanced/expert skills all claim exactly 0 months usage |

Dropped: a former rule flagging "any skill's `duration_months` > total career months" fired on 13,581/100,000 real candidates (13.6%) — skills are commonly self-taught or used before the candidate's recorded career_history entries, so this was never a reliable impossibility signal for this dataset and contradicted submission_spec.docx §7's "~80 honeypots" scale. Removed rather than re-thresholded: an empirical sweep found no `career_months`- or `years_of_experience`-based slack that both preserved the two documented example patterns (impossible tenure-vs-experience; "expert" skill with ~0 months used) and avoided false-positiving on ordinary profiles.

No individual candidate is special-cased — pure internal-consistency checks against the record's own fields.

---

## 6. `S_cross` — cross-encoder rerank (shortlist only)
**File:** `src/reranking/cross_encoder.py: CrossReranker`

- Real mode (`sentence-transformers` available): `ms-marco-MiniLM-L-6-v2` cross-attention score between `JD_CROSS_TEXT` and each shortlisted `profile_doc`, logit → `sigmoid`.
- Fallback mode (`lexical`): Jaccard-style overlap — `|JD_terms ∩ profile_doc_tokens| / |JD_terms|`, scaled by the max value observed in the shortlist.
- Only applied to the top `SHORTLIST_N=300` candidates by `recruiter_fit`; everyone else keeps `fit == recruiter_fit` unchanged.

---

## Quick lookup: which file owns which weight

| Constant | File | Line area |
|---|---|---|
| `W_DENSE`, `W_LEX`, `W_STRUCT`, `W_SIM`, `SHORTLIST_N`, `CROSS_BLEND` | `src/config.py` | Phase-2/3/4 fusion block |
| `STRUCT_WEIGHTS`, `PENALTIES` | `src/config.py` | structured rubric block |
| `ROLE_GATE_*` | `src/config.py` | role_gate tiers |
| `BEHAV_BASE`, `BEHAV_MIN`, `BEHAV_MAX` | `src/config.py` | behavioral block |
| `HONEYPOT_KILL` | `src/config.py` | honeypot block |
| title/skill/keyword taxonomies (`AI_TITLES`, `CRITICAL_SKILLS`, `WRONG_DOMAIN`, `CONSULTING_FIRMS`, etc.) | `src/config.py` | taxonomy block |
