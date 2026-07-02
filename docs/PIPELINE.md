# RAG-based Candidate Ranker (guardrail-compliant)

A multi-stage **Retrieval-Augmented ranking** pipeline for the Redrob *Senior AI
Engineer* JD, implementing a "Hybrid Knowledge-Graph & Multi-Stage LLM
Reranking" blueprint — made competition-legal by running every model **locally,
on CPU, with no network** at rank time.

Where a naive blueprint would reach for *hosted GPT-style rerankers, managed
rerank APIs, or cloud vector databases*, this ranker substitutes the **compliant
local equivalent**: a local bi-encoder into a local vector store, and a local
cross-encoder for the recruiter-style rerank. Nothing in the `rank.py` path
opens a socket.

## The pipeline (four phases)

```
candidates.jsonl
   │
   ▼
Phase 1  Parsing & feature extraction            src/data_processing/
         profile_doc (career-weighted) · structured "recruiter brain"
         rubric · trajectory score · honeypot impossibility gate
   │
   ▼
Phase 2  Dual-engine retrieval                   src/retrieval/
         dense bi-encoder → local vector store (Chroma/numpy)   S_dense
         BM25 over hard-constraint JD terms                     S_lex
         fuse → Sim_hybrid
   │
   ▼
Phase 3  Recruiter-brain scoring matrix          src/reranking/recruiter_brain.py
         recruiter_fit = 0.62·S_struct + 0.38·Sim_hybrid
         × role_gate (title ∪ career, never skills)
         × behavioral_mult (availability)  × honeypot_kill
   │
   ▼  top-300 shortlist
Phase 4  Cross-encoder rerank                     src/reranking/cross_encoder.py
         local ms-marco-MiniLM cross-attention JD × profile     S_cross
         fit = 0.5·recruiter_fit + 0.5·S_cross
   │
   ▼
top 100 → monotonic + tie-break → grounded reasoning → submission.csv
```

## Status of the shipped artifacts

The dense channel deliberately uses the **pure-numpy hashing encoder**, set by
`DENSE_ENCODER_MODE = "hash"` in [`src/config.py`](../src/config.py). This builds
the full-100K index (`artifacts/dense.f16.npy`) in ~5 min on CPU with no model
download and full determinism — confirm via `encoder_mode` in
[`artifacts/build_meta.json`](../artifacts/build_meta.json). The real BGE-small
path exists and is verified working (the `USE_TF=0` / `max_seq_length` /
thread-count fixes are in place), but encoding 100K profiles with BGE takes
~2.75h on this CPU for a marginal ranking gain, so it is **off by default**;
`S_struct` (0.62 weight) + BM25 + the Phase-4 cross-encoder carry ranking
quality. To opt into real BGE instead, set `DENSE_ENCODER_MODE = "auto"` and
rerun `python build_index.py --candidates ./data/candidates.jsonl`.

Phase 4's cross-encoder rerank is a **separate, fast** model (ms-marco-MiniLM,
300-doc shortlist only) and is unaffected by `DENSE_ENCODER_MODE`; it runs in
real mode when `sentence-transformers` is installed and falls back to a
lexical overlap otherwise, either way inside the rank-time budget.

## Why this beats naive RAG

- **Anti-keyword-stuffer.** `role_gate` and `S_struct` are driven by **title +
  career descriptions**, never the (stuffable) skills list. A *Graphic Designer*
  padded with "Pinecone / FAISS / RAG" cannot climb.
- **Honeypot kill.** Impossible profiles (claimed YOE ≫ career timeline, "expert"
  skill with 0 months, etc.) are detected by contradiction and forced out —
  general logic, no ID special-casing.
- **Availability-aware.** A perfect-on-paper but 6-month-dormant, 5%-response
  candidate is multiplicatively down-weighted (~0.60×), not deleted.
- **Head-sharpened.** The local cross-encoder runs only on the 300-shortlist,
  where NDCG@10 is won — the one good idea from the LLM-rerank school, kept
  inside the compute box.

## Rules we tried and rejected (measured, not assumed)

Two candidate rules were removed only after measuring them against the real 100K
pool — the negative results are part of the design:

- A GitHub-activity disqualifier fired on 43% of the pool (16% even restricted
  to target titles), so it was dropped as an unreliable proxy.
- A honeypot rule ("skill duration > total career months") flagged 13.6% of real
  candidates, so it was removed rather than re-thresholded.

Full measurements and reasoning are in [SCORING.md](SCORING.md) (§1 and §5).

## Reproduce

All commands are run from the repository root.

```bash
# 0) install
pip install -r requirements.txt

# 1) OFFLINE precompute (may exceed 5 min — allowed). Builds artifacts/.
python build_index.py --candidates ./data/candidates.jsonl

# 2) THE reproduce command (CPU-only, no network, < 5 min)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 3) validate before submitting
python scripts/validate_submission.py ./submission.csv
```

`rank.py` **transparently falls back** if `artifacts/` is missing or if
`sentence-transformers` / `chromadb` aren't installed: it uses a deterministic
pure-numpy hashing encoder + in-memory BM25 + lexical-overlap rerank, so the CSV
still reproduces on any machine. `build_meta.json` records which encoder produced
the shipped artifacts.

## Sandbox

`streamlit run app.py` — paste/point at a ≤100-candidate sample and watch the
pipeline populate a ranked list with grounded justifications.

## Determinism & budget

Seeded throughout (`config.SEED`); same input → byte-identical output. Rank-time
budget target ~2–3 min for 100K, well under 16 GB (100K×384 f16 ≈ 73 MB for
real BGE-small; the shipped hashing-fallback artifacts are 100K×512 f16 ≈
102 MB — see "Status of the shipped artifacts" above).

All weights, thresholds, taxonomies, and JD query text live in
[`src/config.py`](../src/config.py) — the single source of truth.
