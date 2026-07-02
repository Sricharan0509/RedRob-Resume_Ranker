# Redrob Hackathon — Multi-Stage RAG Candidate Ranker

Ranks 100,000 candidate profiles against the Redrob *Senior AI Engineer* job
description and emits the top 100 as a submission CSV.

The system is a **local, CPU-only, offline** Retrieval-Augmented ranking
pipeline: dense bi-encoder → local vector store → BM25 → structured
"recruiter-brain" scoring → cross-encoder rerank. Nothing on the ranking path
opens a network socket, and the ranking step completes well within the 5-minute
budget using pre-built artifacts.

- **Deep-dive on the pipeline & every scoring weight:** [docs/PIPELINE.md](docs/PIPELINE.md) and [docs/SCORING.md](docs/SCORING.md)
- **Submission metadata:** [submission_metadata.yaml](submission_metadata.yaml)

---

## Problem statement

Rank 100,000 candidate profiles against a fixed *Senior AI Engineer* JD and emit
the top 100 — under a **5-minute, CPU-only, offline** rank-time budget, on a
16 GB machine. The pool also contains deliberately impossible "honeypot"
profiles (~80 per [docs/submission_spec.docx](docs/submission_spec.docx)) that
must be filtered out by general logic, without special-casing individual records.

---

## Quick start

All commands are run **from the repository root**. Requires Python 3.11+.

```bash
pip install -r requirements.txt
```

> **Important — artifacts are tied to a specific `candidates.jsonl`.**
> The pre-built files in [artifacts/](artifacts/) (dense embeddings, BM25 scores,
> id order) were computed from one exact input file. **Any time you use a new or
> updated `candidates.jsonl`, rebuild the artifacts first** — otherwise new
> candidates would be scored against stale vectors. Pick the path that matches
> your situation below.

### Path A — Run on a new / updated `candidates.jsonl` (rebuild first)

Use this whenever the input file is not the exact one the shipped artifacts were
built from. Two steps:

```bash
# 1) Pre-compute artifacts for THIS file  (offline; may exceed 5 min — allowed)
python build_index.py --candidates ./data/candidates.jsonl

# 2) Produce the submission CSV  (CPU-only, no network, < 5 min)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

Step 1 rebuilds and overwrites everything in `artifacts/` for the given file;
step 2 is the budgeted ranking step that writes [submission.csv](submission.csv).
Replace `./data/candidates.jsonl` with your file's path in **both** commands.

### Path B — Reproduce our exact submission (artifacts already match)

If you are using the **same** `candidates.jsonl` our shipped artifacts were built
from, the build step is unnecessary — just run the single ranking command:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

This loads the committed [artifacts/](artifacts/) and reproduces our exact
[submission.csv](submission.csv) within the 5-minute budget.

### Validate the output

```bash
python scripts/validate_submission.py ./submission.csv
```

---

## Notes on the two steps

- **`build_index.py` (pre-computation, unbudgeted).** Parses every profile, encodes
  the dense + BM25 channels, and writes them to `artifacts/`. The rules explicitly
  allow this step to exceed 5 minutes. It **overwrites** any existing artifacts, so
  rerunning it on a new file is always safe. `artifacts/build_meta.json` records
  which encoder and how many candidates (`n`) produced the current artifacts — a
  quick way to confirm they match your input.
- **`rank.py` (ranking, budgeted < 5 min).** Loads `artifacts/` and produces the
  CSV. If `artifacts/` is missing entirely it transparently falls back to computing
  the channels in-budget, so the pipeline still runs on any machine — but for a new
  input the **recommended** path is to run `build_index.py` first (Path A), which
  keeps the ranking step fast and the scores correct.

---

## Repository layout

```
.
├── rank.py                     # ← THE reproduce entrypoint (candidates.jsonl → submission.csv)
├── build_index.py              # offline pre-computation of retrieval artifacts (unbudgeted)
├── app.py                      # Streamlit sandbox for a small sample (demo)
├── submission.csv              # the produced top-100 submission
├── requirements.txt            # pinned dependencies
├── submission_metadata.yaml    # portal metadata 
│
├── src/                        # source package
│   ├── config.py               # single source of truth: all weights & thresholds
│   ├── reasoning.py            # deterministic (non-LLM) reasoning strings
│   ├── data_processing/        # parsing, features, behavioral, honeypot
│   ├── retrieval/              # dense encoder + vector store, BM25, hybrid fusion
│   └── reranking/              # recruiter-brain composite + cross-encoder
│
├── artifacts/                  # pre-computed retrieval artifacts (loaded at rank time)
│   ├── dense.f16.npy           # (100K, dim) profile embeddings
│   ├── ids.npy                 # candidate_id order aligned to dense.f16.npy
│   ├── bm25_scores.npy         # BM25 scores vs the fixed JD query
│   ├── jd_dense.npy            # mean-pooled JD query vector
│   └── build_meta.json         # which encoder/backend produced the artifacts
│
├── data/                       # inputs & samples
│   ├── candidates.jsonl        # official 100K input — NOT committed (see .gitignore)
│   ├── candidate_schema.json
│   ├── sample_candidates.json
│   └── sample_submission.csv
│
├── docs/
│   ├── PIPELINE.md             # architecture / pipeline write-up
│   ├── SCORING.md              # exact formula, weights, and rejected approaches
│   ├── job_description.docx    # original hackathon briefs
│   ├── redrob_signals_doc.docx
│   └── submission_spec.docx
│
└── scripts/
    └── validate_submission.py  # checks the CSV against the submission spec
```

---

## Sandbox demo

```bash
streamlit run app.py
```

Point it at a small (≤ a few hundred) candidate sample — e.g.
`data/sample_candidates.json` — and watch the multi-stage pipeline populate a
ranked list with grounded, per-phase score breakdowns.

---

## Design decisions & trade-offs

The choices that shaped the pipeline, and what each one cost:

- **Hashing dense-encoder by default, not semantic BGE.** `DENSE_ENCODER_MODE =
  "hash"` ([src/config.py](src/config.py)). Real BGE-small encodes 100K profiles
  in ~2.75h on our CPU; the deterministic hashing encoder builds the same index
  in ~5 min. We accepted a weaker dense channel because `S_struct` (0.62 weight)
  + BM25 + the Phase-4 cross-encoder carry ranking quality. The real BGE path is
  implemented and verified — flip to `"auto"` to use it (see
  [docs/PIPELINE.md](docs/PIPELINE.md), "Status of the shipped artifacts").
- **`role_gate` is blind to the skills list.** It reads title ∪ career evidence
  only ([src/data_processing/features.py](src/data_processing/features.py)) —
  that is precisely how a keyword-stuffer would otherwise climb.
- **We persist the BM25 *score vector*, not the postings index.** The tokenized
  index is ~130 MB pickled for 100K docs, over GitHub's 100 MB per-file limit;
  the JD query is fixed at build time, so the per-candidate score vector is all
  `rank.py` needs ([src/config.py](src/config.py)).
- **Cross-encoder runs on the 300-shortlist only** (`SHORTLIST_N`) — the head is
  where NDCG@10 is decided, and full-pool cross-attention would blow the budget.

## Approaches we tried and rejected

Both were removed *after measuring them against the real 100K pool* — the
negative results are part of the engineering (full detail in
[docs/SCORING.md](docs/SCORING.md)):

- **GitHub-activity as a "closed-source only" disqualifier.** Rejected:
  `years_of_experience≥5 AND github_activity_score≤0` fires on 43% of the pool
  (16% even restricted to target titles). Having no GitHub is simply common
  among strong candidates in this data — not a reliable proxy.
- **Honeypot rule "any skill duration > total career months."** Rejected: it
  flagged 13,581/100,000 (13.6%) of real candidates — skills are often
  self-taught or predate recorded career history.

## Known limitations

- The **shipped** dense channel is the hashing fallback, not semantic BGE (see
  `artifacts/build_meta.json`, `encoder_mode`).
- `trajectory_score()` is computed and attached to diagnostics but is **not**
  folded into the final score (see [docs/SCORING.md](docs/SCORING.md) and the
  note in [src/data_processing/features.py](src/data_processing/features.py)).
- Score normalization is **pool-relative** (robust min-max across the scored
  pool). On tiny samples (e.g. the Streamlit demo) it stretches extremes to
  0.0/1.0 in a way it would not at full 100K scale.

## Testing & validation approach

- [scripts/validate_submission.py](scripts/validate_submission.py) checks the
  output CSV against the submission spec before upload.
- **Determinism** is a correctness property here: everything is seeded
  (`src/config.py: SEED`), so the same input yields byte-identical output.
- Rule thresholds (honeypot, disqualifiers) were tuned by **empirical sweeps
  over the full 100K pool**, not chosen a priori — see the rejected-approach
  measurements above.

## AI assistance (transparency)

AI was used as a tool inside a human-led engineering process:

- **Starting point.** An initial multi-stage RAG + rerank architecture blueprint
  was drafted with an LLM and reviewed by the team.
- **What the team owned.** Adapting every hosted-cloud component to a
  guardrail-compliant local/offline equivalent; setting and reviewing all
  scoring weights, role-gating, and disqualifier rules; designing the honeypot
  logic; and empirically tuning and rejecting rules against the real 100K pool
  (see "Approaches we tried and rejected"). Those measured decisions are the
  team's, not the model's.
- **What AI did.** Claude Code wrote and refactored the
  implementation from the reviewed design.

See [submission_metadata.yaml](submission_metadata.yaml) for the full declaration.

---

## Guarantees

- **Deterministic.** Everything is seeded (`src/config.py: SEED`); the same input
  produces byte-identical output.
- **No network at rank time.** No candidate data is sent to any hosted LLM/API.
- **CPU-only, < 16 GB, < 5 min** for the 100K pool with the shipped artifacts.
- **No hidden steps.** The committed `submission.csv` is exactly what `rank.py`
  emits from `data/candidates.jsonl`.

> Note: `artifacts/dense.f16.npy` (~98 MB) is committed directly so the reproduce
> command works immediately. If you prefer a lighter clone, delete it and regenerate
> with `build_index.py` (or use Git LFS to track `artifacts/*.npy`).
