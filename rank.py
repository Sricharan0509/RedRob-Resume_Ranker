#!/usr/bin/env python3
"""
rank.py — RAG candidate ranker, single budgeted entrypoint (CPU-only, no network, <5min).

Implements the multi-stage RAG pipeline end-to-end:

    Phase 1  parse + profile_doc + structured "recruiter brain" features + honeypot
    Phase 2  dual-engine retrieval: dense vector-store cosine + BM25  → Sim_hybrid
    Phase 3  recruiter_fit = W_STRUCT*S_struct + W_SIM*Sim_hybrid, gated + behavioral
    Phase 4  cross-encoder rerank of the top-SHORTLIST_N shortlist
    output   final blend → sort → top 100 → monotonic + tie-break → reasoning → CSV

Loads artifacts/ if present (built by build_index.py); otherwise recomputes the
dense + BM25 channels in-budget with the offline fallback encoder, so the repo
reproduces from scratch on any machine.

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from src import config as C
from src.data_processing.io_utils import stream_candidates
from src.data_processing.profile_doc import build_profile_doc
from src.data_processing.features import struct_score, role_gate
from src.data_processing.behavioral import behavioral_mult
from src.data_processing.honeypot import is_honeypot
from src.retrieval.embeddings import Encoder, build_jd_vector, load_vector_store
from src.retrieval.lexical import BM25Index, tokenize, jd_query_terms
from src.retrieval.hybrid import fuse
from src.reranking.cross_encoder import CrossReranker
from src.reranking import recruiter_brain as brain
from src.reasoning import build_reasoning

HERE = Path(__file__).resolve().parent
ART = HERE / C.ARTIFACT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — stream, parse, structured features
# ─────────────────────────────────────────────────────────────────────────────
def phase1(candidates_path):
    ids, docs, cands = [], [], []
    s_struct, gates, behav, hkill, hflag = [], [], [], [], []
    diags = []

    for i, cand in enumerate(stream_candidates(candidates_path)):
        profile = cand.get("profile", {}) or {}
        career = cand.get("career_history", []) or []
        signals = cand.get("redrob_signals", {}) or {}

        ss, diag = struct_score(cand)
        hp = is_honeypot(cand)

        ids.append(cand["candidate_id"])
        docs.append(build_profile_doc(cand))
        cands.append(cand)                     # kept compact; dropped after use
        s_struct.append(ss)
        gates.append(role_gate(profile.get("current_title", ""), career))
        behav.append(behavioral_mult(signals, profile))
        hkill.append(C.HONEYPOT_KILL if hp else 1.0)
        hflag.append(hp)
        diags.append(diag)

        if (i + 1) % 25000 == 0:
            print(f"      {i + 1:,} scored ...")

    return dict(
        ids=ids, docs=docs, cands=cands, diags=diags, hflag=hflag,
        s_struct=np.array(s_struct, dtype=np.float32),
        gate=np.array(gates, dtype=np.float32),
        behav=np.array(behav, dtype=np.float32),
        hkill=np.array(hkill, dtype=np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — dual-engine retrieval (dense vector store + BM25)
# ─────────────────────────────────────────────────────────────────────────────
def phase2(ids, docs):
    """Return (dense_raw, lex_raw) aligned to streaming order."""
    terms = jd_query_terms()
    have_dense = (ART / C.DENSE_NPY).exists() and (ART / C.IDS_NPY).exists()
    have_bm25 = (ART / C.BM25_SCORES_NPY).exists() and (ART / C.IDS_NPY).exists()

    if have_dense:
        print("      loading dense artifacts ...")
        matrix = np.load(ART / C.DENSE_NPY).astype(np.float32)
        build_ids = np.load(ART / C.IDS_NPY, allow_pickle=True)
        jd_path = ART / C.JD_DENSE_NPY
        if jd_path.exists():
            jd_vec = np.load(jd_path).astype(np.float32)
        else:
            jd_vec = build_jd_vector(Encoder())
        store = load_vector_store(matrix, build_ids)
        dense_all = store.query_all(jd_vec)                 # build order
        row = {cid: k for k, cid in enumerate(build_ids)}
        dense_raw = np.array(
            [dense_all[row[c]] if c in row else 0.0 for c in ids],
            dtype=np.float32)
    else:
        print("      no dense artifact — encoding in-budget (fallback) ...")
        enc = Encoder(prefer_st=True)
        matrix = enc.encode(docs, is_query=False)
        jd_vec = build_jd_vector(enc)
        dense_raw = (matrix.astype(np.float32) @ jd_vec).astype(np.float32)

    if have_bm25:
        print("      loading BM25 score artifact ...")
        lex_all = np.load(ART / C.BM25_SCORES_NPY).astype(np.float32)
        build_ids = np.load(ART / C.IDS_NPY, allow_pickle=True)
        row = {cid: k for k, cid in enumerate(build_ids)}
        lex_raw = np.array(
            [lex_all[row[c]] if c in row else 0.0 for c in ids],
            dtype=np.float32)
    else:
        print("      no BM25 artifact — building in-budget ...")
        bm25 = BM25Index([tokenize(d) for d in docs])
        lex_raw = bm25.scores_for_query(terms)

    return dense_raw, lex_raw


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — cross-encoder rerank of the shortlist
# ─────────────────────────────────────────────────────────────────────────────
def phase4(docs, recruiter_fit_vals, shortlist_idx):
    reranker = CrossReranker()
    print(f"      cross-encoder mode = {reranker.mode} "
          f"on {len(shortlist_idx)} shortlist docs")
    short_docs = [docs[i] for i in shortlist_idx]
    s_cross = reranker.score(short_docs)
    fit = recruiter_fit_vals.copy()
    blended = brain.blend_cross(recruiter_fit_vals[shortlist_idx], s_cross)
    fit[shortlist_idx] = blended
    return fit


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def rank(candidates_path, out_path):
    t0 = time.time()
    print("[Phase 1] parse + structured recruiter-brain features ...")
    d = phase1(candidates_path)
    n = len(d["ids"])
    print(f"      {n:,} candidates | {sum(d['hflag'])} honeypots flagged "
          f"({time.time() - t0:.0f}s)")

    print("[Phase 2] dual-engine retrieval (dense vector store + BM25) ...")
    dense_raw, lex_raw = phase2(d["ids"], d["docs"])
    sim_hybrid, s_dense, s_lex = fuse(dense_raw, lex_raw)
    print(f"      retrieval fused ({time.time() - t0:.0f}s)")

    print("[Phase 3] recruiter-brain composite + gating ...")
    rfit = brain.recruiter_fit(d["s_struct"], sim_hybrid)
    prelim = brain.finalize(rfit, d["gate"], d["behav"], d["hkill"])
    shortlist_idx = np.argsort(-prelim)[:C.SHORTLIST_N]
    print(f"      shortlist of {len(shortlist_idx)} for rerank "
          f"({time.time() - t0:.0f}s)")

    print("[Phase 4] cross-encoder rerank ...")
    fit = phase4(d["docs"], rfit, shortlist_idx)
    final = brain.finalize(fit, d["gate"], d["behav"], d["hkill"])
    print(f"      reranked ({time.time() - t0:.0f}s)")

    print("[Output] sort + top100 + monotonic/tie-break + reasoning ...")
    # Sort on the ROUNDED score (what actually gets written) so the tie-break
    # rule — equal scores ordered by candidate_id ascending — holds in the CSV.
    final_r = np.round(final, 4)
    order = sorted(range(n), key=lambda i: (-float(final_r[i]), d["ids"][i]))
    top = order[:100]

    rows = []
    for rankpos, i in enumerate(top, start=1):
        sc = float(final_r[i])
        reasoning = build_reasoning(
            d["cands"][i], d["diags"][i], rankpos, is_honeypot=d["hflag"][i])
        rows.append((d["ids"][i], rankpos, sc, reasoning))

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for cid, rk, sc, rs in rows:
            w.writerow([cid, rk, f"{sc:.4f}", rs])

    hp_in_top = sum(1 for i in top if d["hflag"][i])
    print(f"      wrote {out_path}")
    print(f"      honeypots in top100: {hp_in_top} (must be < 10)")
    print(f"      rank1  {rows[0][0]} score={rows[0][2]:.4f}")
    print(f"      rank100 {rows[-1][0]} score={rows[-1][2]:.4f}")
    print(f"DONE in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="RAG candidate ranker.")
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--out", default="./submission.csv")
    args = ap.parse_args()
    rank(args.candidates, args.out)


if __name__ == "__main__":
    main()
