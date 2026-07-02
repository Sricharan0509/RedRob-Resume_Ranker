#!/usr/bin/env python3
"""
app.py — Streamlit sandbox for the RAG candidate ranker.

Point it at a small JSONL/JSON sample (<= a few hundred candidates), watch the
multi-stage pipeline run, and inspect the ranked list with grounded
justifications and a per-phase score breakdown.

Run:
    streamlit run app.py
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.data_processing.io_utils import stream_candidates
from src.data_processing.profile_doc import build_profile_doc
from src.data_processing.features import struct_score, role_gate
from src.data_processing.behavioral import behavioral_mult
from src.data_processing.honeypot import is_honeypot
from src.retrieval.embeddings import Encoder, build_jd_vector
from src.retrieval.lexical import BM25Index, tokenize, jd_query_terms
from src.retrieval.hybrid import fuse
from src.reranking.cross_encoder import CrossReranker
from src.reranking import recruiter_brain as brain
from src.reasoning import build_reasoning

st.set_page_config(page_title="Redrob RAG Ranker", layout="wide")
st.title("🔎 Redrob — Multi-Stage RAG Candidate Ranker")
st.caption("Local bi-encoder → vector store → BM25 → recruiter-brain → cross-encoder. "
           "CPU-only, no network.")


def _load_records(uploaded, sample_path):
    if uploaded is not None:
        raw = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".json"):
            data = json.loads(raw)
            return data if isinstance(data, list) else [data]
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    if sample_path and Path(sample_path).exists():
        p = Path(sample_path)
        if p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else [data]
        return list(stream_candidates(p))
    return []


with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Candidate sample (.json or .jsonl)",
                                type=["json", "jsonl"])
    sample_path = st.text_input("…or a path", value="./data/sample_candidates.json")
    topk = st.number_input("Show top-K", min_value=1, max_value=100,
                           value=100, step=1)
    run = st.button("Rank candidates", type="primary")

if run:
    cands = _load_records(uploaded, sample_path)
    if not cands:
        st.error("No candidates loaded.")
        st.stop()
    st.success(f"Loaded {len(cands)} candidates.")

    prog = st.progress(0.0, "Phase 1 — features")
    ids, docs, diags, hflag = [], [], [], []
    s_struct, gates, behav, hkill = [], [], [], []
    for c in cands:
        ss, diag = struct_score(c)
        ids.append(c["candidate_id"])
        docs.append(build_profile_doc(c))
        diags.append(diag)
        hp = is_honeypot(c)
        hflag.append(hp)
        s_struct.append(ss)
        gates.append(role_gate(c.get("profile", {}).get("current_title", ""),
                               c.get("career_history", [])))
        behav.append(behavioral_mult(c.get("redrob_signals", {}) or {},
                                     c.get("profile", {}) or {}))
        hkill.append(C.HONEYPOT_KILL if hp else 1.0)

    prog.progress(0.35, "Phase 2 — dense + BM25 retrieval")
    enc = Encoder(prefer_st=True)
    matrix = enc.encode(docs, is_query=False)
    jd_vec = build_jd_vector(enc)
    dense_raw = matrix.astype(np.float32) @ jd_vec
    bm25 = BM25Index([tokenize(d) for d in docs])
    lex_raw = bm25.scores_for_query(jd_query_terms())
    sim, s_dense, s_lex = fuse(dense_raw, lex_raw)

    prog.progress(0.6, "Phase 3 — recruiter-brain composite")
    rfit = brain.recruiter_fit(np.array(s_struct, np.float32), sim)
    prelim = brain.finalize(rfit, np.array(gates, np.float32),
                            np.array(behav, np.float32), np.array(hkill, np.float32))
    short = np.argsort(-prelim)[:min(C.SHORTLIST_N, len(ids))]

    prog.progress(0.8, f"Phase 4 — cross-encoder rerank ({enc.mode})")
    rr = CrossReranker()
    s_cross = rr.score([docs[i] for i in short])
    fit = rfit.copy()
    fit[short] = brain.blend_cross(rfit[short], s_cross)
    final = brain.finalize(fit, np.array(gates, np.float32),
                           np.array(behav, np.float32), np.array(hkill, np.float32))

    prog.progress(1.0, "Done")

    # Replicate submission.csv exactly: round score to 4 dp, then sort by
    # (-rounded_score, candidate_id) so ties break by candidate_id ascending —
    # identical to rank.py's output ordering and formatting.
    final_r = np.round(final, 4)
    order = sorted(range(len(ids)), key=lambda i: (-float(final_r[i]), ids[i]))
    top = order[:int(topk)]

    rows = [
        {
            "candidate_id": ids[i],
            "rank": rankpos,
            "score": f"{float(final_r[i]):.4f}",
            "reasoning": build_reasoning(cands[i], diags[i], rankpos,
                                         is_honeypot=hflag[i]),
        }
        for rankpos, i in enumerate(top, start=1)
    ]
    df = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])

    st.subheader(f"Ranked candidates  ·  encoder={enc.mode}  rerank={rr.mode}")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
    )
