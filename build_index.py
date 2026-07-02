#!/usr/bin/env python3
"""
build_index.py — OFFLINE precompute for the RAG ranker (UNBUDGETED).

Builds and ships the retrieval artifacts so that rank-time (rank.py) stays under
the 5-minute / no-network / CPU budget:

    artifacts/dense.f16.npy    (N, dim) float16 profile embeddings
    artifacts/ids.npy          (N,)     candidate_id order aligned to dense.npy
    artifacts/bm25_scores.npy  (N,)     BM25 score vs the fixed JD query, same order
    artifacts/jd_dense.npy     (dim,)   mean-pooled JD query vector
    artifacts/build_meta.json  which encoder / vector-store backend was used

If chromadb is installed, the embeddings are ALSO loaded into a local Chroma
collection to demonstrate the vector-database workflow of the RAG pipeline
(the numpy matrix remains the source of truth for the millisecond rank-time
query — see retrieval/embeddings.load_vector_store).

Usage:
    python build_index.py --candidates ./data/candidates.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src import config as C
from src.data_processing.io_utils import stream_candidates
from src.data_processing.profile_doc import build_profile_doc
from src.retrieval.embeddings import Encoder, build_jd_vector
from src.retrieval.lexical import BM25Index, tokenize, jd_query_terms

HERE = Path(__file__).resolve().parent
ART = HERE / C.ARTIFACT_DIR


def main():
    ap = argparse.ArgumentParser(description="Build RAG retrieval artifacts.")
    ap.add_argument("--candidates", default="./data/candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="cap docs (debug)")
    args = ap.parse_args()

    ART.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[1/5] Parsing + building profile_docs from {args.candidates} ...")
    ids, docs = [], []
    for i, cand in enumerate(stream_candidates(args.candidates)):
        ids.append(cand["candidate_id"])
        docs.append(build_profile_doc(cand))
        if args.limit and len(ids) >= args.limit:
            break
        if (i + 1) % 20000 == 0:
            print(f"      {i + 1:,} parsed ...")
    print(f"      {len(ids):,} profile_docs built ({time.time() - t0:.0f}s)")

    print("[2/5] Loading encoder ...")
    enc = Encoder(prefer_st=True)
    print(f"      encoder mode = {enc.mode} (dim={enc.dim})")

    print("[3/5] Encoding profiles (this may exceed 5 min — allowed offline) ...")
    matrix = enc.encode(docs, is_query=False)
    np.save(ART / C.DENSE_NPY, matrix.astype(np.float16))
    np.save(ART / C.IDS_NPY, np.array(ids))
    jd_vec = build_jd_vector(enc)
    np.save(ART / C.JD_DENSE_NPY, jd_vec.astype(np.float32))
    print(f"      dense matrix {matrix.shape} saved ({time.time() - t0:.0f}s)")

    print("[4/5] Building BM25 index + scoring fixed JD query ...")
    bm25 = BM25Index([tokenize(d) for d in docs])
    # JD_BM25_TERMS (config.py) is fixed at build time -- persist the resulting
    # score vector instead of the full tokenized index (see BM25_SCORES_NPY).
    lex_scores = bm25.scores_for_query(jd_query_terms())
    np.save(ART / C.BM25_SCORES_NPY, lex_scores.astype(np.float32))
    print(f"      BM25 over {bm25.N:,} docs -> scores saved ({time.time() - t0:.0f}s)")

    backend = _maybe_load_chroma(ids, matrix)

    meta = dict(
        encoder_mode=enc.mode, dim=int(enc.dim), n=len(ids),
        vector_store=backend, seed=C.SEED,
        dense_model=C.DENSE_MODEL if enc.mode == "st" else "hashing-fallback",
    )
    (ART / C.META_JSON).write_text(json.dumps(meta, indent=2))
    print(f"[5/5] Wrote {C.META_JSON}: {meta}")
    print(f"Done in {time.time() - t0:.0f}s. Artifacts in {ART}")


def _maybe_load_chroma(ids, matrix):
    """Load embeddings into a local Chroma collection if chromadb is present."""
    if C.VECTOR_STORE_BACKEND == "numpy":
        return "numpy"
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(ART / "chroma"))
        try:
            client.delete_collection(C.CHROMA_COLLECTION)
        except Exception:
            pass
        col = client.create_collection(
            C.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        B = 5000
        for s in range(0, len(ids), B):
            e = min(s + B, len(ids))
            col.add(ids=list(ids[s:e]),
                    embeddings=matrix[s:e].astype(float).tolist())
        print(f"      Chroma collection '{C.CHROMA_COLLECTION}' loaded "
              f"({len(ids):,} vectors)")
        return "chroma"
    except Exception as ex:
        print(f"      chromadb not used ({type(ex).__name__}); numpy store only")
        return "numpy"


if __name__ == "__main__":
    main()
