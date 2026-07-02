"""
lexical.py — BM25 lexical channel (Phase 2, dual-engine retrieval).

Dense similarity can blur hard, non-negotiable constraints ("FAISS", "NDCG").
BM25 over the same profile_docs, queried with curated JD hard-constraint terms,
recovers them. Implemented as a compact pure-python BM25 (no external deps)
so the channel always works offline.

At build time (build_index.py) the fixed JD query (JD_BM25_TERMS in config.py)
is scored once and the resulting (N,) vector is persisted to
artifacts/bm25_scores.npy -- not the full tokenized index, which pickles to
~130MB for 100K docs and exceeds GitHub's 100MB per-file push limit. The full
BM25Index class here remains the source of truth: used to build that vector,
and used directly (in-memory, no artifact) by app.py's sandbox and by
rank.py's in-budget fallback path when artifacts are unavailable.
"""

import math
import pickle
from collections import Counter

import numpy as np

from .. import config as C


def tokenize(text: str):
    out, cur = [], []
    for ch in (text or "").lower():
        if ch.isalnum() or ch in "+#.":
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


class BM25Index:
    """Minimal BM25 (Okapi) over a fixed corpus. Serializable via pickle."""

    def __init__(self, tokenized_docs, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = np.array([len(d) for d in tokenized_docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.N = len(tokenized_docs)
        # postings: term -> list of (doc_index, term_freq)
        self.postings = {}
        df = Counter()
        for i, doc in enumerate(tokenized_docs):
            tf = Counter(doc)
            for term, f in tf.items():
                self.postings.setdefault(term, []).append((i, f))
            df.update(tf.keys())
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def scores_for_query(self, query_terms) -> np.ndarray:
        """BM25 score of every doc for the OR of query_terms → (N,)."""
        scores = np.zeros(self.N, dtype=np.float32)
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_i, f in postings:
                dl = self.doc_len[doc_i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[doc_i] += idf * (f * (self.k1 + 1)) / (denom or 1)
        return scores

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def jd_query_terms():
    """Expand the configured JD hard-constraint terms through the tokenizer."""
    terms = []
    for t in C.JD_BM25_TERMS:
        terms.extend(tokenize(t))
    return terms
