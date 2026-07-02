"""
embeddings.py — dense bi-encoder + local vector store (Phase 2).

The "RAG" retrieval engine. Two layers:

  1. Encoder      — sentence-transformers BGE-small if available; otherwise a
                    deterministic pure-numpy hashing encoder (no network, no
                    heavy deps) so the pipeline ALWAYS reproduces offline.
  2. Vector store — ChromaDB collection if installed; otherwise a numpy flat
                    (brute-force cosine) index. Both expose the same .query().

At rank time we do not re-encode 100K profiles: build_index.py precomputes the
matrix and we load it. The vector store is queried with the mean-pooled JD
vector to produce S_dense for every candidate.

Everything here is CPU-only and offline.
"""

import hashlib
import os

# Must be set before `sentence_transformers` (-> transformers) is imported
# anywhere in the process: on machines with both torch and tensorflow
# installed, transformers' TF-integration path breaks under Keras 3. This
# affects nothing when TF isn't installed (the common case per requirements.txt).
os.environ.setdefault("USE_TF", "0")

import numpy as np

from .. import config as C

# ── seeds for determinism ────────────────────────────────────────────────────
np.random.seed(C.SEED)

_threads_set = False


def _apply_cpu_thread_budget():
    """torch defaults to 4 threads regardless of core count; opt into all of
    them once per process. No-op if torch isn't installed."""
    global _threads_set
    if _threads_set or not C.USE_ALL_CPU_THREADS:
        return
    try:
        import torch
        torch.set_num_threads(os.cpu_count() or 1)
    except Exception:
        pass
    _threads_set = True


# ─────────────────────────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────────────────────────
class Encoder:
    """
    Wraps a sentence-transformers model when present, else a hashing encoder.
    `mode` in {"st", "hash"} is recorded in build_meta.json so rank.py loads a
    matching query encoder (the two must agree or cosine is meaningless).
    """

    def __init__(self, prefer_st: bool = True):
        self.model = None
        self.mode = "hash"
        self.dim = C.FALLBACK_DIM
        # Config can force the hash encoder regardless of caller preference, so
        # the dense build never falls into the ~2.75h real-BGE path by accident.
        if C.DENSE_ENCODER_MODE == "hash":
            prefer_st = False
        if prefer_st:
            try:
                from sentence_transformers import SentenceTransformer
                _apply_cpu_thread_budget()
                self.model = SentenceTransformer(C.DENSE_MODEL, device="cpu")
                self.model.max_seq_length = C.MAX_SEQ_LENGTH  # CPU OOM guard
                self.mode = "st"
                self.dim = C.DENSE_DIM
            except Exception:
                self.model = None
                self.mode = "hash"
                self.dim = C.FALLBACK_DIM

    # -- public --------------------------------------------------------------
    def encode(self, texts, is_query: bool = False, batch_size: int = None):
        if batch_size is None:
            batch_size = C.ENCODE_BATCH_SIZE
        if self.mode == "st":
            if is_query:
                texts = [C.BGE_QUERY_PREFIX + t for t in texts]
            vecs = self.model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            return vecs.astype(np.float32)
        return self._hash_encode(texts)

    # -- deterministic hashing fallback --------------------------------------
    def _hash_encode(self, texts):
        """
        Bag-of-hashed-tokens → L2-normalized vector. Deterministic, offline,
        dependency-free. Not as strong as BGE but keeps the RAG pipeline whole
        and reproducible in any sandbox; the structured + BM25 + cross channels
        carry ranking quality when this fallback is active.
        """
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokenize(t):
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 32) & 1 else -1.0
                out[i, idx] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def _tokenize(text: str):
    tok = []
    cur = []
    for ch in (text or "").lower():
        if ch.isalnum() or ch in "+#.":
            cur.append(ch)
        else:
            if cur:
                tok.append("".join(cur))
                cur = []
    if cur:
        tok.append("".join(cur))
    return tok


def build_jd_vector(encoder: Encoder) -> np.ndarray:
    """Mean-pool the positive JD passages into one query vector (L2-normed)."""
    vecs = encoder.encode(C.JD_DENSE_PASSAGES, is_query=True)
    q = vecs.mean(axis=0)
    n = np.linalg.norm(q)
    return (q / n if n > 0 else q).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Vector store
# ─────────────────────────────────────────────────────────────────────────────
class NumpyVectorStore:
    """Flat brute-force cosine index over an in-memory float32 matrix."""

    def __init__(self, matrix: np.ndarray, ids: np.ndarray):
        self.matrix = matrix.astype(np.float32)   # (N, dim), rows L2-normed
        self.ids = ids

    def query_all(self, qvec: np.ndarray) -> np.ndarray:
        """Cosine of every row against qvec → (N,) in [-1, 1]."""
        q = qvec.astype(np.float32)
        n = np.linalg.norm(q)
        if n > 0:
            q = q / n
        return self.matrix @ q


def load_vector_store(matrix: np.ndarray, ids: np.ndarray):
    """
    Return an object with .query_all(). We keep the numpy store at *rank* time
    even if Chroma built the artifacts: a single JD-vs-all dot product over
    100K×384 f16 is milliseconds and needs no DB process. Chroma is used at
    *build* time (see build_index.py) to demonstrate the vector-DB workflow.
    """
    return NumpyVectorStore(matrix, ids)
