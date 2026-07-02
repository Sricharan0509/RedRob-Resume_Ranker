"""
hybrid.py — dual-engine fusion (Phase 2, Stage-1 filter).

Combines the dense vector-store cosine (S_dense) and BM25 lexical (S_lex) into a
single retrieval similarity per candidate. Each raw channel is normalized to
[0,1] with a robust min-max over the pool, then blended by config weights.

    Sim_hybrid = W_DENSE * S_dense_norm + W_LEX * S_lex_norm
"""

import numpy as np
from .. import config as C


def normalize01(x: np.ndarray) -> np.ndarray:
    """Robust min-max to [0,1]; clips the top/bottom 0.5% to resist outliers."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = np.percentile(x, 0.5)
    hi = np.percentile(x, 99.5)
    if hi <= lo:
        return np.zeros_like(x)
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def fuse(dense_raw: np.ndarray, lex_raw: np.ndarray):
    """Return (Sim_hybrid, S_dense_norm, S_lex_norm), all (N,) in [0,1]."""
    s_dense = normalize01(dense_raw)
    s_lex = normalize01(lex_raw)
    sim = C.W_DENSE * s_dense + C.W_LEX * s_lex
    return sim, s_dense, s_lex
