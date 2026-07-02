"""
cross_encoder.py — high-fidelity reranking (Phase 4).

An LLM cross-encoder ("Act as an elite tech recruiter…") was the original
proposal. The hosted-LLM version is banned at rank time (network/API), so we use its compliant
sibling: a *local* cross-encoder (ms-marco-MiniLM-L-6-v2) that performs true
cross-attention between the JD and each profile — the one head-sharpening idea
from the LLM-rerank school, run on CPU over only the ~300-candidate shortlist.

If sentence-transformers is unavailable, we fall back to a deterministic lexical
cross-overlap score so the stage still contributes signal offline.
"""

import os

# Must be set before sentence_transformers (-> transformers) is imported;
# see retrieval/embeddings.py for why. Harmless if already set.
os.environ.setdefault("USE_TF", "0")

import numpy as np
from .. import config as C
from ..retrieval.embeddings import _apply_cpu_thread_budget
from ..retrieval.lexical import tokenize, jd_query_terms


class CrossReranker:
    def __init__(self):
        self.model = None
        self.mode = "lexical"
        try:
            from sentence_transformers import CrossEncoder
            _apply_cpu_thread_budget()
            self.model = CrossEncoder(
                C.CROSS_MODEL, device="cpu", max_length=C.MAX_SEQ_LENGTH)
            self.mode = "cross-encoder"
        except Exception:
            self.model = None
            self.mode = "lexical"

    def score(self, profile_docs) -> np.ndarray:
        """Return S_cross in [0,1] for each shortlist profile_doc vs the JD."""
        if self.mode == "cross-encoder":
            pairs = [(C.JD_CROSS_TEXT, doc) for doc in profile_docs]
            logits = self.model.predict(
                pairs, batch_size=C.CROSS_BATCH_SIZE, show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32)
            return _sigmoid(logits)
        return self._lexical_overlap(profile_docs)

    def _lexical_overlap(self, profile_docs) -> np.ndarray:
        jd_terms = set(jd_query_terms())
        out = np.zeros(len(profile_docs), dtype=np.float32)
        for i, doc in enumerate(profile_docs):
            toks = set(tokenize(doc))
            if not toks:
                continue
            inter = len(jd_terms & toks)
            out[i] = inter / (len(jd_terms) or 1)
        # spread to [0,1] via simple scaling on observed range
        mx = out.max() if out.size else 0.0
        return out / mx if mx > 0 else out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
