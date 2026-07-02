"""
profile_doc.py — weighted profile-document builder (Phase 1).

We up-weight *what the candidate actually did* (career-history descriptions)
over *what they claim* (headline/summary/skills), because keyword-stuffers
inflate the self-summary and skills list. The resulting document is the single
text that feeds the dense encoder, the BM25 index, and the cross-encoder — so
all three retrieval channels reason over the same career-grounded view.
"""

from .. import config as C


def build_profile_doc(candidate: dict) -> str:
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    headline = profile.get("headline", "") or ""
    summary = profile.get("summary", "") or ""
    title = profile.get("current_title", "") or ""

    head_block = f"{title}. {headline}. {summary}".strip()

    career_bits = []
    for job in career:
        t = job.get("title", "") or ""
        desc = job.get("description", "") or ""
        comp = job.get("company", "") or ""
        if t or desc:
            career_bits.append(f"{t} at {comp}: {desc}")
    career_block = " ".join(career_bits)

    skill_names = " ".join((s.get("name", "") or "") for s in skills)

    parts = []
    parts += [head_block] * C.DOC_W_HEADLINE_SUMMARY
    parts += [career_block] * C.DOC_W_CAREER_DESC
    parts += [skill_names] * C.DOC_W_SKILLS

    doc = " \n ".join(p for p in parts if p.strip())
    # keep it bounded so encoders stay fast and within max sequence length
    return doc[:C.DOC_MAX_CHARS]
