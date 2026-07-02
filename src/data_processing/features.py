"""
features.py — structured "recruiter brain" rubric (Phase 1 + Phase 3).

S_struct is the anti-stuffer backbone. It rewards title/trajectory, what the
candidate actually built (career descriptions), trust-weighted skills, an
experience-shape curve, evaluation-framework literacy, and product-company
signal — then subtracts capped penalties.

role_gate is a separate multiplicative decision based on title ∪ career
evidence, NEVER on the skills list (that is exactly how stuffers win).

trajectory_score exposes a "Score_Trajectory" as a standalone
diagnostic (attached to struct_score()'s `diag` dict, not folded into
S_struct itself) so the deck/interview can show it. reasoning.py does not
currently read it.
"""

from math import log1p, log
from .. import config as C

_PROF_W = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.15}
_LOG11 = log(11)


# ── helpers ──────────────────────────────────────────────────────────────────
def career_text(career) -> str:
    return " ".join((j.get("description", "") or "") + " " + (j.get("title", "") or "")
                    for j in career).lower()


def ml_signal_count(career) -> int:
    text = career_text(career)
    return sum(text.count(kw) for kw in C.ML_PRODUCTION_KW)


def _has_retrieval_build(career) -> bool:
    text = career_text(career)
    return sum(1 for kw in C.RETRIEVAL_BUILD_KW if kw in text) >= 2


def title_class(title: str) -> str:
    t = (title or "").lower()
    if any(kw in t for kw in C.NON_TECHNICAL_TITLES):
        return "nontech"
    if any(kw in t for kw in C.AI_TITLES):
        return "ai"
    if any(kw in t for kw in C.DATA_SCIENTIST_TITLES):
        return "ds"
    if any(kw in t for kw in C.ADJACENT_TITLES):
        return "adjacent"
    return "unknown"


# ── role_gate ────────────────────────────────────────────────────────────────
def role_gate(title: str, career) -> float:
    cls = title_class(title)
    if cls == "nontech":
        return C.ROLE_GATE_NONTECH
    if cls in ("ai", "ds"):
        return C.ROLE_GATE_CORE
    ml_hits = ml_signal_count(career)
    if cls == "adjacent":
        if _has_retrieval_build(career):
            return C.ROLE_GATE_ADJ_EVID       # the no-buzzword Tier-5 lift
        if ml_hits >= 2:
            return C.ROLE_GATE_ADJ_TECH
        return C.ROLE_GATE_GENERIC
    # unknown title
    if _has_retrieval_build(career):
        return C.ROLE_GATE_ADJ_EVID
    if ml_hits >= 4:
        return C.ROLE_GATE_ADJ_TECH
    return C.ROLE_GATE_GENERIC


# ── S_struct components ──────────────────────────────────────────────────────
def _role_fit(title, career) -> float:
    cls = title_class(title)
    if cls == "nontech":
        return 0.05
    if cls == "ai":
        ai_roles = sum(
            1 for j in career
            if any(kw in (j.get("title", "") or "").lower() for kw in C.AI_TITLES)
        )
        return min(1.0, 0.85 + ai_roles * 0.05)
    if cls == "ds":
        ml = ml_signal_count(career)
        return 0.82 if ml >= 4 else (0.68 if ml >= 1 else 0.50)
    if cls == "adjacent":
        if _has_retrieval_build(career):
            return 0.66
        ml = ml_signal_count(career)
        return 0.55 if ml >= 5 else (0.42 if ml >= 2 else 0.28)
    ml = ml_signal_count(career)
    return 0.40 if ml >= 4 else 0.18


def _career_evidence(career) -> float:
    return min(1.0, ml_signal_count(career) / 14.0)


def _skill_trust(skills, assessments) -> tuple:
    """Return (0-1 trust score, [matched critical skill names])."""
    total = 0.0
    matched = []
    for s in skills:
        name = (s.get("name", "") or "").lower().strip()
        if not any(kw in name for kw in C.CRITICAL_SKILLS):
            continue
        matched.append(name)
        prof = _PROF_W.get(s.get("proficiency", "beginner"), 0.5)
        end = min(1.0, log1p(s.get("endorsements", 0) or 0) / _LOG11)
        dur = min((s.get("duration_months", 0) or 0) / 24.0, 1.0)
        contrib = prof * (0.5 + 0.5 * end) * (0.4 + 0.6 * dur)
        for aname, ascore in (assessments or {}).items():
            if aname.lower() == name and ascore >= 70:
                contrib *= 1.2
                break
        if (s.get("duration_months", 0) or 0) == 0 and prof >= 1.0:
            contrib *= 0.4                       # expert w/ 0 months → suspicious
        total += contrib
    # dedupe matched names, preserve order
    seen, deduped = set(), []
    for n in matched:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return min(1.0, total / 4.0), deduped


def _experience_fit(years: float) -> float:
    if years < 2:
        return 0.20
    if years < 4:
        return 0.55
    if years < 5:
        return 0.78
    if years <= 9:
        return 1.00
    if years <= 12:
        return 0.85
    return 0.65


def _eval_framework(career, skills) -> float:
    text = career_text(career) + " " + " ".join(
        (s.get("name", "") or "").lower() for s in skills)
    strong = ("ndcg", "mrr", "learning-to-rank", "offline evaluation",
              "a/b test", "a/b testing", "ab test")
    some = ("map", "evaluation", "precision", "recall", "offline", "online")
    if any(k in text for k in strong):
        return 1.0
    if any(k in text for k in some):
        return 0.5
    return 0.0


def _product_company(candidate, career) -> float:
    prof = candidate.get("profile", {}) or {}
    inds = [(prof.get("current_industry", "") or "").lower()]
    inds += [(j.get("industry", "") or "").lower() for j in career]
    tech_terms = ("technology", "software", "internet", "saas", "fintech",
                  "ai", "machine learning", "data", "e-commerce", "product")
    tech_jobs = sum(1 for i in inds if any(t in i for t in tech_terms))
    svc = sum(1 for i in inds if "consult" in i or "services" in i or "outsourc" in i)
    base = min(1.0, 0.35 + tech_jobs * 0.22)
    if svc and tech_jobs == 0:
        base *= 0.5
    return base


# ── "Score_Trajectory": career-progression coherence ─────────────────────────
def trajectory_score(candidate: dict) -> float:
    """
    Alignment of career progression for a *senior* IC role.
    Rewards: steady 6-8y arc, growing seniority, product-company tenure.
    Penalizes: title-chasing (<1.5y hops), consulting-only, thin history.

    NOT folded into S_struct -- struct_score() attaches this to `diag` for
    reporting/diagnostics only (title-chasing and thin-history are separately
    penalized via PENALTIES["title_chaser"] etc. in _penalties()). Exposed
    standalone so the deck/interview can chart it.
    """
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    years = float(profile.get("years_of_experience", 0) or 0)

    if not career:
        return 0.35

    tenures = [int(j.get("duration_months", 0) or 0) for j in career]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 0
    stability = min(1.0, avg_tenure / 24.0)             # 24mo avg → full marks

    shape = _experience_fit(years)                      # peak 6-8y

    # seniority arc: does the latest title read more senior than the first?
    sr_words = ("senior", "lead", "staff", "principal", "head", "manager")
    first_sr = any(w in (career[-1].get("title", "") or "").lower() for w in sr_words)
    last_sr = any(w in (career[0].get("title", "") or "").lower() for w in sr_words)
    arc = 1.0 if (last_sr and not first_sr) else (0.8 if last_sr else 0.6)

    return round(0.45 * stability + 0.35 * shape + 0.20 * arc, 4)


# ── penalties ────────────────────────────────────────────────────────────────
# Human-readable label per penalty key, for grounded reasoning (reasoning.py).
# Each maps 1:1 to a JD disqualifier and only surfaces when that penalty fired.
PENALTY_REASONS = {
    "cv_speech_robotics_primary":
        "their skills lean toward computer vision or speech, with little "
        "retrieval or NLP work to balance it",
    "academic_only_no_prod":
        "their background looks mostly research or academic, with little sign "
        "of shipping systems to production",
    "langchain_recent_only":
        "their LLM work is recent LangChain-style tooling, without the older "
        "retrieval fundamentals the job asks for",
    "title_chaser":
        "they have changed jobs fairly often (average stay under 18 months)",
    "consulting_only_career":
        "their whole career has been at services or consulting firms, which "
        "the job screens against",
    "architecture_tech_lead_stale":
        "their current role is architect or tech-lead, so they may be less "
        "hands-on with code lately",
}


def _penalties(candidate, career, skills) -> tuple:
    """Return (total_penalty, [fired penalty keys]). Reasons are diagnostic
    only; the score uses total_penalty exactly as before."""
    pen = 0.0
    fired = []
    text = career_text(career)
    prof = candidate.get("profile", {}) or {}
    summary = (prof.get("summary", "") or "").lower()

    skill_names = " ".join((s.get("name", "") or "").lower() for s in skills)
    wrong = sum(1 for kw in C.WRONG_DOMAIN if kw in skill_names or kw in text)
    ir_present = any(kw in (skill_names + " " + text)
                     for kw in ("nlp", "retrieval", "search", "ranking", "embedding"))
    if wrong >= 3 and not ir_present:
        pen += C.PENALTIES["cv_speech_robotics_primary"]
        fired.append("cv_speech_robotics_primary")

    academic = any(k in (summary + " " + text) for k in C.ACADEMIC_KW)
    production = any(k in text for k in C.PROD_KW)
    if academic and not production:
        pen += C.PENALTIES["academic_only_no_prod"]
        fired.append("academic_only_no_prod")

    langchain = any(k in (summary + " " + text + " " + skill_names) for k in C.LANGCHAIN_KW)
    pre_llm = any(k in (text + " " + skill_names)
                  for k in ("recommendation", "learning-to-rank", "bm25",
                            "pytorch", "scikit", "xgboost", "information retrieval"))
    if langchain and not pre_llm:
        pen += C.PENALTIES["langchain_recent_only"]
        fired.append("langchain_recent_only")

    tenures = [j.get("duration_months", 24) or 24 for j in career]
    if len(tenures) >= 3 and (sum(tenures) / len(tenures)) < 18:
        pen += C.PENALTIES["title_chaser"]
        fired.append("title_chaser")

    total = len(career)
    consulting = sum(
        1 for j in career
        if any(f in (j.get("company", "") or "").lower() for f in C.CONSULTING_FIRMS)
    )
    if total >= 2 and consulting == total:
        pen += C.PENALTIES["consulting_only_career"]
        fired.append("consulting_only_career")

    current_job = next((j for j in career if j.get("is_current")), None)
    if current_job:
        current_title = (current_job.get("title", "") or "").lower()
        current_tenure = int(current_job.get("duration_months", 0) or 0)
        if (any(kw in current_title for kw in C.ARCHITECTURE_TECH_LEAD_KW)
                and current_tenure >= 18):
            pen += C.PENALTIES["architecture_tech_lead_stale"]
            fired.append("architecture_tech_lead_stale")

    return pen, fired


# ── public API ───────────────────────────────────────────────────────────────
def struct_score(candidate: dict) -> tuple:
    """Return (S_struct in [0,1], {feature diagnostics for reasoning})."""
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    assessments = signals.get("skill_assessment_scores", {}) or {}
    years = float(profile.get("years_of_experience", 0) or 0)
    title = profile.get("current_title", "") or ""

    rf = _role_fit(title, career)
    ce = _career_evidence(career)
    st, matched_skills = _skill_trust(skills, assessments)
    ef = _experience_fit(years)
    evf = _eval_framework(career, skills)
    pc = _product_company(candidate, career)

    w = C.STRUCT_WEIGHTS
    pos = (rf * w["role_fit"] + ce * w["career_evidence"] + st * w["skill_trust"]
           + ef * w["experience_fit"] + evf * w["eval_framework"]
           + pc * w["product_company"])
    pen, pen_reasons = _penalties(candidate, career, skills)
    score = max(0.0, min(1.0, pos - pen))

    diag = dict(
        title_class=title_class(title),
        role_fit=rf,
        career_evidence=ce,
        ml_signals=ml_signal_count(career),
        skill_trust=st,
        matched_skills=matched_skills,
        eval_framework=evf,
        product_company=pc,
        penalties=pen,
        penalty_reasons=pen_reasons,
        trajectory=trajectory_score(candidate),
    )
    return score, diag
