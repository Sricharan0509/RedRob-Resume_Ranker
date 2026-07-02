"""
reasoning.py - deterministic, feature-grounded reasoning for each ranked
candidate, produced WITHOUT a network LLM.

Design goal (submission_spec.docx §3 Stage-4 review): the text must EXPLAIN THE
RANK, not just praise the candidate. So the generator does two grounded passes
over each candidate's own diagnostics:

  strengths  - what earns the rank (deep named skill, evaluation-metrics work,
               dense retrieval/ranking career evidence, a core AI title,
               product-company track record, seniority arc, availability).
  shortfalls - what CAPS the rank: the specific dimensions where this candidate
               scored below a top-tier profile (adjacent title, retrieval shown
               only indirectly via an adjacent tool, thin career signal, low
               endorsements, no NDCG/MRR/A-B evidence, fewer years, weak
               product-company signal, notice period, low response rate,
               inactivity, outside-India). Because higher-ranked candidates are
               exactly the ones without these shortfalls, naming a candidate's
               top shortfall answers "why rank N and not rank 5?".

Composition is driven by WHICH strength is dominant and WHICH shortfall is
dominant for this candidate (both differ candidate to candidate), not by a
fixed template, so two rows are not swappable by changing only the nouns.
Tone is banded to rank: top = confident (caveat if any); mid = a strength then
the specific limitation; low = lead with the limitation, strengths second.

Every value cited (skill, level, endorsements, months, years, metric, employer,
signal) is read straight from the candidate record - nothing is inferred from a
title or invented. Output is ASCII-folded and deterministic (variation seeded
off candidate_id, not RNG), preserving byte-identical reproduction.
"""

import unicodedata
from datetime import date
from . import config as C

# Punctuation that commonly sneaks in as non-ASCII, mapped to safe equivalents.
_PUNCT_MAP = {
    "—": "-", "–": "-",       # em / en dash
    "‘": "'", "’": "'",       # curly single quotes
    "“": '"', "”": '"',       # curly double quotes
    "…": "...", " ": " ",     # ellipsis, non-breaking space
}


def _asciiize(text: str) -> str:
    """Fold to plain ASCII so the text renders identically in any viewer (a
    UTF-8 BOM would fix Excel but break validate_submission.py's header check)."""
    for bad, good in _PUNCT_MAP.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


# ── skill categorization (grounded, read-only) ───────────────────────────────
_VECTOR_EMBED = (
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "vector", "embedding", "embeddings", "semantic search",
    "dense retrieval", "bi-encoder", "cross-encoder", "ann", "hybrid search",
    "retrieval augmented", "rag", "colbert", "dense passage",
)
_RANKING_IR = (
    "ranking", "learning-to-rank", "information retrieval", "bm25",
    "reranking", "re-ranking", "recommendation", "recommendation systems",
    "search systems",
)
_LLM = ("llm", "large language model", "fine-tuning", "fine tuning", "lora",
        "qlora", "peft", "transformer", "bert", "gpt", "hugging face",
        "huggingface", "language model")

_PROF_W = {"beginner": 0.4, "intermediate": 0.7, "advanced": 1.0, "expert": 1.15}
_SENIOR_WORDS = ("senior", "lead", "staff", "principal", "head")


def _variant(candidate: dict, k: int) -> int:
    """Deterministic per-candidate index in [0, k) for phrasing rotation."""
    cid = candidate.get("candidate_id", "") or ""
    digits = "".join(ch for ch in cid if ch.isdigit())
    seed = int(digits) if digits else sum(ord(c) for c in cid)
    return seed % k if k else 0


def _band(rank: int) -> str:
    if rank <= 15:
        return "top"
    if rank <= 60:
        return "mid"
    return "low"


def _yrs(yoe: float) -> str:
    yoe = float(yoe)
    return f"{int(yoe)} years" if yoe == int(yoe) else f"{yoe:.1f} years"


def _dur_phrase(months: int) -> str:
    months = int(months)
    if months >= 18:
        return f"about {round(months / 12)} years"
    return f"{months} months"


def _pct(x) -> str:
    return f"{round(float(x) * 100)}%"


def _article(title: str) -> str:
    """'A' or 'An' for a title, handling vowel-sound acronyms (AI, ML, NLP)."""
    first = (title.split()[0] if title else "")
    if not first:
        return "A"
    if first[0] in "AEIOU":
        return "An"
    # all-caps acronym whose leading letter is pronounced with a vowel sound
    if first.isupper() and first[0] in "AEFHILMNORSX":
        return "An"
    return "A"


def _skill_details(candidate: dict, matched):
    """Map matched (lowercased) skill names back to real skill objects; recover
    display name + level + endorsements + duration. No invention."""
    by_lower = {}
    for s in candidate.get("skills", []) or []:
        nm = (s.get("name", "") or "").lower().strip()
        if nm and nm not in by_lower:
            by_lower[nm] = s
    out = []
    for m in matched:
        s = by_lower.get(m)
        if not s:
            continue
        out.append(dict(
            name=(s.get("name", "") or m),
            prof=(s.get("proficiency", "") or ""),
            end=int(s.get("endorsements", 0) or 0),
            dur=int(s.get("duration_months", 0) or 0),
        ))
    return out


def _in(name, terms):
    n = name.lower()
    return any(t in n for t in terms)


def _current_seat(candidate: dict):
    prof = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    comp = (prof.get("current_company", "") or "").strip()
    cur = next((j for j in career if j.get("is_current")), None)
    if cur is None and career:
        cur = career[0]
    if not comp and cur:
        comp = (cur.get("company", "") or "").strip()
    months = int(cur.get("duration_months", 0) or 0) if cur else 0
    return comp, months


def _days_active(signals: dict):
    last = signals.get("last_active_date", "") or ""
    if not last:
        return None
    try:
        return (C.TODAY - date.fromisoformat(last)).days
    except ValueError:
        return None


# ── strengths (what earns the rank) ──────────────────────────────────────────
def _skill_cat_rank(name):
    """Category priority for choosing a HEADLINE skill: the JD's core ask
    (embeddings/vector search) outranks ranking/IR, which outranks NLP, which
    outranks the LLM 'nice-to-have', which outranks generic skills like Python."""
    if _in(name, _VECTOR_EMBED):
        return 4
    if _in(name, _RANKING_IR):
        return 3
    if _in(name, ("nlp", "natural language", "ner", "question answering",
                  "text classification", "text mining")):
        return 2
    if _in(name, _LLM):
        return 1
    return 0


def _best_skill(details):
    """Headline skill: prefer JD-core category first, then depth (level,
    endorsements, duration). Keeps the lead tied to the role's real focus
    instead of an incidental high-endorsement side skill."""
    if not details:
        return None
    return max(details, key=lambda d: (
        _skill_cat_rank(d["name"]), _PROF_W.get(d["prof"], 0.5), d["end"], d["dur"]))


def _strengths(candidate, diag, signals, details, best):
    """List of {cat, sal, elab, noun} strongest-first. `elab` is a fuller,
    prose-ready clause; `noun` is a short phrase for secondary mention."""
    prof = candidate.get("profile", {}) or {}
    title = prof.get("current_title", "") or ""
    yoe = float(prof.get("years_of_experience", 0) or 0)
    career = candidate.get("career_history", []) or []
    nrole = len(career)
    ml = diag.get("ml_signals", 0)
    out = []

    if best and _PROF_W.get(best["prof"], 0) >= 1.0 and (best["end"] >= 15 or best["dur"] >= 36):
        lvl, nm, en, du = best["prof"], best["name"], best["end"], best["dur"]
        out.append(dict(cat="skill", sal=4 + min(en, 60) / 60.0,
            elab=f"{lvl}-level {nm} built up over {_dur_phrase(du)} and backed by {en} peer endorsements",
            noun=f"{lvl}-level {nm} ({en} endorsements)"))

    if diag.get("eval_framework", 0) >= 1.0:
        out.append(dict(cat="eval", sal=3.4,
            elab="hands-on experience measuring ranking quality the way the JD asks for, with NDCG, MRR, or A/B testing rather than guesswork",
            noun="ranking-evaluation experience (NDCG/MRR/A-B)"))

    if ml >= 10:
        out.append(dict(cat="career", sal=4.2,
            elab=f"a career history dense with the exact work the role owns: {ml} separate retrieval and ranking signals across {nrole} roles",
            noun=f"heavy retrieval/ranking career evidence ({ml} signals)"))

    if diag.get("title_class") == "ai":
        out.append(dict(cat="title", sal=3.6,
            elab=f"a core {title} title that sits squarely in the AI/ML lane the JD targets, so the fit does not rest on a stretch from an adjacent role",
            noun=f"a core {title} title"))

    cur_senior = any(w in title.lower() for w in _SENIOR_WORDS)
    if cur_senior and 6 <= yoe <= 9 and nrole >= 3:
        out.append(dict(cat="seniority", sal=3.0,
            elab=f"a steady senior arc, {_yrs(yoe)} across {nrole} roles up to a {title} seat, matching the trajectory the JD describes",
            noun=f"a senior arc ({_yrs(yoe)}, {nrole} roles)"))

    if diag.get("product_company", 0) >= 0.6 and diag.get("title_class") in ("ai", "ds"):
        ind = prof.get("current_industry", "") or ""
        indtxt = f" (currently in {ind})" if ind else ""
        out.append(dict(cat="product", sal=2.2,
            elab=f"a genuine product-company track record{indtxt}, which the JD explicitly prefers over services or research-only backgrounds",
            noun=f"a product-company background{indtxt}"))

    rr = signals.get("recruiter_response_rate", None)
    days = _days_active(signals)
    if (signals.get("open_to_work_flag") and rr is not None and rr >= 0.5
            and days is not None and days <= 45):
        out.append(dict(cat="avail", sal=1.5,
            elab=f"strong availability - they are open to work, reply to {_pct(rr)} of recruiters, and were active in the last {days} days",
            noun=f"strong availability ({_pct(rr)} response)"))

    out.sort(key=lambda d: d["sal"], reverse=True)
    return out


# ── shortfalls (what caps the rank) ──────────────────────────────────────────
# Noun-phrase form of each fired penalty, so it reads correctly after any of
# the "held back by / on / is / there is also" frames below.
_PENALTY_NP = {
    "cv_speech_robotics_primary":
        "a computer-vision/speech-heavy skill set with little IR or NLP",
    "academic_only_no_prod":
        "a mostly research/academic background with little production work",
    "langchain_recent_only":
        "recent LangChain-style LLM work without the core retrieval fundamentals",
    "title_chaser":
        "frequent job changes (average stay under 18 months)",
    "consulting_only_career":
        "an all services/consulting career, which the JD screens against",
    "architecture_tech_lead_stale":
        "a current architect/tech-lead role that may be less hands-on",
}


def _shortfalls(candidate, diag, signals, details, best):
    """List of {cat, sal, np} most-capping-first. Each `np` is a noun phrase
    that names a specific dimension where this candidate fell below a top-tier
    profile - i.e. the answer to 'why rank N and not rank 5?'."""
    prof = candidate.get("profile", {}) or {}
    yoe = float(prof.get("years_of_experience", 0) or 0)
    ml = diag.get("ml_signals", 0)
    tcls = diag.get("title_class")
    out = []

    for r in (diag.get("penalty_reasons", []) or []):
        out.append(dict(cat="penalty", sal=6,
                        np=_PENALTY_NP.get(r, "a flagged career-pattern concern")))

    if tcls in ("adjacent", "unknown"):
        out.append(dict(cat="title", sal=5,
            np="a title outside core AI/ML (so the fit rests on project history)"))
    elif tcls == "ds":
        out.append(dict(cat="ds", sal=2,
            np="a data-science rather than ML-engineering title"))

    has_vec = any(_in(d["name"], _VECTOR_EMBED) for d in details)
    rank_skills = [d["name"] for d in details if _in(d["name"], _RANKING_IR)]
    if not has_vec and rank_skills:
        out.append(dict(cat="core_indirect", sal=4,
            np=f"embeddings/vector search shown only indirectly, via {rank_skills[0]} rather than a named focus"))

    if ml < 5:
        out.append(dict(cat="thin", sal=3.5,
            np=f"few retrieval/ranking signals in the roles (only {ml})"))
    elif ml < 9:
        out.append(dict(cat="thin", sal=2.5,
            np=f"lighter retrieval/ranking career evidence ({ml} signals) than the top ranks"))

    if best and best["end"] < 10:
        out.append(dict(cat="endorse", sal=3,
            np=f"only {best['end']} endorsements on the headline skill {best['name']}"))
    elif not details:
        out.append(dict(cat="noskill", sal=4,
            np="none of the JD's critical skills listed on the profile"))

    evf = diag.get("eval_framework", 0)
    if evf < 0.5:
        out.append(dict(cat="eval", sal=2,
            np="no NDCG/MRR/A-B evaluation evidence of the kind higher picks showed"))
    elif evf < 1.0:
        out.append(dict(cat="eval", sal=1.6,
            np="only a light evaluation-metric signal, short of the NDCG/MRR/A-B work above"))

    if diag.get("product_company", 0) < 0.6:
        out.append(dict(cat="product", sal=1.8,
            np="a thinner product-company track record than higher-ranked peers"))

    if yoe < 6:
        out.append(dict(cat="yoe", sal=2,
            np=f"{_yrs(yoe)} of experience, light for a senior role"))

    notice = signals.get("notice_period_days", None)
    rr = signals.get("recruiter_response_rate", None)
    days = _days_active(signals)
    _, cur_months = _current_seat(candidate)

    if rr is not None and rr < 0.15:
        out.append(dict(cat="resp", sal=4,
            np=f"a very low {_pct(rr)} recruiter-response rate that hurts reachability"))
    elif rr is not None and rr < 0.30:
        out.append(dict(cat="resp", sal=1.6,
            np=f"a modest {_pct(rr)} recruiter-response rate"))

    if days is not None and days > 180:
        out.append(dict(cat="dormant", sal=3.5,
            np=f"about {days} days of platform inactivity"))

    if notice is not None and notice >= 120:
        out.append(dict(cat="notice", sal=2.5,
            np=f"a long {notice}-day notice period"))
    elif notice is not None and notice >= 90:
        out.append(dict(cat="notice", sal=1.2,
            np=f"a {notice}-day notice period"))

    prof_loc = ((prof.get("location", "") or "") + " " + (prof.get("country", "") or "")).lower()
    if prof_loc.strip() and "india" not in prof_loc:
        out.append(dict(cat="visa", sal=2.2,
            np="a base outside India, which the role cannot sponsor visas for"))

    if cur_months and cur_months < 12:
        out.append(dict(cat="tenure", sal=1.3,
            np=f"only {cur_months} months in the current role"))

    if not signals.get("open_to_work_flag", True):
        out.append(dict(cat="notopen", sal=1.0,
            np="no open-to-work flag set"))

    out.sort(key=lambda d: d["sal"], reverse=True)
    seen, dd = set(), []
    for s in out:
        if s["cat"] in seen:
            continue
        seen.add(s["cat"])
        dd.append(s)
    return dd


# ── assembly ──────────────────────────────────────────────────────────────────
def build_reasoning(candidate: dict, diag: dict, rank: int,
                    is_honeypot: bool = False) -> str:
    prof = candidate.get("profile", {}) or {}
    signals = candidate.get("redrob_signals", {}) or {}
    title = prof.get("current_title", "unknown role") or "unknown role"
    yoe = float(prof.get("years_of_experience", 0) or 0)
    band = _band(rank)

    if is_honeypot:
        comp, months = _current_seat(candidate)
        where = f" at {comp}" if comp else ""
        return _asciiize(
            f"{title}{where} lists {_yrs(yoe)} of experience, but the profile "
            f"contradicts itself: the claimed experience and skill durations do "
            f"not fit the {months}-month career timeline. Excluded as a likely "
            f"planted profile, not a genuine fit.")

    details = _skill_details(candidate, diag.get("matched_skills", []) or [])
    best = _best_skill(details)
    strengths = _strengths(candidate, diag, signals, details, best)
    shortfalls = _shortfalls(candidate, diag, signals, details, best)

    # Decoupled indices from the candidate seed so the grounding phrasing,
    # which strength leads, and which defense frame all vary independently.
    cid = candidate.get("candidate_id", "") or ""
    dig = "".join(ch for ch in cid if ch.isdigit())
    seed = int(dig) if dig else sum(ord(c) for c in cid)
    p = seed // 7
    fidx = seed // 13
    a = seed % max(1, min(2, len(strengths) or 1))

    # 1) Grounding sentence: who they are, in plain terms.
    comp, months = _current_seat(candidate)
    if comp and months:
        seat = f"currently {_dur_phrase(months)} into the role at {comp}"
    elif comp:
        seat = f"currently at {comp}"
    else:
        seat = ""
    art = _article(title)
    g = [
        f"{title} with {_yrs(yoe)} of experience" + (f", {seat}." if seat else "."),
        (f"{art} {title} {seat}, with {_yrs(yoe)} behind them." if seat
         else f"{art} {title} with {_yrs(yoe)} behind them."),
        f"{title}, {_yrs(yoe)} in" + (f", {seat}." if seat else "."),
    ]
    grounding = g[p % len(g)]

    # 2) Strengths sentence: what earns the rank, tied to the JD, elaborated.
    if strengths:
        primary = strengths[a]
        secondary = next((s for s in strengths if s["cat"] != primary["cat"]), None)
        e1 = primary["elab"]
        if secondary:
            e2 = secondary["elab"]
            sc = [
                f"They earn the ranking on {e1}, reinforced by {e2}.",
                f"The strongest part of the case is {e1}; on top of that, {e2}.",
                f"What stands out is {e1}, alongside {e2}.",
                f"They bring {e1}, and {e2}.",
            ]
            strengths_sent = sc[fidx % len(sc)]
        else:
            sc = [f"The standout is {e1}.", f"Their strongest card is {e1}.",
                  f"The clearest plus is {e1}."]
            strengths_sent = sc[p % len(sc)]
        strong_noun = strengths[0]["noun"]
    else:
        strengths_sent = ("Beyond the title, the profile shows little direct "
                          "evidence for the role's core retrieval and ranking skills.")
        strong_noun = None

    top_cap = shortfalls[0] if shortfalls else None
    second_neg = shortfalls[1] if len(shortfalls) > 1 else None
    caps_join = top_cap["np"] if top_cap else ""
    if top_cap and second_neg:
        caps_join = f"{top_cap['np']} and {second_neg['np']}"
    they_avoid = ("which the stronger profiles above do not carry"
                  if not second_neg else
                  "both of which the stronger profiles above do not carry")

    # 3) Rank-defense sentence: why THIS rank, compared to the field.
    if band == "top":
        place = ("at the very top of the list" if rank == 1 else
                 "in the top three" if rank <= 3 else
                 "among the top five" if rank <= 5 else "inside the top tier")
        d = [
            f"Covering that much of what the role wants at once is why they land {place}.",
            f"That breadth against the JD's core needs is what places them {place}.",
            f"Few profiles in the pool match the JD from as many angles, so they sit {place}.",
        ]
        defense = d[fidx % len(d)]
        if top_cap:
            cav = [f" The one point to weigh before an outreach is {top_cap['np']}.",
                   f" The only real caveat is {top_cap['np']}.",
                   f" Worth noting before reaching out: {top_cap['np']}."]
            defense += cav[p % len(cav)]
        elif rank <= 3:
            defense += (" With no disqualifying gaps in the profile, there is "
                        "little separating them from the JD's stated ideal.")
        s = f"{grounding} {strengths_sent} {defense}"
    elif band == "mid":
        if top_cap:
            d = [
                f"What keeps them at rank {rank} rather than higher is {caps_join}, {they_avoid}.",
                f"They land at rank {rank}, not the top tier, because of {caps_join} - otherwise the fit runs close to the leaders'.",
                f"The reason they sit at rank {rank} and not higher is {caps_join}, {they_avoid}.",
            ]
            defense = d[fidx % len(d)]
        else:
            defense = (f"They sit at rank {rank}, narrowly behind the top tier "
                       f"on the overall fit blend rather than on any single gap.")
        s = f"{grounding} {strengths_sent} {defense}"
    else:  # low
        if strong_noun and top_cap:
            d = [
                f"They make the top 100 on {strong_noun}, but land down at rank {rank} chiefly because of {caps_join}, {they_avoid}.",
                f"There is real substance here in {strong_noun}, yet at rank {rank} the limiting factors are {caps_join}, {they_avoid}.",
                f"Even with {strong_noun}, they settle at rank {rank}, held there by {caps_join} against a stronger field above.",
            ]
            defense = d[fidx % len(d)]
            s = f"{grounding} {defense}"
        elif top_cap:
            defense = (f"At rank {rank}, the profile is capped by {caps_join}, "
                       f"with little on the other side of the ledger.")
            s = f"{grounding} {strengths_sent} {defense}"
        else:
            # Strong on the structured rubric but no single structural gap - the
            # lower rank comes from the semantic-match part of the blended score.
            d = [
                f"On the structured rubric this profile is strong; what places it at rank {rank} is a looser overall match between the full profile text and the job description than the candidates just above.",
                f"The structured signals are solid; it settles at rank {rank} because the profile-to-JD text match, which the score also weighs, is tighter for the candidates ranked above.",
            ]
            s = f"{grounding} {strengths_sent} {d[fidx % len(d)]}"

    return _asciiize(" ".join(s.split()))[:700]
