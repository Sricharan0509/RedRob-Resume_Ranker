"""
honeypot.py — impossibility detectors (Phase 1 hard-kill gate).

Any rule firing → the candidate is later multiplied by HONEYPOT_KILL and drops
out of the top 100. General logic only; no individual IDs are special-cased.
Each rule tests for a logical contradiction a real profile cannot contain.
"""

import re
from .. import config as C

_YOE_IN_SUMMARY_RE = re.compile(r"with\s+([0-9]+(?:\.[0-9]+)?)\s+years", re.I)


def is_honeypot(candidate: dict) -> bool:
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []

    years_exp = float(profile.get("years_of_experience", 0) or 0)
    career_months = sum(int(j.get("duration_months", 0) or 0) for j in career)
    career_years = career_months / 12 if career_months else 0.0

    # 1. career start predates plausible working age vs. claimed experience
    for job in career:
        start = str(job.get("start_date", "") or "")
        if len(start) >= 4:
            try:
                start_year = int(start[:4])
                if (C.TODAY.year - start_year) > years_exp + 6:
                    return True
            except ValueError:
                pass

    # 2. any "expert" skill with < 2 months of usage
    for s in skills:
        if (s.get("proficiency") == "expert"
                and int(s.get("duration_months", 99) or 0) < 2):
            return True

    # 3. structured YOE diverges sharply from actual career timeline
    if career_months and abs(career_years - years_exp) > 4:
        return True

    # 4. internal date contradictions (end < start, future start)
    for job in career:
        sd = str(job.get("start_date", "") or "")
        ed = str(job.get("end_date", "") or "")
        try:
            if len(sd) >= 4:
                sy = int(sd[:4])
                if sy > C.TODAY.year:
                    return True
                if len(ed) >= 4 and ed[:4].isdigit():
                    ey = int(ed[:4])
                    if ey < sy:
                        return True
        except ValueError:
            pass

    # 5. summary states a YOE that contradicts the structured field, and the
    #    career history backs the summary → structured field was tampered
    m = _YOE_IN_SUMMARY_RE.search(profile.get("summary", "") or "")
    if m:
        stated = float(m.group(1))
        if abs(stated - years_exp) > 2.5 and abs(career_years - years_exp) > 2.5:
            return True

    # 6. multiple advanced/expert skills with zero recorded usage
    impossible = sum(
        1 for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and int(s.get("duration_months", 1) or 0) == 0
    )
    if impossible >= 3:
        return True

    return False
