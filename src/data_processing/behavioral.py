"""
behavioral.py — availability multiplier ("Behavioral & Activity Signals").

behavioral_mult = clip(BEHAV_BASE + Σ adjustments, BEHAV_MIN, BEHAV_MAX).
Multiplicative so availability *modifies*, never *dominates*, skill fit: a
perfect-on-paper but dormant candidate (6mo inactive, 5% response) lands ≈0.60 —
down-weighted, not deleted.

Also folds in the JD's "On location, comp, and logistics" section: outside
India, the JD is explicit ("we don't sponsor work visas"), so `willing_to_relocate`
(redrob_signals_doc signal #15) only matters for candidates outside India —
within India the JD is "flexible" on location and never conditions fit on it.
"""

from datetime import date
from .. import config as C


def behavioral_mult(signals: dict, profile: dict) -> float:
    adj = 0.0

    country = (profile.get("country", "") or "").lower()
    location = (profile.get("location", "") or "").lower()
    if country or location:
        outside_india = "india" not in (country + " " + location)
        if outside_india:
            # Willingness doesn't resolve "we don't sponsor work visas" — it
            # only removes the *relocation* half of the JD's two concerns, so
            # this stays a caution, not a bonus. Not willing to relocate at
            # all compounds with the visa issue → the larger penalty.
            adj += -0.03 if signals.get("willing_to_relocate", False) else -0.08

    rr = signals.get("recruiter_response_rate", 0.40)
    if rr is None:
        rr = 0.40
    adj += (rr - 0.40) * 0.25

    last = signals.get("last_active_date", "") or ""
    if last:
        try:
            days = (C.TODAY - date.fromisoformat(last)).days
            if days < 30:
                adj += 0.08
            elif days < 90:
                adj += 0.04
            elif days > 180:
                adj -= 0.08
        except ValueError:
            pass

    adj += 0.06 if signals.get("open_to_work_flag", False) else -0.04

    notice = signals.get("notice_period_days", 90)
    if notice is None:
        notice = 90
    if notice <= 30:
        adj += 0.05
    elif notice <= 60:
        adj += 0.02
    elif notice >= 120:
        adj -= 0.05

    icr = signals.get("interview_completion_rate", 0.50)
    if icr is None:
        icr = 0.50
    adj += (icr - 0.50) * 0.10

    oar = signals.get("offer_acceptance_rate", -1)
    if oar is not None and oar >= 0:
        adj += oar * 0.05  # -1 == no history → neutral

    if (signals.get("saved_by_recruiters_30d", 0) or 0) > 0 \
            or (signals.get("profile_views_received_30d", 0) or 0) >= 50:
        adj += 0.03

    if signals.get("verified_email") and signals.get("verified_phone"):
        adj += 0.02

    return max(C.BEHAV_MIN, min(C.BEHAV_BASE + adj, C.BEHAV_MAX))
