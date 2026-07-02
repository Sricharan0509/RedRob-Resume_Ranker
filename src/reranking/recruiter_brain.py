"""
recruiter_brain.py — the composite scoring matrix (Phase 3, "secret sauce").

Turns the per-candidate channels into one hireability score that mimics a great
recruiter: it weighs *fit* (structured rubric + semantic retrieval), gates on
*role* (title ∪ career, never skills), modifies by *availability* (behavioral),
and hard-kills the *impossible* (honeypots).

    recruiter_fit = W_STRUCT*S_struct + W_SIM*Sim_hybrid          # Phase 3
    fit           = (1-CROSS_BLEND)*recruiter_fit + CROSS_BLEND*S_cross  # +Phase 4 (shortlist)
    base          = role_gate * fit
    final         = base * behavioral_mult * honeypot_kill

Vectorized for the full pool; the cross-encoder blend is applied only to the
shortlist rows.
"""

import numpy as np
from .. import config as C


def recruiter_fit(s_struct: np.ndarray, sim_hybrid: np.ndarray) -> np.ndarray:
    return C.W_STRUCT * s_struct + C.W_SIM * sim_hybrid


def blend_cross(recruiter_fit_vals: np.ndarray, s_cross: np.ndarray) -> np.ndarray:
    """Applied to shortlist rows only (Phase 4)."""
    return (1.0 - C.CROSS_BLEND) * recruiter_fit_vals + C.CROSS_BLEND * s_cross


def finalize(fit: np.ndarray, role_gate: np.ndarray,
             behavioral: np.ndarray, honeypot_kill: np.ndarray) -> np.ndarray:
    return role_gate * fit * behavioral * honeypot_kill
