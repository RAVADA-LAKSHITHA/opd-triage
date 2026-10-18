"""Hard-coded escalation layer.

Runs BEFORE any ML model and overrides it. Triage level 1 = most urgent,
5 = least urgent (ESI-style).
"""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Vitals:
    sbp: Optional[float] = None   # systolic BP
    dbp: Optional[float] = None   # diastolic BP
    pulse: Optional[float] = None
    temp_c: Optional[float] = None
    spo2: Optional[float] = None


@dataclass
class RuleResult:
    level: Optional[int] = None            # None = no override, let model decide
    reasons: list = field(default_factory=list)
    red_flags: list = field(default_factory=list)
    department: Optional[str] = None
    needs_recheck: list = field(default_factory=list)  # vitals that were invalid/missing


# ---------- 1. Vitals cleaning (handles 0 BP, 96 degree C, etc.) ----------
def clean_vitals(sbp=None, dbp=None, pulse=None, temp=None, spo2=None) -> tuple:
    """Return (Vitals, list_of_invalid_fields). Invalid values become None."""
    bad = []

    def num(x):
        try:
            if x is None or x == "":
                return None
            return float(x)
        except (TypeError, ValueError):
            return None

    sbp, dbp, pulse, temp, spo2 = map(num, (sbp, dbp, pulse, temp, spo2))

    if sbp is not None and not (50 <= sbp <= 260):
        bad.append("sbp"); sbp = None
    if dbp is not None and not (30 <= dbp <= 150):
        bad.append("dbp"); dbp = None
    if pulse is not None and not (20 <= pulse <= 250):
        bad.append("pulse"); pulse = None
    if spo2 is not None and not (50 <= spo2 <= 100):
        bad.append("spo2"); spo2 = None

    if temp is not None:
        if 25 <= temp <= 45:
            pass                                   # plausible Celsius
        elif 90 <= temp <= 110:
            temp = round((temp - 32) * 5 / 9, 1)   # looks like Fahrenheit
        else:
            bad.append("temp"); temp = None

    return Vitals(sbp, dbp, pulse, temp, spo2), bad


# ---------- 2. Critical-vitals rules ----------
def vitals_rules(v: Vitals) -> RuleResult:
    r = RuleResult()

    def hit(level, reason, flag, dept="Emergency"):
        # keep the MOST urgent (lowest number) level
        if r.level is None or level < r.level:
            r.level = level
            r.department = dept
        r.reasons.append(reason)
        r.red_flags.append(flag)

    if v.spo2 is not None:
        if v.spo2 < 90:
            hit(1, f"SpO2 {v.spo2}% < 90", "severe hypoxia")
        elif v.spo2 < 93:
            hit(2, f"SpO2 {v.spo2}% < 93", "low oxygen")
    if v.sbp is not None:
        if v.sbp < 90:
            hit(1, f"Systolic BP {v.sbp} < 90", "hypotension / shock")
        elif v.sbp >= 180:
            hit(2, f"Systolic BP {v.sbp} >= 180", "severe hypertension")
    if v.pulse is not None:
        if v.pulse < 40 or v.pulse > 150:
            hit(1, f"Pulse {v.pulse} critical", "critical heart rate")
        elif v.pulse > 130 or v.pulse < 50:
            hit(2, f"Pulse {v.pulse} abnormal", "abnormal heart rate")
    if v.temp_c is not None:
        if v.temp_c >= 40 or v.temp_c < 35:
            hit(2, f"Temp {v.temp_c}C critical", "extreme temperature")
    return r


# ---------- 3. Red-flag complaint keywords (English + Hinglish) ----------
# (keyword, level, flag name, department)
KEYWORD_RULES = [
    ("unconscious", 1, "unconscious", "Emergency"),
    ("behosh", 1, "unconscious", "Emergency"),
    ("not breathing", 1, "not breathing", "Emergency"),
    ("seizure", 1, "seizure", "Emergency"),
    ("fits", 1, "seizure", "Emergency"),
    ("daura", 1, "seizure/attack", "Emergency"),
    ("chest pain", 2, "chest pain", "Cardiology"),
    ("seene mein dard", 2, "chest pain", "Cardiology"),
    ("chest mein dard", 2, "chest pain", "Cardiology"),
    ("seene me dard", 2, "chest pain", "Cardiology"),
    ("breathless", 2, "breathing difficulty", "Emergency"),
    ("difficulty breathing", 2, "breathing difficulty", "Emergency"),
    ("saans", 2, "breathing difficulty", "Emergency"),
    ("heavy bleeding", 2, "heavy bleeding", "Emergency"),
    ("vomiting blood", 2, "vomiting blood", "Emergency"),
    ("khoon", 2, "bleeding", "Emergency"),
    ("slurred speech", 2, "stroke sign", "Neurology"),
    ("face droop", 2, "stroke sign", "Neurology"),
    ("paralysis", 2, "stroke sign", "Neurology"),
    ("snake bite", 2, "snake bite", "Emergency"),
    ("poison", 2, "poisoning", "Emergency"),
    ("suicid", 2, "self-harm risk", "Psychiatry"),
    # Telugu (transliterated)
    ("gunde noppi", 2, "chest pain", "Cardiology"),
    ("pakshavatham", 2, "stroke sign", "Neurology"),
    ("moorcha", 1, "seizure/attack", "Emergency"),
    ("spruha ledu", 1, "unconscious", "Emergency"),
]


def keyword_rules(text: str) -> RuleResult:
    r = RuleResult()
    t = (text or "").lower()
    for kw, level, flag, dept in KEYWORD_RULES:
        if re.search(r"\b" + re.escape(kw), t):   # word-start match: 'fits' != 'benefits'
            if r.level is None or level < r.level:
                r.level = level
                r.department = dept
            if flag not in r.red_flags:
                r.red_flags.append(flag)
            r.reasons.append(f'complaint mentions "{kw}"')
    return r


# ---------- 4. Combined entry point ----------
def apply_rules(complaint: str, sbp=None, dbp=None, pulse=None, temp=None, spo2=None) -> RuleResult:
    v, bad = clean_vitals(sbp, dbp, pulse, temp, spo2)
    vr = vitals_rules(v)
    kr = keyword_rules(complaint)

    out = RuleResult(needs_recheck=bad)
    out.reasons = vr.reasons + kr.reasons
    out.red_flags = list(dict.fromkeys(vr.red_flags + kr.red_flags))

    candidates = [x for x in (vr, kr) if x.level is not None]
    if candidates:
        best = min(candidates, key=lambda x: x.level)
        out.level = best.level
        out.department = best.department
    return out