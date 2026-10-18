"""Synthetic OPD intake data generator (mimics the hackathon dataset format).

Usage:
    python -m src.triage.generate_data --n 30000 --out data/synthetic_opd.csv
"""
import argparse
import random

import pandas as pd

NORMAL = dict(sbp=(105, 135), dbp=(65, 85), pulse=(62, 95), temp=(36.4, 37.2), spo2=(96, 100))


def S(name, w, level, dept, complaints, age=(5, 80), **vit):
    return dict(name=name, w=w, level=level, dept=dept, complaints=complaints, age=age, vit=vit)


SCENARIOS = [
    # ---- true emergencies (levels 1-2) : ~4.1% total ----
    S("shock_collapse", 0.5, 1, "Emergency",
      ["patient unconscious", "behosh ho gaya hai", "collapsed at home not responding",
       "suddenly fell down and not waking"],
      age=(30, 85), sbp=(55, 88), pulse=(125, 165), spo2=(70, 89)),
    S("heart_attack", 1.5, 2, "Cardiology",
      ["severe chest pain radiating to left arm", "chest pain with sweating",
       "seene mein dard aur pasina", "seene me bahut dard ho raha hai",
       "gunde lo noppi chaala ekkuva", "chaati noppi sweating"],
      age=(45, 85), sbp=(100, 185), pulse=(90, 125), spo2=(92, 98)),
    S("resp_distress", 1.0, 2, "Emergency",
      ["breathless since morning", "difficulty breathing cannot speak full sentence",
       "saans nahi aa rahi", "saans phool rahi hai", "oopiri aadatam ledu"],
      age=(10, 85), pulse=(100, 135), temp=(36.5, 38.5), spo2=(80, 91)),
    S("stroke", 0.5, 2, "Neurology",
      ["slurred speech and face droop", "sudden weakness on left side",
       "ek taraf ka haath pair kaam nahi kar raha", "mukham vankara poyindi, matladaleka pothunnadu"],
      age=(55, 88), sbp=(160, 210), pulse=(70, 100)),
    S("seizure", 0.3, 1, "Emergency",
      ["seizure at home", "child had fits for 5 minutes", "daura pada hai", "fits vachindi"],
      age=(2, 60), pulse=(90, 140), spo2=(88, 97)),
    S("major_bleeding", 0.3, 2, "Emergency",
      ["heavy bleeding after road accident", "bahut khoon beh raha hai", "head injury accident bleeding",
       "accident ayyindi chaala raktam"],
      age=(15, 70), sbp=(70, 110), pulse=(105, 140)),
    # ---- level 3 (urgent) : 20% ----
    S("fever_vomiting", 8, 3, "General Medicine",
      ["high fever for 3 days with vomiting", "tez bukhar aur ulti", "jwaram ekkuva, vaanthulu"],
      temp=(38.6, 39.9), pulse=(95, 118)),
    S("abdominal_pain", 6, 3, "Surgery",
      ["severe stomach pain since last night", "pet mein tez dard", "kadupu noppi chaala ekkuva"],
      pulse=(85, 110)),
    S("suspected_fracture", 4, 3, "Orthopedics",
      ["fell down and leg is swollen cannot walk", "haath mein fracture jaisa dard", "kaalu vaapu, nadavaleka pothunnanu"]),
    S("asthma_mild", 2, 3, "Pulmonology",
      ["wheezing and cough, using inhaler", "dama ki takleef badh gayi", "ibbandi ga undi, wheezing"],
      spo2=(93, 96)),
    # ---- level 4 (less urgent) : 40% ----
    S("mild_fever_cold", 22, 4, "General Medicine",
      ["mild fever and cold", "halka bukhar aur khansi", "jwaram jalubu", "cold cough since 2 days"],
      temp=(37.5, 38.4)),
    S("skin_rash", 6, 4, "Dermatology",
      ["itching rash on arms", "skin pe daane aur khujli", "chermam meeda daddurlu"]),
    S("ent_pain", 6, 4, "ENT",
      ["ear pain since yesterday", "gale mein dard aur kharash", "chevi noppi"]),
    S("musculoskeletal", 6, 4, "Orthopedics",
      ["knee pain while walking", "kamar dard kai din se", "mokaalu noppi"]),
    # ---- level 5 (non-urgent) : 36% ----
    S("refill", 12, 5, "General Medicine",
      ["need medicine refill for sugar", "BP ki goli khatam ho gayi", "sugar tablets kavali"],
      age=(35, 85), sbp=(125, 150)),
    S("mild_headache", 10, 5, "General Medicine",
      ["mild headache", "halka sar dard", "thala noppi konchem"]),
    S("checkup", 8, 5, "General Medicine",
      ["routine health checkup", "general checkup karwana hai", "health checkup kosam vachanu"]),
    S("minor_cut", 6, 5, "Surgery",
      ["small cut on finger", "chhoti chot lagi hai", "chinna gaayam ayyindi"]),
]

VAGUE = ["feeling weak", "not feeling well", "thakaan lag rahi hai", "just come check me", "baagaledu"]
DRAMATIC = ["very severe pain please see me first", "mujhe jaldi dikhao bahut takleef hai", "cannot wait please urgent"]

COND_MEDS = {
    "diabetes": "metformin", "hypertension": "amlodipine", "asthma": "salbutamol inhaler",
    "CAD": "aspirin, atorvastatin", "CKD": "none", "thyroid": "thyroxine",
}


def add_typo(s, rng):
    if len(s) < 6 or rng.random() > 0.25:
        return s
    i = rng.randrange(1, len(s) - 1)
    mode = rng.choice(["drop", "swap", "dup"])
    if mode == "drop":
        return s[:i] + s[i + 1:]
    if mode == "swap":
        return s[:i - 1] + s[i] + s[i - 1] + s[i + 1:]
    return s[:i] + s[i] + s[i:]


def sample_vitals(sc, rng):
    v = {}
    for k, (lo, hi) in NORMAL.items():
        lo, hi = sc["vit"].get(k, (lo, hi))
        v[k] = round(rng.uniform(lo, hi), 1) if k == "temp" else int(rng.uniform(lo, hi))
    return v


def corrupt(v, rng):
    """Simulate missing / mis-entered vitals. Ground truth label is NOT affected."""
    if rng.random() < 0.02:
        return {k: None for k in v}
    v = dict(v)
    for k in v:
        if rng.random() < 0.04:
            v[k] = None
    if v["sbp"] is not None and rng.random() < 0.02:
        v["sbp"] = 0
    if v["spo2"] is not None and rng.random() < 0.01:
        v["spo2"] = 0
    if v["temp"] is not None and rng.random() < 0.03:
        v["temp"] = round(v["temp"] * 9 / 5 + 32, 1)   # entered in Fahrenheit
    return v


def generate(n=30000, seed=42):
    rng = random.Random(seed)
    weights = [s["w"] for s in SCENARIOS]
    rows = []
    for i in range(n):
        sc = rng.choices(SCENARIOS, weights=weights)[0]
        v = corrupt(sample_vitals(sc, rng), rng)

        complaint = rng.choice(sc["complaints"])
        if sc["level"] <= 2 and rng.random() < 0.15:        # under-reporting
            complaint = rng.choice(VAGUE)
        elif sc["level"] >= 4 and rng.random() < 0.05:      # over-reporting
            complaint = rng.choice(DRAMATIC)
        complaint = add_typo(complaint, rng)

        conds = rng.sample(list(COND_MEDS), k=rng.choice([0, 0, 1, 1, 2]))
        rows.append(dict(
            patient_id=f"P{i:06d}",
            age=int(rng.uniform(*sc["age"])),
            sex=rng.choice(["M", "F"]),
            sbp=v["sbp"], dbp=v["dbp"], pulse=v["pulse"], temp=v["temp"], spo2=v["spo2"],
            complaint=complaint,
            conditions=";".join(conds) if conds else "none",
            medications="; ".join(COND_MEDS[c] for c in conds) if conds else "none",
            triage_level=sc["level"],
            department=sc["dept"],
            is_emergency=int(sc["level"] <= 2),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_opd.csv")
    a = ap.parse_args()
    df = generate(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out}")
    print(df.triage_level.value_counts().sort_index().to_string())
    print(f"emergency rate: {df.is_emergency.mean():.3%}")