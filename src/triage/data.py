"""Synthetic OPD intake generator (same schema as the problem statement).

Swap in the organisers' CSV any time: `python -m src.triage.evaluate --data path.csv`.
Column aliases are handled in model.prep().
"""
import argparse
import numpy as np
import pandas as pd

# vitals profile: (sbp, dbp, pulse, temp, spo2)
VIT = {
    "normal": (120, 78, 80, 36.9, 98), "fever": (118, 76, 96, 38.7, 97),
    "cardiac": (150, 92, 108, 36.9, 94), "stroke": (178, 100, 84, 36.9, 96),
    "hypox": (118, 76, 118, 37.4, 86), "seizure": (140, 90, 110, 37.8, 93),
    "bleed": (92, 58, 122, 36.8, 96), "sepsis": (84, 50, 128, 39.6, 92),
    "wheeze": (130, 80, 102, 37.0, 94),
}
D = ["", "since morning", "since 2 days", "for 3 days", "since last night", "kal se", "2 din se", "subah se", "ninna nunchi"]
# name, dept, {level: p}, weight, vitals, (age_lo, age_hi), complaint templates
C = [
    # ---- true emergencies (~4% of traffic) ----
    ("cardiac", "Cardiology", {2: .8, 1: .2}, .012, "cardiac", (40, 85), ["chest pain {d}", "severe chest pain radiating to left arm", "seene mein dard {d}", "gunde noppi {d}", "chest tightness and sweating", "heaviness in chest with sweating"]),
    ("stroke", "Neurology", {2: 1}, .007, "stroke", (45, 88), ["slurred speech and face droop", "sudden weakness on one side", "paralysis of left hand", "pakshavatham", "achanak ek taraf kamzori aur bolne mein dikkat"]),
    ("resp", "Emergency", {1: .4, 2: .6}, .008, "hypox", (5, 85), ["breathless and blue lips", "difficulty breathing {d}", "saans lene mein takleef {d}", "severe breathlessness at rest"]),
    ("seizure", "Emergency", {1: 1}, .004, "seizure", (3, 70), ["seizure at home", "daura pada", "fits aaye", "unconscious after fit", "behosh ho gaya", "moorcha"]),
    ("bleed", "Emergency", {2: 1}, .005, "bleed", (10, 80), ["heavy bleeding after accident", "vomiting blood", "khoon bahut beh raha hai", "deep cut bleeding not stopping"]),
    ("sepsis", "Emergency", {1: 1}, .004, "sepsis", (20, 85), ["very high fever and confusion", "drowsy and not responding well", "unconscious {d}", "high fever with shivering and weakness"]),
    # ---- urgent ----
    ("abdomen", "General Surgery", {3: 1}, .04, "normal", (12, 75), ["severe stomach pain {d}", "pet mein tez dard {d}", "kadupu noppi {d}", "pain in right lower abdomen"]),
    ("fracture", "Orthopedics", {3: 1}, .03, "normal", (5, 80), ["fell down, leg swelling cannot walk", "haath toot gaya shayad", "wrist pain after fall"]),
    ("asthma", "Pulmonology", {3: 1}, .03, "wheeze", (5, 75), ["wheezing and cough {d}", "dama attack mild", "chest tight with cough at night"]),
    ("pregnancy", "Gynecology", {3: 1}, .02, "normal", (18, 40), ["pregnant 7 months, pain in lower abdomen", "garbh mein dard", "pregnant with bleeding spots"]),
    ("child_fever", "Pediatrics", {3: .6, 4: .4}, .025, "fever", (0, 10), ["baby fever {d} not feeding", "bachche ko tez bukhar", "child fever and vomiting"]),
    # ---- semi-urgent / routine ----
    ("fever", "General Medicine", {4: .7, 5: .3}, .12, "fever", (5, 80), ["fever {d}", "bukhar hai {d}", "jwaram {d}", "body pain and fever"]),
    ("uti", "General Medicine", {4: 1}, .03, "normal", (15, 80), ["burning urination", "peshab mein jalan", "pain while passing urine"]),
    ("ent", "ENT", {4: 1}, .04, "normal", (4, 70), ["ear pain {d}", "kaan mein dard", "throat pain difficulty swallowing"]),
    ("backpain", "Orthopedics", {4: .6, 5: .4}, .05, "normal", (25, 80), ["back pain {d}", "kamar dard", "knee pain on walking"]),
    ("diabetes", "General Medicine", {4: 1}, .04, "normal", (30, 80), ["sugar high, feeling tired", "sugar badh gaya", "excess thirst and frequent urination"]),
    ("gastro", "General Medicine", {4: 1}, .05, "normal", (3, 80), ["loose motions {d}", "vomiting and loose motion", "dast aur ulti", "vaanthulu {d}"]),
    ("cold", "General Medicine", {5: 1}, .15, "normal", (3, 80), ["cold and cough", "mild cough {d}", "jalubu dagu", "khansi zukham", "runny nose"]),
    ("headache", "General Medicine", {5: 1}, .06, "normal", (10, 75), ["headache {d}", "sir dard", "tala noppi"]),
    ("skin", "Dermatology", {5: 1}, .07, "normal", (3, 70), ["itching rash on arm", "khujli aur daane", "skin allergy"]),
    ("eye", "Ophthalmology", {5: 1}, .04, "normal", (3, 70), ["eye redness", "aankh lal hai", "itchy eyes"]),
    ("dental", "Dental", {5: 1}, .04, "normal", (8, 70), ["toothache", "daant dard {d}"]),
    ("refill", "General Medicine", {5: 1}, .04, "normal", (30, 85), ["bp tablet refill", "routine checkup", "medicine lena hai"]),
    ("vague", "General Medicine", {5: .5, 4: .4, 3: .1}, .03, "normal", (5, 85), ["not feeling well", "weakness", "bas aise hi", "tabiyat theek nahi"]),
]
VAGUE = C[-1][-1]
EMERG = {"cardiac", "stroke", "resp", "seizure", "bleed", "sepsis"}
SD = np.array([12, 8, 9, .4, 1.2])
KC = ["", "none", "diabetes", "hypertension", "asthma", "heart disease", "diabetes;hypertension"]
MED = ["", "metformin", "amlodipine", "salbutamol inhaler", "aspirin", "insulin"]


def typo(s, rng):
    if len(s) > 6 and rng.random() < .08:
        i = rng.integers(1, len(s) - 1)
        s = s[:i] + s[i + 1:]
    return s.upper() if rng.random() < .05 else s


def generate(n=30000, seed=7, per_day=800):
    rng = np.random.default_rng(seed)
    w = np.array([c[3] for c in C]); w = w / w.sum()
    rows = []
    for i, k in enumerate(rng.choice(len(C), n, p=w)):
        name, dept, lv, _, vit, (a0, a1), tpl = C[k]
        emerg = name in EMERG
        level = int(rng.choice(list(lv), p=list(lv.values())))
        t = VAGUE if (emerg and rng.random() < .15) else tpl   # patients under-report
        text = typo(str(rng.choice(t)).format(d=rng.choice(D)).strip(), rng)
        v = np.array(VIT[vit]) + rng.normal(0, 1, 5) * SD * (1.4 if emerg else 1)
        v[4] = min(v[4], 100)
        if emerg and rng.random() < .25:                        # some emergencies look normal on vitals
            v = np.array(VIT["normal"]) + rng.normal(0, 1, 5) * SD
        sbp, dbp, pulse, temp, spo2 = v
        if rng.random() < .015: sbp = dbp = 0                   # mis-entered BP
        if rng.random() < .015: temp = temp * 9 / 5 + 32        # Fahrenheit typed as C
        vals = [round(sbp), round(dbp), round(pulse), round(temp, 1), round(spo2)]
        vals = [np.nan if rng.random() < .05 else x for x in vals]   # missing
        if not emerg and rng.random() < .03:                    # label noise (never on emergencies)
            level = int(np.clip(level + rng.choice([-1, 1]), 3, 5))
        age = int(rng.integers(a0, a1 + 1))
        kc = rng.choice(["heart disease", "hypertension", "diabetes"]) if name == "cardiac" and rng.random() < .5 else rng.choice(KC)
        day = i // per_day
        rows.append([i, day, 0, age, rng.choice(["M", "F"]), *vals, text, kc, rng.choice(MED), level, dept, int(emerg)])
    df = pd.DataFrame(rows, columns=["id", "arrival_day", "arrival_min", "age", "sex", "bp_sys", "bp_dia", "pulse", "temp", "spo2",
                                     "complaint", "known_conditions", "medications", "triage_level", "department", "is_emergency"])
    df["arrival_min"] = df.groupby("arrival_day")["id"].transform(lambda s: np.sort(rng.uniform(0, 600, len(s))))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", default="data/opd_synthetic.csv")
    a = ap.parse_args()
    d = generate(a.n)
    d.to_csv(a.out, index=False)
    print(f"{len(d)} rows -> {a.out} | emergencies {d.is_emergency.mean():.1%} | levels {d.triage_level.value_counts().sort_index().to_dict()}")
