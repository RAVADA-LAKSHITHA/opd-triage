"""Harder holdout set for stress-testing (distribution shift vs. the training templates).

    python -m src.triage.generate_hard --n 5000 --out data/holdout_hard.csv
"""
import argparse
import random

from .generate_data import SCENARIOS, add_typo, generate

MILD = [c for s in SCENARIOS if s["level"] >= 4 for c in s["complaints"]]
EMERG = [c for s in SCENARIOS if s["level"] <= 2 for c in s["complaints"]]


def make_hard(n=5000, seed=2026):
    """Same labels as generate(n, seed), but noisier and more overlapping inputs."""
    rng = random.Random(seed)
    df = generate(n, seed)
    for i in df.index:
        lvl = int(df.at[i, "triage_level"])
        if lvl <= 2:
            if rng.random() < 0.30:       # emergency described with a mild-sounding complaint
                df.at[i, "complaint"] = add_typo(rng.choice(MILD), rng)
            if rng.random() < 0.30:       # emergency with near-normal vitals
                df.at[i, "sbp"] = rng.randint(105, 135)
                df.at[i, "pulse"] = rng.randint(62, 95)
                df.at[i, "spo2"] = rng.randint(96, 100)
        else:
            r = rng.random()
            if r < 0.08:                  # anxious patient using emergency words
                df.at[i, "complaint"] = add_typo(rng.choice(EMERG), rng)
            elif r < 0.15:                # non-emergency with abnormal-looking vitals
                df.at[i, "pulse"] = rng.randint(100, 125)
                df.at[i, "spo2"] = rng.randint(92, 95)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data/holdout_hard.csv")
    a = ap.parse_args()
    df = make_hard(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out}  (emergency rate {df.is_emergency.mean():.2%})")