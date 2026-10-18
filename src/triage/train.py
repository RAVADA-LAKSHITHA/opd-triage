"""Train + quick report.  python -m src.triage.train --data data/synthetic_opd.csv"""
import argparse
import os

from sklearn.model_selection import train_test_split

from .costs import COST
from .data import load_raw
from .model import TriageModel


def mean_cost(true, pred):
    return float(COST[true - 1, pred - 1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_opd.csv")
    ap.add_argument("--out", default="models/triage.joblib")
    a = ap.parse_args()

    df = load_raw(a.data)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.triage_level)
    model = TriageModel().fit(train)
    p = model.predict_df(test)

    y = test.triage_level.values
    emerg = test.is_emergency.values == 1
    abstain = p.abstain.values

    print(f"test rows: {len(test)}  true emergencies: {emerg.sum()}")
    print("--- model only ---")
    print(f"mean cost: {mean_cost(y, p.model_level.values):.4f}")
    print(f"emergency recall: {(p.model_level.values[emerg] <= 2).mean():.1%}")
    print("--- model + rules + abstain ---")
    print(f"mean cost: {mean_cost(y, p.final_level.values):.4f}")
    print(f"level accuracy: {(p.final_level.values == y).mean():.1%}")
    print(f"department accuracy: {(p.final_dept.values == test.department.values).mean():.1%}")
    esc = (p.final_level.values <= 2) | abstain
    print(f"emergency recall (abstain counts as escalated): {esc[emerg].mean():.1%}")
    print(f"emergency recall (strict, no abstain credit): {(p.final_level.values[emerg] <= 2).mean():.1%}")
    print(f"abstain rate: {abstain.mean():.2%}")
    print(f"emergencies under-triaged and NOT abstained: {(~esc[emerg]).sum()}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    model.save(a.out)
    print(f"saved model -> {a.out}")


if __name__ == "__main__":
    main()