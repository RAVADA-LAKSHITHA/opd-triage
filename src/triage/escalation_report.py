"""How many patients get escalated, how many of those are real emergencies,
and the recall/workload trade-off at different thresholds.

    python -m src.triage.escalation_report --data data/holdout.csv --model models/triage.joblib
"""
import argparse

import pandas as pd

from .data import load_raw
from .model import TriageModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/holdout.csv")
    ap.add_argument("--model", default="models/triage.joblib")
    a = ap.parse_args()

    df = load_raw(a.data)
    model = TriageModel.load(a.model)
    p = model.predict_df(df)

    emerg = df.is_emergency.values == 1
    esc = p.final_level.values <= 2
    abst = p.abstain.values
    flagged = esc | abst

    print(f"patients: {len(df)}   true emergencies: {emerg.sum()} ({emerg.mean():.1%})")
    print(f"escalated (level 1-2)       : {esc.sum()} ({esc.mean():.1%} of patients)")
    print(f"  precision (true emergency): {emerg[esc].mean():.1%}")
    print(f"abstained (nurse now)       : {abst.sum()} ({abst.mean():.1%})")
    print(f"escalated or abstained      : {flagged.mean():.1%}  -> emergency recall {flagged[emerg].mean():.1%}")
    print(f"At 800 walk-ins/day: ~{round(esc.mean() * 800)} escalations + ~{round(abst.mean() * 800)} nurse reviews")

    print("\nEscalated NON-emergencies by true level:")
    print(df.triage_level[esc & ~emerg].value_counts().sort_index().to_string())

    print("\nThreshold sweep (escalate if P(emergency) >= t, or a hard rule fires):")
    rule_fired = p.rule_level.notna().values
    rows = []
    for t in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]:
        e = (p.p_emergency.values >= t) | rule_fired
        rows.append(dict(threshold=t,
                         escalation_rate=e.mean(),
                         emergency_recall=e[emerg].mean(),
                         precision=emerg[e].mean() if e.any() else float("nan"),
                         missed=int((~e[emerg]).sum())))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()