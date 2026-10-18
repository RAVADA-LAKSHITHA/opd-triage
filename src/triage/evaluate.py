"""Cost-weighted evaluation.
    python -m src.triage.evaluate --data data/holdout.csv --model models/triage.joblib
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .costs import ABSTAIN_COST, COST
from .data import load_raw
from .model import TriageModel
from .queue_sim import compare


def evaluate(df, preds):
    y = df.triage_level.values.astype(int)
    final = preds.final_level.values.astype(int)
    abstain = preds.abstain.values.astype(bool)
    emerg = df.is_emergency.values == 1

    row_cost = np.where(abstain, ABSTAIN_COST, COST[y - 1, final - 1])
    strict = final <= 2
    safe = strict | abstain

    m = {
        "n": int(len(df)),
        "n_emergencies": int(emerg.sum()),
        "mean_cost": float(row_cost.mean()),
        "baseline_cost_all_level4": float(COST[y - 1, 3].mean()),
        "baseline_cost_all_level3": float(COST[y - 1, 2].mean()),
        "emergency_recall_strict": float(strict[emerg].mean()),
        "emergency_recall_with_abstain": float(safe[emerg].mean()),
        "emergencies_missed": int((~safe[emerg]).sum()),
        "emergency_under_triage_rate": float((final[emerg] > y[emerg]).mean()),
        "level_accuracy": float((final == y).mean()),
        "department_accuracy": float((preds.final_dept.values == df.department.values).mean()),
        "under_triage_rate": float((final > y).mean()),
        "over_triage_rate": float((final < y).mean()),
        "abstain_rate": float(abstain.mean()),
        "abstain_precision": float((final[abstain] != y[abstain]).mean()) if abstain.any() else None,
        "rules_override_rate": float(preds.rule_level.notna().mean()),
    }
    prio_levels = np.where(abstain, 2, final)       # "send to nurse now" = high priority
    m.update(compare(emerg, prio_levels, arrival_gap=4.0, doctors=2, service=7.0))
    return m, y, final


def print_report(m, y, final):
    print("=" * 60)
    print("SAFETY (reported first)")
    print(f"  Emergency sensitivity (strict)        : {m['emergency_recall_strict']:.1%}")
    print(f"  Emergency sensitivity (abstain=safe)  : {m['emergency_recall_with_abstain']:.1%}")
    print(f"  Emergencies missed entirely           : {m['emergencies_missed']} of {m['n_emergencies']}")
    print("COST (lower is better; 20x penalty for under-triaging emergencies)")
    print(f"  Model mean cost            : {m['mean_cost']:.4f}")
    print(f"  Baseline: all level 4      : {m['baseline_cost_all_level4']:.4f}")
    print(f"  Baseline: all level 3      : {m['baseline_cost_all_level3']:.4f}")
    print("ACCURACY")
    print(f"  Level accuracy             : {m['level_accuracy']:.1%}")
    print(f"  Department accuracy        : {m['department_accuracy']:.1%}")
    print(f"  Under-triage / over-triage : {m['under_triage_rate']:.2%} / {m['over_triage_rate']:.2%}")
    print("ABSTENTION & RULES")
    ap = m["abstain_precision"]
    print(f"  Abstain rate               : {m['abstain_rate']:.2%}")
    print(f"  Abstain precision          : {'n/a' if ap is None else f'{ap:.1%}'} (abstained cases the model would have got wrong)")
    print(f"  Rules override rate        : {m['rules_override_rate']:.2%}")
    print("WAIT TIME (simulated, assumptions: 2 doctors, 7 min consult, 1 arrival / 4 min)")
    print(f"  Emergency wait FCFS -> triage queue : {m['emergency_wait_fcfs']:.1f} -> "
          f"{m['emergency_wait_priority']:.1f} min ({m['emergency_wait_reduction_pct']:.0f}% less)")
    print("CONFUSION (rows = true level, cols = final level)")
    print(pd.crosstab(pd.Series(y, name="true"), pd.Series(final, name="pred")).to_string())
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/holdout.csv")
    ap.add_argument("--model", default="models/triage.joblib")
    ap.add_argument("--out", default="reports/metrics.json")
    a = ap.parse_args()
    df = load_raw(a.data)
    model = TriageModel.load(a.model)
    m, y, final = evaluate(df, model.predict_df(df))
    print_report(m, y, final)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(m, f, indent=2)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()