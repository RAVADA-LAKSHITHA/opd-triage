"""Fill the results section of README.md from the trained model + holdout data.

    python -m src.triage.update_readme --demo-url https://your-app.streamlit.app
"""
import argparse
import re

from .data import load_raw
from .evaluate import evaluate
from .model import TriageModel

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def build_block(df, model, demo_url=""):
    p = model.predict_df(df)
    m, _, _ = evaluate(df, p)
    emerg = df.is_emergency.values == 1
    esc = p.final_level.values <= 2
    precision = emerg[esc].mean() if esc.any() else float("nan")
    ap = m["abstain_precision"]

    lines = [START, "## Results", ""]
    if demo_url:
        lines += [f"**Live demo:** {demo_url}", ""]
    lines += [
        f"Measured on {m['n']:,} unseen synthetic holdout records ({m['n_emergencies']} true emergencies). "
        "Synthetic data: treat as optimistic (see Limitations).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Emergency sensitivity (strict, rules + model)** | **{m['emergency_recall_strict']:.1%}** |",
        f"| Emergency sensitivity (abstain counts as escalated) | {m['emergency_recall_with_abstain']:.1%} |",
        f"| Emergencies missed entirely | {m['emergencies_missed']} of {m['n_emergencies']} |",
        f"| Mean cost, our system | {m['mean_cost']:.3f} |",
        f"| Mean cost, baseline: everyone level 4 / everyone level 3 | "
        f"{m['baseline_cost_all_level4']:.3f} / {m['baseline_cost_all_level3']:.3f} |",
        f"| Level accuracy / department accuracy | {m['level_accuracy']:.1%} / {m['department_accuracy']:.1%} |",
        f"| Under-triage / over-triage rate | {m['under_triage_rate']:.2%} / {m['over_triage_rate']:.2%} |",
        f"| Patients escalated (level 1-2) / precision of escalation | {esc.mean():.1%} / {precision:.1%} |",
        f"| Abstain rate / abstain precision | {m['abstain_rate']:.2%} / "
        f"{'n/a' if ap is None else f'{ap:.1%}'} |",
        f"| Hard rules overrode the model | {m['rules_override_rate']:.2%} of patients |",
        f"| Simulated emergency wait, first-come-first-served -> triage queue | "
        f"{m['emergency_wait_fcfs']:.1f} -> {m['emergency_wait_priority']:.1f} min "
        f"({m['emergency_wait_reduction_pct']:.0f}% less) |",
        "",
        "Wait simulation assumptions: 2 doctors, 7 minute consult, one arrival every 4 minutes.",
        END,
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/holdout.csv")
    ap.add_argument("--model", default="models/triage.joblib")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--demo-url", default="")
    a = ap.parse_args()

    block = build_block(load_raw(a.data), TriageModel.load(a.model), a.demo_url)
    text = open(a.readme, encoding="utf-8").read()
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit("Markers not found in README.md")
    open(a.readme, "w", encoding="utf-8").write(pattern.sub(lambda _: block, text))
    print(block)
    print(f"\nupdated {a.readme}")


if __name__ == "__main__":
    main()