"""Train, evaluate (asymmetric cost + emergency recall), simulate wait times, save model & metrics."""
import argparse, heapq, json, os
import numpy as np
import pandas as pd

from .data import generate
from .model import COST, ABSTAIN_COST, ABSTAIN_TAU, Triager, prep


def score(y, out, emerg):
    lvl, ab = out["level"].values, out["abstain"].values
    cost = np.where(ab, ABSTAIN_COST, COST[y - 1, lvl - 1])
    caught = (lvl <= 2) | ab
    return {"accuracy": float(((lvl == y) & ~ab).mean()), "within_1": float(((abs(lvl - y) <= 1) | ab).mean()),
            "mean_cost": float(cost.mean()), "emergency_recall": float(caught[emerg].mean()),
            "emergencies_missed": int((~caught[emerg]).sum()), "abstain_rate": float(ab.mean())}


def simulate(arr, prio, svc, docs=2):
    """Non-preemptive priority queue, `docs` doctors. Returns wait (min) per patient."""
    order, free, q = np.argsort(arr), [0.0] * docs, []
    heapq.heapify(free); wait = np.zeros(len(arr)); i, n = 0, len(arr)
    while i < n or q:
        t = heapq.heappop(free)
        if not q and arr[order[i]] > t:
            t = arr[order[i]]
        while i < n and arr[order[i]] <= t:
            j = order[i]; heapq.heappush(q, (prio[j], arr[j], j)); i += 1
        _, a, j = heapq.heappop(q)
        wait[j] = t - a; heapq.heappush(free, t + svc[j])
    return wait


def wait_study(test, level, mean_service=1.4, seed=0):
    rng, fc, tr = np.random.default_rng(seed), [], []
    for d in test["arrival_day"].unique():
        m = (test["arrival_day"] == d).values
        arr, svc = test.loc[m, "arrival_min"].values, rng.exponential(mean_service, m.sum())
        e = test.loc[m, "is_emergency"].values == 1
        fc.append(simulate(arr, np.zeros(m.sum()), svc)[e]); tr.append(simulate(arr, level[m], svc)[e])
    fc, tr = np.concatenate(fc), np.concatenate(tr)
    return {"fcfs_mean_wait_min": float(fc.mean()), "triage_mean_wait_min": float(tr.mean()),
            "reduction_pct": float(100 * (1 - tr.mean() / fc.mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/opd_synthetic.csv")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--min-recall", type=float, default=0.0, help="exit 1 if emergency recall is below this (CI gate)")
    a = ap.parse_args()
    if os.path.exists(a.data):
        df = pd.read_csv(a.data)
    else:
        df = generate(a.n); os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True); df.to_csv(a.data, index=False)
    cut = df["arrival_day"].max() * 0.8                         # time-based split: last 20% of days = test
    tr, te = df[df["arrival_day"] <= cut], df[df["arrival_day"] > cut].reset_index(drop=True)
    m = Triager().fit(tr)
    y, em = te["triage_level"].values.astype(int), te["is_emergency"].values == 1
    variants = {
        "1. accuracy-optimised ML": dict(use_cost=False, use_rules=False, tau=None),
        "2. + asymmetric (20x) decision": dict(use_cost=True, use_rules=False, tau=None),
        "3. + hard-coded rules": dict(use_cost=True, use_rules=True, tau=None),
        "4. + abstention (FINAL)": dict(use_cost=True, use_rules=True, tau=ABSTAIN_TAU),
    }
    res = {}
    print(f"train {len(tr)} / test {len(te)} | emergencies in test: {em.sum()}\n")
    print(f"{'variant':34s} {'acc':>6s} {'±1':>6s} {'cost':>6s} {'EMERG RECALL':>13s} {'missed':>7s} {'abstain':>8s}")
    for k, kw in variants.items():
        out = m.predict(te, **kw); s = score(y, out, em); res[k] = s
        print(f"{k:34s} {s['accuracy']:6.3f} {s['within_1']:6.3f} {s['mean_cost']:6.2f} {s['emergency_recall']:13.3f} {s['emergencies_missed']:7d} {s['abstain_rate']:8.3f}")
        if k.endswith("(FINAL)"): final = out
    dept_ok = float((final["department"].values == te["department"].values)[~final["abstain"].values].mean())
    w = wait_study(te, final["level"].values)
    res["department_accuracy_non_abstained"] = dept_ok; res["wait_time_emergencies"] = w
    print(f"\ndepartment accuracy (non-abstained): {dept_ok:.3f}")
    print(f"emergency wait vs FCFS: {w['fcfs_mean_wait_min']:.1f} -> {w['triage_mean_wait_min']:.1f} min ({w['reduction_pct']:.0f}% less)")
    os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
    m.save(); json.dump(res, open("results/metrics.json", "w"), indent=2)
    rec = res["4. + abstention (FINAL)"]["emergency_recall"]
    if rec < a.min_recall:
        raise SystemExit(f"FAIL: emergency recall {rec:.3f} < {a.min_recall}")


if __name__ == "__main__":
    main()
