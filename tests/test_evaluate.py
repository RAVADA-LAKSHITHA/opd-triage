from src.triage.evaluate import evaluate
from src.triage.generate_data import generate
from src.triage.model import TriageModel
from src.triage.queue_sim import compare


def test_evaluate_metrics_and_baseline():
    df = generate(3500, seed=11)
    train, test = df.iloc[:2500], df.iloc[2500:].reset_index(drop=True)
    model = TriageModel().fit(train)
    m, _, _ = evaluate(test, model.predict_df(test))
    assert m["mean_cost"] < m["baseline_cost_all_level4"]
    assert m["emergency_recall_with_abstain"] >= m["emergency_recall_strict"]
    assert 0 <= m["abstain_rate"] <= 1


def test_priority_queue_reduces_emergency_wait():
    df = generate(1500, seed=5)
    r = compare(df.is_emergency.values == 1, df.triage_level.values,
                arrival_gap=4.0, doctors=2, service=7.0)
    assert r["emergency_wait_priority"] < r["emergency_wait_fcfs"]