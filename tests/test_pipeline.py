import pytest
from src.triage.data import generate
from src.triage.evaluate import simulate
from src.triage.model import COST, Triager


@pytest.fixture(scope="module")
def model_and_data():
    df = generate(6000, seed=1)
    return Triager().fit(df.iloc[:5000]), df.iloc[5000:].reset_index(drop=True)


def test_generator_schema_and_emergency_rate():
    df = generate(3000)
    assert {"complaint", "bp_sys", "spo2", "triage_level", "department", "is_emergency"} <= set(df.columns)
    assert 0.02 < df.is_emergency.mean() < 0.07


def test_cost_matrix_is_asymmetric():
    assert COST[0, 4] == 80 and COST[4, 0] == 4          # L1->L5 vs L5->L1
    assert COST[1, 4] / COST[4, 3] == 60                  # under-triage emergency >> over-triage mild


def test_emergency_recall_high(model_and_data):
    m, te = model_and_data
    out = m.predict(te)
    em = te.is_emergency.values == 1
    assert ((out.level <= 2) | out.abstain).values[em].mean() >= 0.95


def test_rules_override_model(model_and_data):
    m, _ = model_and_data
    assert m.triage_one("mild cold", sbp=70, pulse=80, spo2=98)["level"] == 1
    r = m.triage_one("seene mein dard", sbp=120, spo2=98)
    assert r["level"] <= 2 and "chest pain" in r["red_flags"]


def test_garbage_vitals_dont_crash(model_and_data):
    m, _ = model_and_data
    r = m.triage_one("fever", sbp=0, dbp=0, temp=96, spo2=None)
    assert "sbp" in r["needs_recheck"]


def test_priority_queue_beats_fcfs():
    import numpy as np
    arr = np.arange(20) * 0.5; svc = np.full(20, 2.0); prio = np.r_[np.full(19, 5), 1]
    assert simulate(arr, prio, svc)[-1] < simulate(arr, np.zeros(20), svc)[-1]
