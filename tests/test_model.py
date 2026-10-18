import pytest

from src.triage.generate_data import generate
from src.triage.model import TriageModel


@pytest.fixture(scope="module")
def fitted():
    df = generate(3500, seed=7)
    train, test = df.iloc[:2500], df.iloc[2500:]
    return TriageModel().fit(train), test


def test_emergency_recall_with_abstain(fitted):
    model, test = fitted
    p = model.predict_df(test)
    emerg = test.is_emergency.values == 1
    escalated = (p.final_level.values <= 2) | p.abstain.values
    assert escalated[emerg].mean() >= 0.85


def test_chest_pain_escalates(fitted):
    model, _ = fitted
    r = model.predict_one("severe chest pain with sweating", age=60, sbp=150, pulse=105, spo2=95)
    assert r["final_level"] <= 2


def test_vague_complaint_critical_vitals_rule_override(fitted):
    model, _ = fitted
    r = model.predict_one("feeling fine just checkup", spo2=82)
    assert r["final_level"] == 1 and not r["abstain"]


def test_mild_case_not_escalated(fitted):
    model, _ = fitted
    r = model.predict_one("mild fever and cold", age=25, sbp=118, dbp=78, pulse=80, temp=37.9, spo2=98)
    assert r["final_level"] >= 4