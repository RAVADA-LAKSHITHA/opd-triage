from src.triage.data import add_clean_vitals, add_rule_outputs
from src.triage.generate_data import generate


def test_generator_shape_and_labels():
    df = generate(5000, seed=1)
    assert len(df) == 5000
    assert set(df.triage_level.unique()) <= {1, 2, 3, 4, 5}
    assert 0.025 < df.is_emergency.mean() < 0.06


def test_generator_has_messy_vitals():
    df = generate(5000, seed=1)
    assert df.sbp.isna().sum() > 0
    assert (df.sbp == 0).sum() > 0


def test_cleaning_removes_invalid_values():
    df = add_clean_vitals(generate(3000, seed=2))
    assert df.sbp_c.dropna().between(50, 260).all()
    assert df.spo2_c.dropna().between(50, 100).all()
    assert df.temp_c.dropna().between(25, 45).all()
    assert df.sbp_missing.sum() > 0


def test_rules_rarely_over_escalate():
    df = add_rule_outputs(generate(3000, seed=3))
    non_emerg = df[df.is_emergency == 0]
    assert non_emerg.rule_level.le(2).mean() < 0.02