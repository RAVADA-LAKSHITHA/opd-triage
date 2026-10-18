from src.triage.generate_data import generate
from src.triage.generate_hard import make_hard


def test_hard_changes_inputs_but_keeps_labels():
    easy, hard = generate(1500, seed=4), make_hard(1500, seed=4)
    assert list(easy.columns) == list(hard.columns)
    assert (easy.triage_level == hard.triage_level).all()
    assert (easy.complaint != hard.complaint).mean() > 0.05