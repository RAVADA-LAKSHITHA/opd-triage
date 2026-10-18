from src.triage.rules import apply_rules, clean_vitals


def test_zero_bp_treated_as_missing():
    v, bad = clean_vitals(sbp=0, spo2=98)
    assert v.sbp is None and "sbp" in bad


def test_96_degrees_converted_from_fahrenheit():
    v, bad = clean_vitals(temp=96)
    assert 35.5 < v.temp_c < 36.0 and not bad


def test_absurd_temp_rejected():
    v, bad = clean_vitals(temp=300)
    assert v.temp_c is None and "temp" in bad


def test_low_spo2_is_level_1():
    r = apply_rules("cough", spo2=85)
    assert r.level == 1


def test_hypotension_is_level_1():
    assert apply_rules("dizzy", sbp=70).level == 1


def test_chest_pain_english():
    r = apply_rules("severe chest pain since morning")
    assert r.level == 2 and r.department == "Cardiology"


def test_chest_pain_hinglish():
    assert apply_rules("seene mein dard ho raha hai").level == 2


def test_mild_fever_no_override():
    r = apply_rules("mild fever and cold", sbp=120, pulse=80, temp=37.8, spo2=98)
    assert r.level is None


def test_most_urgent_wins():
    r = apply_rules("chest pain", spo2=85)   # level 2 keyword + level 1 vitals
    assert r.level == 1

def test_fits_keyword_needs_word_start():
    assert apply_rules("this will benefit me").level is None
    assert apply_rules("fits aaye").level == 1
