# OPD Triage Assistant

![CI](https://github.com/RAVADA-LAKSHITHA/opd-triage/actions/workflows/ci.yml/badge.svg)

Safety-first triage decision support for an overloaded government hospital OPD (800 walk-ins/day, 2 doctors).
From a short intake form and a free-text complaint (English, Hinglish, transliterated Indian languages),
it assigns a **triage level (1-5)**, recommends a **department**, lists **red-flag symptoms**, and says
**"CANNOT ASSESS - send to nurse now"** when it is not sure.

Built for the Talentap Mega Hackathon, Problem H1. Decision support only: a clinician makes the final call.

<!-- RESULTS:START -->
## Results

**Live demo:** https://opd-triage-jebn7goobs2ndahdnklext.streamlit.app/

Measured on 5,000 unseen synthetic holdout records (208 true emergencies). Synthetic data: treat as optimistic (see Limitations).

| Metric | Value |
|---|---|
| **Emergency sensitivity (strict, rules + model)** | **100.0%** |
| Emergency sensitivity (abstain counts as escalated) | 100.0% |
| Emergencies missed entirely | 0 of 208 |
| Mean cost, our system | 0.036 |
| Mean cost, baseline: everyone level 4 / everyone level 3 | 2.808 / 2.112 |
| Level accuracy / department accuracy | 97.8% / 98.2% |
| Under-triage / over-triage rate | 0.46% / 1.78% |
| Patients escalated (level 1-2) / precision of escalation | 4.3% / 96.3% |
| Abstain rate / abstain precision | 0.80% / 47.5% |
| Hard rules overrode the model | 2.94% of patients |
| Simulated emergency wait, first-come-first-served -> triage queue | 9.0 -> 1.8 min (80% less) |

Wait simulation assumptions: 2 doctors, 7 minute consult, one arrival every 4 minutes.
<!-- RESULTS:END -->

## How it works

```
 intake form + complaint
          |
          v
 [1] Vitals cleaning ---- invalid values (BP 0, SpO2 0, temp 300) -> ignored + "re-measure" notice
          |                 temp 90-110 treated as Fahrenheit and converted
          +----------------------------+
          v                            v
 [2] HARD-CODED RULES            [3] ML model
     critical vitals +               char n-gram TF-IDF (complaint)
     red-flag keywords               + cleaned vitals + missing flags + conditions
     (override the model)            -> probability per level 1-5
          |                            |
          |                            v
          |                      [4] Expected-cost decision
          |                          pick the level with the lowest expected cost
          |                          under the asymmetric cost matrix
          |                            |
          |                            v
          |                      [5] Abstention
          |                          confidence < 0.60 and no rule fired
          |                          -> "CANNOT ASSESS - send to nurse now"
          v                            v
        final level = the MORE URGENT of rule level and model level
```

**Why char n-grams:** patient text is messy (typos, Hinglish, transliteration). Character n-grams
tolerate spelling variation without needing a translation step, and run in milliseconds,
which fits the nurse's 60-second budget.

## Problem requirements -> where they live

| Requirement | Implementation |
|---|---|
| Triage level 1-5, department, red-flag symptoms | `src/triage/model.py` (`predict_df`), app **Triage** tab |
| Asymmetric safety (under-triage ~20x worse) | `src/triage/costs.py` cost matrix; model picks the minimum expected-cost level; `evaluate.py` scores with it |
| Mandatory hard-coded escalation layer, rules shown | `src/triage/rules.py`; app **Escalation rules** tab |
| Emergency sensitivity reported separately and prominently | `src/triage/evaluate.py`; app **Metrics** tab; first block of the report |
| Abstention mechanism | `abstain_threshold=0.60` in `model.py`; abstaining costs a small fixed amount |
| Handles 0 BP, 96 degree temperatures, missing vitals | `clean_vitals()` in `rules.py`; missing-value flags as model features |
| Stretch: wait-time reduction vs first-come-first-served | `src/triage/queue_sim.py`; app **Queue demo** tab |

## Cost matrix (our assumption of the stated ~20x ratio)

| True level | Cost per level of **under**-triage | Cost per level of **over**-triage |
|---|---|---|
| 1-2 (emergency) | 20 | 1 |
| 3 (urgent) | 3 | 1 |
| 4-5 (mild) | 1 | 1 |

Abstaining ("send to nurse now") costs a flat 2: much cheaper than a missed emergency, a little
more than a correct answer. The model predicts the level that minimises expected cost, so a patient
with even a small chance of being an emergency is escalated. The price is more over-triage; the
results above report the escalation rate and precision so that trade-off is visible, not hidden.

## Hard-coded escalation rules

| Vital | Level 1 | Level 2 |
|---|---|---|
| SpO2 | < 90% | < 93% |
| Systolic BP | < 90 | >= 180 |
| Pulse | < 40 or > 150 | < 50 or > 130 |
| Temperature | | >= 40 C or < 35 C |

Red-flag complaint keywords (English and Hinglish) include: unconscious / behosh, seizure / daura,
chest pain / seene mein dard, breathless / saans, heavy bleeding / khoon, stroke signs, poisoning,
snake bite, self-harm risk. Full list in `src/triage/rules.py` and in the app.

## Limitations (please read)

- **Synthetic data.** Our 30k training records come from our own template-based generator
  (`generate_data.py`) because the official dataset is released after applying. Train and test share
  templates, so scores **overstate real-world performance**. The generator also has artefacts
  (for example vague complaints appear only for emergencies). Retraining on the official data is
  one command (see below).
- **Keyword rules are not exhaustive** for Indian languages. The ML layer covers some transliterations,
  but that is unproven outside our synthetic data.
- **Probabilities are not explicitly calibrated** (plain logistic regression). Next step: isotonic
  calibration and tuning of the escalation threshold against real data.
- **Wait-time numbers are simulated** (assumptions: 2 doctors, 7 min consult, one arrival every 4 min),
  not measured from the real arrival log.
- The cost matrix is our reading of the stated 20x ratio. The official `evaluate.py` may differ.
- **Not a medical device.** Prototype only.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest -q                                                      # tests
python -m src.triage.generate_data --n 30000 --out data/synthetic_opd.csv
python -m src.triage.train --data data/synthetic_opd.csv                 # trains models/triage.joblib
python -m src.triage.generate_data --n 5000 --seed 999 --out data/holdout.csv
python -m src.triage.evaluate --data data/holdout.csv                    # cost-weighted report
python -m src.triage.escalation_report --data data/holdout.csv           # workload + threshold sweep
streamlit run app/streamlit_app.py                                       # demo UI
```

**Using the official dataset:** put it in `data/` with the same columns (`age, sex, sbp, dbp, pulse, temp,
spo2, complaint, conditions, medications, triage_level, department, is_emergency`), then run the
`train` and `evaluate` commands with `--data` pointing at it.

## Repo layout

```
src/triage/rules.py              hard-coded escalation layer + vitals cleaning
src/triage/model.py              cost-sensitive model, rules override, abstention
src/triage/costs.py              asymmetric cost matrix
src/triage/evaluate.py           cost-weighted evaluation + emergency sensitivity
src/triage/escalation_report.py  escalation rate, precision, threshold sweep
src/triage/queue_sim.py          FCFS vs triage-queue wait simulation
src/triage/generate_data.py      synthetic intake data generator
src/triage/train.py              training + quick report
app/streamlit_app.py             nurse-facing UI (4 tabs)
tests/                           pytest suite (run in CI on every PR)
```