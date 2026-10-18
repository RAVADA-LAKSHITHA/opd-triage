# OPD Triage Assistant (H1)

Intake form + free-text complaint → triage level (1–5), department, red flags — safely.

```
complaint + vitals ─► clean vitals (0 BP, 96°C…) ─► ML (TF-IDF char n-grams + vitals, LogReg)
                                   │                     │ class probabilities
                                   │                     ▼
                                   │      asymmetric Bayes decision (under-triage ×20)
                                   │                     │ + abstain if expected cost > τ
                                   ▼                     ▼
                      HARD-CODED RULES  ───override───►  final level / dept / red flags
```

| Requirement | Where |
|---|---|
| Level, department, red flags | `Triager.predict` / app |
| 20× asymmetric safety | `model.cost_matrix`, used for the decision *and* scoring |
| Hard-coded escalation rules | `src/triage/rules.py` (+ `tests/test_rules.py`) |
| Emergency recall, reported separately | `evaluate.py` → `results/metrics.json` |
| Abstention ("cannot assess – send to nurse now") | `ABSTAIN_TAU` in `model.py` |
| Stretch: wait-time vs FCFS | `evaluate.wait_study` (priority-queue sim, 2 doctors, 800/day) |

## Run
```bash
make install && make run      # generates data, trains, evaluates, launches UI
make test
```
**Real data:** drop the organisers' CSV at `data/opd_synthetic.csv` (or `--data path.csv`). Column aliases
(`bp_sys`→`sbp`, `temp`→`temp_c`, …) are handled in `model.prep`; needs `triage_level`, `department`, `is_emergency`,
`arrival_day`, `arrival_min` for the full evaluation. Without it, `src/triage/data.py` generates a same-schema synthetic set.

## CI/CD
`.github/workflows/ci.yml`: pytest → train+evaluate (fails if emergency recall < 0.95) → on `main`, build Docker image and push to GHCR.

## Caveats
- Synthetic data is easy, so metrics are optimistic; re-check on the real set and re-tune `ABSTAIN_TAU`.
- Keyword rules are substring/word-start matches and don't handle negation ("no chest pain").
- Wait-time sim uses an assumed 1.4 min mean consult time; use the provided arrival log for the real figure.
