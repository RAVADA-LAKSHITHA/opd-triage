# 3-minute demo script

**0:00 - Problem (30s).** 800 walk-ins, 2 doctors, first-come-first-served. Chest pain waits behind mild fever.
Our tool triages in under a second and never trusts the model alone for critical vitals.

**0:30 - Live triage, Triage tab (60s).** Enter these one at a time:
1. Complaint `seene mein dard`, no vitals -> Level 2, Cardiology, red flag, "rule layer fired".
2. Complaint `checkup`, SpO2 field = `82` -> Level 1. Say: the complaint sounds harmless, the hard-coded rule overrides.
3. Complaint `mild fever and cold`, Temp = `98.6`, Systolic BP = `0` -> Level 4. Say: BP of 0 is ignored and flagged to re-measure; 98.6 recognised as Fahrenheit.
4. Complaint `feeling weak`, no vitals -> "CANNOT ASSESS: send to nurse now". Say: abstaining is cheaper than guessing.

**1:30 - Escalation rules tab (20s).** The rules are shown, as the problem requires. Final level is always the more urgent of rules and model.

**1:50 - Metrics tab (30s).** Lead with emergency sensitivity, then missed emergencies, then cost vs the "everyone is level 4" baseline.

**2:20 - Queue demo (25s).** Run the simulation: average emergency wait, first-come-first-served vs triage queue.

**2:45 - Honest close (15s).** Trained on synthetic data, scores are optimistic, next step is the official dataset and calibration.