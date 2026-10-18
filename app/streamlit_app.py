import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.triage.generate_data import generate
from src.triage.model import TriageModel
from src.triage.queue_sim import compare
from src.triage.rules import KEYWORD_RULES, apply_rules

st.set_page_config(page_title="OPD Triage Assistant", page_icon="🏥", layout="wide")

MODEL_PATH = "models/triage.joblib"
LEVEL_NAMES = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less urgent", 5: "Non-urgent"}


@st.cache_resource
def get_model():
    if os.path.exists(MODEL_PATH):
        return TriageModel.load(MODEL_PATH)
    return TriageModel().fit(generate(8000, seed=1))


model = get_model()
st.title("🏥 OPD Triage Assistant")
st.caption("Decision support only. A clinician makes the final decision. Prototype trained on synthetic data.")

tab1, tab2, tab3, tab4 = st.tabs(["Triage", "Escalation rules", "Metrics", "Queue demo"])

# ---------------- Triage ----------------
with tab1:
    with st.form("intake"):
        c1, c2 = st.columns([2, 1])
        with c1:
            complaint = st.text_area("Chief complaint (any language, e.g. 'seene mein dard')", height=110)
            conditions = st.multiselect("Known conditions",
                                        ["diabetes", "hypertension", "asthma", "CAD", "CKD", "thyroid"])
        with c2:
            age = st.number_input("Age", 0, 120, 40)
            sex = st.selectbox("Sex", ["M", "F"])
        v1, v2, v3, v4, v5 = st.columns(5)
        sbp = v1.number_input("Systolic BP", value=None, placeholder="mmHg")
        dbp = v2.number_input("Diastolic BP", value=None, placeholder="mmHg")
        pulse = v3.number_input("Pulse", value=None, placeholder="bpm")
        temp = v4.number_input("Temp", value=None, placeholder="°C or °F")
        spo2 = v5.number_input("SpO2 %", value=None, placeholder="%")
        go = st.form_submit_button("Triage patient", type="primary")

    if go:
        cond = ";".join(conditions) if conditions else "none"
        r = model.predict_one(complaint, age=age, sex=sex, sbp=sbp, dbp=dbp,
                              pulse=pulse, temp=temp, spo2=spo2, conditions=cond)
        rr = apply_rules(complaint, sbp, dbp, pulse, temp, spo2)
        lvl = int(r["final_level"])

        if r["abstain"]:
            st.warning("### CANNOT ASSESS: send to nurse now")
        elif lvl <= 2:
            st.error(f"### Level {lvl}: {LEVEL_NAMES[lvl]}. ESCALATE, doctor now")
        elif lvl == 3:
            st.warning(f"### Level {lvl}: {LEVEL_NAMES[lvl]}")
        else:
            st.success(f"### Level {lvl}: {LEVEL_NAMES[lvl]}")

        a, b, c = st.columns(3)
        a.metric("Department", r["final_dept"])
        b.metric("P(emergency)", f"{r['p_emergency']:.0%}")
        c.metric("Model confidence", f"{r['confidence']:.0%}")

        if rr.red_flags:
            st.markdown("**Red flags detected:** " + ", ".join(rr.red_flags))
        if rr.reasons:
            st.markdown("**Rule layer fired (overrides model):**")
            for x in rr.reasons:
                st.markdown(f"- {x}")
        if rr.needs_recheck:
            st.info("Invalid or mis-entered vitals ignored, please re-measure: " + ", ".join(rr.needs_recheck))
        if sbp is None and pulse is None and spo2 is None:
            st.info("No vitals entered. Triage is based on the complaint only.")

# ---------------- Rules ----------------
with tab2:
    st.subheader("Hard-coded escalation layer (runs before and overrides the model)")
    st.markdown("""
| Vital | Level 1 (immediate) | Level 2 (emergent) |
|---|---|---|
| SpO2 | < 90% | < 93% |
| Systolic BP | < 90 | >= 180 |
| Pulse | < 40 or > 150 | < 50 or > 130 |
| Temperature | | >= 40 C or < 35 C |

**Invalid vitals are ignored, not trusted:** BP of 0, SpO2 of 0, pulse outside 20-250, temperature
outside 25-45 C (values 90-110 are treated as Fahrenheit and converted).
The final level is always the **more urgent** of the rule level and the model level.
""")
    st.markdown("**Red-flag complaint keywords (English / Hinglish):**")
    st.dataframe(pd.DataFrame(KEYWORD_RULES, columns=["keyword", "min level", "red flag", "department"]),
                 use_container_width=True, hide_index=True)

# ---------------- Metrics ----------------
# ---------------- Metrics ----------------
with tab3:
    def show(path, title):
        if not os.path.exists(path):
            st.info(f"{path} not found. Run `python -m src.triage.evaluate`.")
            return
        m = json.load(open(path))
        st.markdown(f"#### {title}")
        a, b, c, d = st.columns(4)
        a.metric("Emergency sensitivity (strict)", f"{m['emergency_recall_strict']:.1%}")
        b.metric("Sensitivity (abstain = safe)", f"{m['emergency_recall_with_abstain']:.1%}")
        c.metric("Emergencies missed", f"{m['emergencies_missed']} / {m['n_emergencies']}")
        d.metric("Mean cost (naive L4 baseline)", f"{m['mean_cost']:.2f} ({m['baseline_cost_all_level4']:.2f})")
        with st.expander("All metrics"):
            st.json(m)

    show("reports/metrics.json", "Standard synthetic holdout (optimistic)")
    show("reports/metrics_hard.json", "Stress test: noisy, overlapping inputs")
    st.caption("The stress test shows what happens when complaints and vitals stop giving the game away: "
               "escalation precision drops and some emergencies become hard to detect from the intake form alone.")

# ---------------- Queue demo ----------------
with tab4:
    st.subheader("Waiting time: first-come-first-served vs triage queue")
    c1, c2, c3, c4 = st.columns(4)
    n = c1.slider("Patients", 200, 3000, 800, 100)
    docs = c2.slider("Doctors", 1, 4, 2)
    gap = c3.slider("Minutes between arrivals", 1.0, 8.0, 4.0, 0.5)
    svc = c4.slider("Consult minutes", 3.0, 15.0, 7.0, 0.5)
    if st.button("Run simulation"):
        sample = generate(n, seed=123)
        p = model.predict_df(sample)
        lv = np.where(p.abstain, 2, p.final_level)
        res = compare(sample.is_emergency.values == 1, lv, arrival_gap=gap, doctors=docs, service=svc)
        st.metric("Emergency avg wait (minutes)",
                  f"{res['emergency_wait_priority']:.1f}",
                  f"{res['emergency_wait_priority'] - res['emergency_wait_fcfs']:.1f} vs FCFS",
                  delta_color="inverse")
        st.bar_chart(pd.DataFrame({
            "FCFS": [res["emergency_wait_fcfs"], res["all_wait_fcfs"]],
            "Triage queue": [res["emergency_wait_priority"], res["all_wait_priority"]],
        }, index=["Emergencies", "All patients"]))
        st.caption("Simulated with synthetic arrivals. Replace with the provided arrival log when available.")


