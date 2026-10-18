import os
import streamlit as st
from src.triage.model import Triager

st.set_page_config(page_title="OPD Triage", page_icon="🏥", layout="centered")


@st.cache_resource
def get_model():
    if not os.path.exists("models/triage.joblib"):
        from src.triage.evaluate import main
        import sys; sys.argv = ["evaluate"]; main()
    return Triager.load()


COL = {1: "#b91c1c", 2: "#ea580c", 3: "#ca8a04", 4: "#2563eb", 5: "#16a34a"}
NAME = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less urgent", 5: "Non-urgent"}
st.title("🏥 OPD Triage Assistant")
st.caption("Decision support only. A nurse makes the final call.")
with st.form("f"):
    complaint = st.text_input("Chief complaint", placeholder="e.g. seene mein dard / chest pain since morning")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 0, 120, 35); sex = c2.selectbox("Sex", ["M", "F"]); kc = c3.text_input("Known conditions")
    v = st.columns(5)
    sbp = v[0].number_input("SBP", 0, 300, 120); dbp = v[1].number_input("DBP", 0, 200, 80)
    pulse = v[2].number_input("Pulse", 0, 300, 80); temp = v[3].number_input("Temp", 0.0, 120.0, 37.0, step=0.1)
    spo2 = v[4].number_input("SpO2", 0, 100, 98)
    go = st.form_submit_button("Triage", use_container_width=True)
if go:
    r = get_model().triage_one(complaint, age, sex, sbp or None, dbp or None, pulse or None, temp or None, spo2 or None, kc)
    if r["abstain"]:
        st.warning("### ⚠️ Cannot assess – send to nurse now")
    else:
        L = int(r["level"])
        st.markdown(f"<div style='background:{COL[L]};color:#fff;padding:18px;border-radius:10px;font-size:26px'>"
                    f"Level {L} · {NAME[L]} → <b>{r['department']}</b></div>", unsafe_allow_html=True)
    if r["red_flags"]: st.error("🚩 Red flags: " + ", ".join(r["red_flags"]))
    if r["reasons"]: st.write("**Rule triggers:** " + "; ".join(r["reasons"]))
    if r["needs_recheck"]: st.info("Re-check vitals: " + ", ".join(r["needs_recheck"]))
    st.caption(f"P(emergency) = {r['p_emergency']:.0%}")
