"""ML layer + asymmetric decision + abstention, wrapped by the rules layer."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .rules import apply_rules, clean_vitals

LEVELS = np.arange(1, 6)
W_EMERG = 20.0        # under-triaging a true emergency (level<=2) costs 20x an over-triage step
ABSTAIN_COST = 1.0    # cost of sending a patient to the nurse for manual assessment
ABSTAIN_TAU = 1.0     # abstain when the best achievable expected cost exceeds this
NUM = ["age", "sbp", "dbp", "pulse", "temp_c", "spo2"]
ALIASES = {"bp_sys": "sbp", "bp_dia": "dbp", "temp": "temp_c", "chief_complaint": "complaint",
           "triage": "triage_level", "level": "triage_level", "dept": "department"}


def cost_matrix():
    """C[true-1, pred-1]. Under-triage = (pred-true)*weight, over-triage = (true-pred)*1."""
    C = np.zeros((5, 5))
    for t in LEVELS:
        for p in LEVELS:
            C[t - 1, p - 1] = (p - t) * (W_EMERG if t <= 2 else 1.0) if p > t else (t - p)
    return C


COST = cost_matrix()


def prep(df):
    """Rename aliases, clean vitals (0 BP, 96C...), build model input."""
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns and v not in df.columns}).copy()
    for c in ["sbp", "dbp", "pulse", "temp_c", "spo2"]:
        df["raw_" + c] = df[c]                      # rules layer re-cleans these, so bad vitals get flagged
    cleaned = [clean_vitals(r.sbp, r.dbp, r.pulse, r.temp_c, r.spo2) for r in df.itertuples()]
    for c, f in zip(["sbp", "dbp", "pulse", "temp_c", "spo2"], ["sbp", "dbp", "pulse", "temp_c", "spo2"]):
        df[c] = [getattr(v, f) for v, _ in cleaned]
    df["text"] = (df["complaint"].fillna("").astype(str).str.lower() + " | " + df.get("known_conditions", "").fillna("").astype(str).str.lower())
    df["sex"] = df["sex"].fillna("?").astype(str)
    return df


def make_pipe():
    pre = ColumnTransformer([
        ("txt", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True), "text"),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), NUM),
        ("sex", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=4, max_iter=2000))])


class Triager:
    def fit(self, df):
        d = prep(df)
        self.level_m = make_pipe().fit(d, d["triage_level"])
        self.dept_m = make_pipe().fit(d, d["department"])
        return self

    def predict(self, df, use_cost=True, use_rules=True, tau=ABSTAIN_TAU):
        """tau=None disables abstention. Returns one row per patient."""
        d = prep(df)
        P = np.zeros((len(d), 5))
        P[:, self.level_m.classes_.astype(int) - 1] = self.level_m.predict_proba(d)
        exp_cost = P @ COST
        lvl = (exp_cost.argmin(1) if use_cost else P.argmax(1)) + 1
        best = exp_cost.min(1)
        dept = self.dept_m.predict(d).astype(object)
        abstain = np.zeros(len(d), bool) if tau is None else best > tau
        flags, reasons, recheck, rule_lvl = [], [], [], np.full(len(d), np.nan)
        for i, r in enumerate(d.itertuples()):
            res = apply_rules(r.complaint, r.raw_sbp, r.raw_dbp, r.raw_pulse, r.raw_temp_c, r.raw_spo2) if use_rules else None
            if res is None:
                flags.append([]); reasons.append([]); recheck.append([]); continue
            flags.append(res.red_flags); reasons.append(res.reasons); recheck.append(res.needs_recheck)
            if res.level is not None:                         # rules override the model
                rule_lvl[i] = res.level
                lvl[i] = min(lvl[i], res.level); dept[i] = res.department; abstain[i] = False
        final = np.where(abstain, 2, lvl)                      # abstain => nurse now (treated as urgent)
        dept = np.where(abstain, "Nurse review", dept)
        return pd.DataFrame({"level": final, "model_level": lvl, "department": dept, "abstain": abstain,
                             "p_emergency": P[:, :2].sum(1), "rule_level": rule_lvl, "red_flags": flags,
                             "reasons": reasons, "needs_recheck": recheck}, index=df.index)

    def triage_one(self, complaint, age=35, sex="M", sbp=None, dbp=None, pulse=None, temp=None, spo2=None, known_conditions="", **kw):
        row = pd.DataFrame([dict(complaint=complaint, age=age, sex=sex, sbp=sbp, dbp=dbp, pulse=pulse, temp_c=temp,
                                 spo2=spo2, known_conditions=known_conditions)])
        return self.predict(row, **kw).iloc[0].to_dict()

    def save(self, path="models/triage.joblib"):
        joblib.dump(self, path)

    @staticmethod
    def load(path="models/triage.joblib"):
        return joblib.load(path)
