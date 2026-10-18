"""Cost-sensitive triage model + rules override + abstention."""
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .costs import COST
from .data import add_clean_vitals, add_rule_outputs

NUM_COLS = ["age", "sbp_c", "dbp_c", "pulse_c", "temp_c", "spo2_c"]
MISS_COLS = ["sbp_missing", "dbp_missing", "pulse_missing", "temp_missing", "spo2_missing"]
CONDITIONS = ["diabetes", "hypertension", "asthma", "CAD", "CKD", "thyroid"]


class TriageModel:
    def __init__(self, abstain_threshold: float = 0.60):
        self.abstain_threshold = abstain_threshold
        self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                     max_features=5000, sublinear_tf=True, lowercase=True)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.level_clf = LogisticRegression(max_iter=2000, C=5.0)
        self.dept_clf = LogisticRegression(max_iter=2000, C=5.0)

    # ---------- features ----------
    def _build(self, df, fit=False):
        d = add_clean_vitals(df.reset_index(drop=True))
        d["complaint"] = d["complaint"].fillna("").astype(str)
        conds = d["conditions"].fillna("none").astype(str)

        extra = pd.DataFrame({
            "spo2_lt93": (d.spo2_c < 93).astype(int),
            "sbp_lt90": (d.sbp_c < 90).astype(int),
            "pulse_gt130": (d.pulse_c > 130).astype(int),
            "temp_ge39": (d.temp_c >= 39).astype(int),
            "sex_m": (d.sex == "M").astype(int),
        })
        for c in CONDITIONS:
            extra["cond_" + c] = conds.str.contains(c).astype(int)
        extra = pd.concat([extra, d[MISS_COLS].reset_index(drop=True)], axis=1)

        if fit:
            num = self.imputer.fit_transform(d[NUM_COLS])
            text = self.tfidf.fit_transform(d["complaint"])
            dense = self.scaler.fit_transform(np.hstack([num, extra.values]))
        else:
            num = self.imputer.transform(d[NUM_COLS])
            text = self.tfidf.transform(d["complaint"])
            dense = self.scaler.transform(np.hstack([num, extra.values]))
        return hstack([text, csr_matrix(dense)]).tocsr()

    # ---------- train ----------
    def fit(self, df):
        X = self._build(df, fit=True)
        self.level_clf.fit(X, df["triage_level"].values)
        self.dept_clf.fit(X, df["department"].values)
        return self

    def _level_proba(self, X):
        P = np.zeros((X.shape[0], 5))
        raw = self.level_clf.predict_proba(X)
        for i, c in enumerate(self.level_clf.classes_):
            P[:, int(c) - 1] = raw[:, i]
        return P

    # ---------- predict ----------
    def predict_df(self, df):
        df = df.reset_index(drop=True)
        X = self._build(df)
        P = self._level_proba(X)

        exp_cost = P @ COST                     # expected cost of predicting each level
        model_level = exp_cost.argmin(axis=1) + 1
        conf = P.max(axis=1)
        model_dept = self.dept_clf.predict(X)

        rules = add_rule_outputs(df)[["rule_level", "rule_dept", "rule_flags"]]
        rule_fired = rules["rule_level"].notna().values

        out = pd.DataFrame({
            "model_level": model_level,
            "model_dept": model_dept,
            "confidence": conf,
            "p_emergency": P[:, :2].sum(axis=1),
        })
        out = pd.concat([out, rules], axis=1)
        out["final_level"] = np.fmin(out["model_level"], out["rule_level"]).astype(int)
        out["abstain"] = (conf < self.abstain_threshold) & ~rule_fired
        out["final_dept"] = np.where(rule_fired, out["rule_dept"], out["model_dept"])
        out["action"] = np.where(
            out["abstain"], "CANNOT ASSESS - send to nurse now",
            np.where(out["final_level"] <= 2, "ESCALATE - doctor now", "Queue by level"))
        return out

    def predict_one(self, complaint, age=40, sex="M", sbp=None, dbp=None,
                    pulse=None, temp=None, spo2=None, conditions="none"):
        row = pd.DataFrame([dict(complaint=complaint, age=age, sex=sex, sbp=sbp, dbp=dbp,
                                 pulse=pulse, temp=temp, spo2=spo2, conditions=conditions)])
        return self.predict_df(row).iloc[0].to_dict()

    # ---------- persistence ----------
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)