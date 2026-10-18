"""Load + clean intake data, and report how well the hard rules alone perform."""
import sys

import numpy as np
import pandas as pd

from .rules import apply_rules, clean_vitals


def load_raw(path):
    df = pd.read_csv(path)
    df["complaint"] = df["complaint"].fillna("")
    return df


def add_clean_vitals(df):
    """Adds sbp_c, dbp_c, pulse_c, temp_c, spo2_c (NaN if invalid) and *_missing flags."""
    cleaned = []
    for r in df[["sbp", "dbp", "pulse", "temp", "spo2"]].itertuples(index=False):
        v, _ = clean_vitals(r.sbp, r.dbp, r.pulse, r.temp, r.spo2)
        cleaned.append([v.sbp, v.dbp, v.pulse, v.temp_c, v.spo2])
    cols = ["sbp_c", "dbp_c", "pulse_c", "temp_c", "spo2_c"]
    out = df.copy()
    out[cols] = pd.DataFrame(cleaned, columns=cols, index=df.index).astype(float)
    for c in cols:
        out[c.replace("_c", "") + "_missing"] = out[c].isna().astype(int)
    return out


def add_rule_outputs(df):
    res = [apply_rules(r.complaint, r.sbp, r.dbp, r.pulse, r.temp, r.spo2)
           for r in df.itertuples(index=False)]
    out = df.copy()
    out["rule_level"] = [x.level if x.level is not None else np.nan for x in res]
    out["rule_dept"] = [x.department for x in res]
    out["rule_flags"] = ["; ".join(x.red_flags) for x in res]
    return out


def rules_report(df):
    d = add_rule_outputs(df)
    fired = d["rule_level"].le(2)
    emerg = d["is_emergency"] == 1
    print(f"rows: {len(d)}  true emergencies: {emerg.sum()}")
    print(f"rules-only emergency recall : {(fired & emerg).sum() / emerg.sum():.1%}")
    print(f"rules false-escalation rate : {(fired & ~emerg).sum() / (~emerg).sum():.2%}")
    print("-> emergencies the rules MISS must be caught by the ML model.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/synthetic_opd.csv"
    rules_report(load_raw(path))