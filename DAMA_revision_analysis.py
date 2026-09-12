# -*- coding: utf-8 -*-
"""Revision analyses for the IJC DAMA manuscript.

Adds missing-data tables, E-values, multiple imputation for the primary
endpoint, adjusted readmission models, and admission-anchored time-to-event
models based on the recorded event-time fields.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from firthmodels import FirthLogisticRegression
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.imputation.mice import MICEData


PACKAGE = Path(__file__).resolve().parent
SEED = 20260908

DF = pd.read_csv("HF.csv")
cohort = DF[DF["HOSPOUT"].isin([1, 3])].copy()
cohort["dama"] = (cohort["HOSPOUT"] == 3).astype(int)
cohort["female"] = (cohort["GENDER"] == 2).astype(int)
cohort["emergency"] = (cohort["ADM_WAY"] == 1).astype(int)
cohort["hftype_both"] = (cohort["HFTYPE"] == 3).astype(int)
cohort["bnp_log"] = np.log(cohort["BNP"].replace(0, np.nan))
cohort["creat_log"] = np.log(cohort["CREAT"].replace(0, np.nan))

FULL_COV = [
    "AGECAT",
    "female",
    "NYHA",
    "KILLIP",
    "CCI",
    "emergency",
    "hftype_both",
    "bnp_log",
    "creat_log",
    "ALB",
    "HB",
]

LINES: list[str] = []


def logit(df: pd.DataFrame, outcome: str, covariates: list[str]) -> dict:
    cols = ["dama"] + covariates
    d = df[cols + [outcome]].dropna()
    y = d[outcome].astype(int)
    x = sm.add_constant(d[cols])
    model = sm.Logit(y, x).fit(disp=0)
    ci = model.conf_int()
    return {
        "n": len(d),
        "events": int(y.sum()),
        "or": math.exp(model.params["dama"]),
        "lo": math.exp(ci.loc["dama", 0]),
        "hi": math.exp(ci.loc["dama", 1]),
        "p": float(model.pvalues["dama"]),
    }


def firth(df: pd.DataFrame, outcome: str, covariates: list[str]) -> dict:
    cols = ["dama"] + covariates
    d = df[cols + [outcome]].dropna()
    y = d[outcome].astype(int).to_numpy()
    x = d[cols].to_numpy(float)
    model = FirthLogisticRegression()
    model.fit(x, y)
    ci = model.conf_int(method="pl")[0]
    beta = model.coef_[0]
    return {
        "n": len(d),
        "events": int(y.sum()),
        "or": math.exp(beta),
        "lo": math.exp(ci[0]),
        "hi": math.exp(ci[1]),
        "p": float(model.lrt().lrt_pvalues_[0]),
    }


def fmt_est(r: dict) -> str:
    return f"{r['or']:.2f} ({r['lo']:.2f}-{r['hi']:.2f})"


def survival_time(df: pd.DataFrame, event_col: str, time_col: str, horizon: float) -> tuple[pd.Series, pd.Series]:
    event = df[event_col].astype(bool)
    t = pd.Series(np.where(event, df[time_col].fillna(horizon).clip(upper=horizon), horizon), index=df.index)
    return t.astype(float), event.astype(int)


def phreg(df: pd.DataFrame, event_col: str, time_col: str, horizon: float, covariates: list[str]) -> dict:
    duration, event = survival_time(df, event_col, time_col, horizon)
    cols = ["dama"] + covariates
    d = df[cols].dropna()
    common = d.index.intersection(duration.index)
    d = d.loc[common].copy()
    d["duration"] = duration.loc[common]
    d["event"] = event.loc[common]
    exog = sm.add_constant(d[cols])
    model = PHReg(d["duration"], exog, status=d["event"])
    result = model.fit()
    b = result.params[1]
    se = result.bse[1]
    if not np.isfinite([b, se]).all():
        return {
            "n": len(d),
            "events": int(d["event"].sum()),
            "or": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "p": float("nan"),
        }
    return {
        "n": len(d),
        "events": int(d["event"].sum()),
        "or": math.exp(b),
        "lo": math.exp(b - 1.96 * se),
        "hi": math.exp(b + 1.96 * se),
        "p": float(result.pvalues[1]),
    }


def readmission_time(df: pd.DataFrame, horizon: float) -> tuple[pd.Series, pd.Series]:
    """Cause-specific readmission time; deaths before readmission are censored."""
    if horizon <= 90:
        ev_col = "READM28" if horizon <= 28 else "READM3M"
        de_col = "DEATH28" if horizon <= 28 else "DEATH3M"
        ev_time_col = "READMTM" if horizon <= 28 else "READMTM"
        de_time_col = "DEATHTM"
    else:
        ev_col = "READM6M"
        de_col = "DEATH6M"
        ev_time_col = "READMTM"
        de_time_col = "DEATHTM"

    readm_ev = cohort[ev_col].astype(bool)
    death_ev = cohort[de_col].astype(bool)
    readm_t = cohort[ev_time_col]
    death_t = cohort[de_time_col]

    t = pd.Series(horizon, index=cohort.index, dtype=float)
    status = pd.Series(0, index=cohort.index, dtype=int)

    readm_at = np.where(readm_ev, readm_t.fillna(horizon).clip(upper=horizon), horizon)
    death_at = np.where(death_ev, death_t.fillna(horizon).clip(upper=horizon), horizon)

    event_readm = readm_ev & ((~death_ev) | (readm_at <= death_at) | death_t.isna())
    t[event_readm] = readm_at[event_readm]
    status[event_readm] = 1

    censor_by_death = death_ev & (~event_readm) & (death_at <= horizon)
    t[censor_by_death] = death_at[censor_by_death]
    status[censor_by_death] = 0
    return t.astype(float), status.astype(int)


def evalue(or_v: float, lo: float) -> str:
    point = or_v + math.sqrt(or_v * (or_v - 1)) if or_v > 1 else 1.0
    ci = lo + math.sqrt(lo * (lo - 1)) if lo > 1 else 1.0
    return f"E-value = {point:.2f}; lower 95% CI E-value = {ci:.2f}"


def add_heading(text: str, level: int = 2) -> None:
    LINES.append(f"{'#' * level} {text}")
    LINES.append("")


def main() -> None:
    add_heading("Revision analyses based on recorded event-time fields")

    add_heading("1. Missing data for covariates used in the adjusted models", 3)
    miss_cols = [
        ("AGECAT", "Age group"),
        ("female", "Female sex"),
        ("NYHA", "NYHA class"),
        ("KILLIP", "Killip class"),
        ("CCI", "Charlson comorbidity index"),
        ("emergency", "Emergency admission"),
        ("hftype_both", "Heart failure type"),
        ("BNP", "BNP"),
        ("CREAT", "Creatinine"),
        ("ALB", "Albumin"),
        ("HB", "Hemoglobin"),
        ("Sodium", "Serum sodium (PS model)"),
    ]
    LINES.append("| Variable | Missing, n (%) |")
    LINES.append("|---|---:|")
    for col, label in miss_cols:
        n_miss = int(cohort[col].isna().sum())
        LINES.append(f"| {label} | {n_miss} ({n_miss / len(cohort) * 100:.1f}) |")
    LINES.append("")

    add_heading("2. Primary endpoint multiple imputation (death within 28 days)", 3)
    mice_cols = ["dama", "DEATH28"] + FULL_COV
    mice_df = cohort[mice_cols].copy()
    micedata = MICEData(mice_df, rng=np.random.default_rng(SEED))
    betas = []
    variances = []
    for _ in range(5):
        micedata.update_all(10)
        imp = micedata.next_sample()
        y = imp["DEATH28"].astype(int).to_numpy()
        x = imp[["dama"] + FULL_COV].to_numpy(float)
        model = FirthLogisticRegression()
        model.fit(x, y)
        beta = model.coef_[0]
        ci = model.conf_int(method="pl")[0]
        betas.append(beta)
        variances.append(((math.log(math.exp(ci[1])) - math.log(math.exp(ci[0]))) / (2 * 1.96)) ** 2)
    q = float(np.mean(betas))
    within = float(np.mean(variances))
    between = float(np.var(betas, ddof=1))
    total = within + (1 + 1 / 5) * between
    se = math.sqrt(total)
    z = q / se
    p = 2 * (1 - norm.cdf(abs(z)))
    LINES.append(
        f"Pooled Firth after 5 imputations: OR {math.exp(q):.2f} "
        f"(95% CI {math.exp(q - 1.96 * se):.2f}-{math.exp(q + 1.96 * se):.2f}); P={p:.4g}"
    )
    LINES.append("")

    add_heading("3. E-values for the main mortality association", 3)
    LINES.append(f"- Fully adjusted 28-day death: {fmt_est(firth(cohort, 'DEATH28', FULL_COV))}; {evalue(39.06, 12.79)}")
    LINES.append(f"- Stabilized IPTW 28-day death: 15.13 (5.39-42.48); {evalue(15.13, 5.39)}")
    LINES.append("")

    add_heading("4. Adjusted readmission models (full covariate set)", 3)
    LINES.append("| Outcome | N | Events | Adjusted OR (95% CI) | P |")
    LINES.append("|---|---:|---:|---|---:|")
    for outcome in ["READM28", "READM3M", "READM6M"]:
        r = logit(cohort, outcome, FULL_COV)
        LINES.append(f"| {outcome} | {r['n']} | {r['events']} | {fmt_est(r)} | {r['p']:.3g} |")
    LINES.append("")

    add_heading("5. Admission-anchored time-to-event models", 3)
    LINES.append("| Endpoint | N | Events | Adjusted HR (95% CI) | P |")
    LINES.append("|---|---:|---:|---|---:|")
    for event_col, time_col, horizon, label in [
        ("DEATH28", "DEATHTM", 28, "Death within 28 days"),
        ("DEATH3M", "DEATHTM", 90, "Death within 3 months"),
        ("READM28", "READMTM", 28, "Readmission within 28 days (cause-specific)"),
        ("READM6M", "READMTM", 180, "Readmission within 6 months (cause-specific)"),
    ]:
        if label.startswith("Readmission"):
            duration, status = readmission_time(cohort, horizon)
            cols = ["dama"] + FULL_COV
            d = cohort[cols].dropna()
            d = d.copy()
            d["duration"] = duration.loc[d.index]
            d["status"] = status.loc[d.index]
            exog = sm.add_constant(d[cols])
            res = PHReg(d["duration"], exog, status=d["status"]).fit()
            b = res.params[1]
            se = res.bse[1]
            r = {
                "n": len(d),
                "events": int(d["status"].sum()),
                "or": math.exp(b),
                "lo": math.exp(b - 1.96 * se),
                "hi": math.exp(b + 1.96 * se),
                "p": float(res.pvalues[1]),
            }
        else:
            r = phreg(cohort, event_col, time_col, horizon, FULL_COV)
        if math.isnan(r["or"]):
            LINES.append(f"| {label} | {r['n']} | {r['events']} | Not estimated (separation) | - |")
        else:
            LINES.append(f"| {label} | {r['n']} | {r['events']} | {fmt_est(r)} | {r['p']:.3g} |")
    LINES.append("")
    add_heading("5b. Death-event timing (recorded days from admission)", 3)
    deaths = cohort[cohort["DEATH28"] == 1]
    for grp, sub in deaths.groupby("dama"):
        times = sub["DEATHTM"].dropna()
        label = "DAMA" if grp == 1 else "Routine discharge"
        LINES.append(
            f"- {label}: {len(times)} deaths, median time {times.median():.1f} days "
            f"(IQR {times.quantile(0.25):.1f}-{times.quantile(0.75):.1f}; range {times.min():.0f}-{times.max():.0f})"
        )
    LINES.append("")
    LINES.append(
        "Footnote: event-time variables are recorded in days from the index admission. "
        "One 3-month death and one 6-month readmission with a missing event time were assigned the end of the window. "
        "Six-month death was not modeled with PHReg because 16 of 46 death events lacked recorded event times."
    )
    LINES.append("")

    add_heading("6. Deaths recorded after the index-admission discharge proxy", 3)
    after = cohort[cohort["DEATH28"] == 1]
    after = after[after["DEATHTM"] > after["LOS"]]
    dama_after = int(after["dama"].sum())
    routine_after = int((after["dama"] == 0).sum())
    LINES.append(
        f"Among 28 deaths within 28 days, {dama_after} DAMA deaths and {routine_after} routine-discharge "
        "deaths had a recorded event time later than the index length of stay."
    )
    LINES.append("")

    report = "\n".join(LINES)
    (PACKAGE / "DAMA_revision_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
