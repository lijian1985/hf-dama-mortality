import math

import numpy as np
import pandas as pd
import statsmodels.api as sm


DF = pd.read_csv("HF.csv")


def smd_continuous(x: pd.Series, y: pd.Series) -> float:
    a = x.dropna()
    b = y.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sa = a.std(ddof=1)
    sb = b.std(ddof=1)
    pooled = math.sqrt((sa * sa + sb * sb) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def smd_binary(x: pd.Series, y: pd.Series) -> float:
    a = x.dropna()
    b = y.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p1 = float(a.mean())
    p2 = float(b.mean())
    den = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    if den == 0:
        return 0.0
    return float((p1 - p2) / den)


def clean_log(x: pd.Series) -> pd.Series:
    y = x.replace(0, np.nan)
    return np.log(y)


def describe_numeric(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    return f"{s.median():.2f} ({s.quantile(0.25):.2f}-{s.quantile(0.75):.2f})"


def run_logit(df: pd.DataFrame, outcome: str, covariates: list[str], label: str) -> dict:
    cols = ["dama"] + covariates
    d = df[cols + [outcome]].dropna()
    y = d[outcome].astype(int)
    x = sm.add_constant(d[cols])
    try:
        model = sm.Logit(y, x).fit(disp=0)
    except Exception as exc:  # singular or separation
        return {"label": label, "n": len(d), "events": int(y.sum()), "error": str(exc)}
    params = model.params
    ci = model.conf_int()
    orv = math.exp(params["dama"])
    lo = math.exp(ci.loc["dama", 0])
    hi = math.exp(ci.loc["dama", 1])
    return {
        "label": label,
        "n": len(d),
        "events": int(y.sum()),
        "OR": round(orv, 3),
        "CI_lo": round(lo, 3),
        "CI_hi": round(hi, 3),
        "p": model.pvalues["dama"],
    }


cohort = DF[DF["HOSPOUT"].isin([1, 3])].copy()
cohort["dama"] = (cohort["HOSPOUT"] == 3).astype(int)

# Recalculate composite outcomes from raw components.
cohort["comp28"] = ((cohort["DEATH28"] == 1) | (cohort["READM28"] == 1)).astype(int)
cohort["comp3"] = ((cohort["DEATH3M"] == 1) | (cohort["READM3M"] == 1)).astype(int)
cohort["comp6"] = ((cohort["DEATH6M"] == 1) | (cohort["READM6M"] == 1)).astype(int)

cohort["bnp_log"] = clean_log(cohort["BNP"])
cohort["creat_log"] = clean_log(cohort["CREAT"])

cohort["gender_female"] = (cohort["GENDER"] == 2).astype(int)
cohort["occu_farmer"] = (cohort["OCCU"] == 2).astype(int)
cohort["hftype_both"] = (cohort["HFTYPE"] == 3).astype(int)
cohort["admission_emergency"] = (cohort["ADM_WAY"] == 1).astype(int)

a = cohort[cohort["dama"] == 1]
b = cohort[cohort["dama"] == 0]

lines = []
lines.append("# 自动出院与心衰患者短期预后：初步分析")
lines.append("")
lines.append(f"总样本：{len(cohort)}；自动出院 {len(a)}；正常出院 {len(b)}")
lines.append("")

lines.append("## 结局事件")
lines.append("")
lines.append("| 结局 | 正常出院 | 自动出院 |")
lines.append("|---|---:|---:|")
for col in [
    "DEATH28",
    "DEATH3M",
    "DEATH6M",
    "READM28",
    "READM3M",
    "READM6M",
    "comp28",
    "comp3",
    "comp6",
]:
    lines.append(f"| {col} | {int(b[col].sum())} ({b[col].mean()*100:.1f}%) | {int(a[col].sum())} ({a[col].mean()*100:.1f}%) |")
lines.append("")

lines.append("## 基线特征")
lines.append("")
lines.append("| 变量 | 正常出院 | 自动出院 | SMD |")
lines.append("|---|---:|---:|---:|")

numeric_cols = [
    "AGECAT",
    "BMI",
    "SBP",
    "DBP",
    "PULSE",
    "RESP",
    "NYHA",
    "KILLIP",
    "CCI",
    "BNP",
    "CREAT",
    "EGFR",
    "ALB",
    "HB",
    "Sodium",
    "LOS",
    "NDRUGS",
]
for col in numeric_cols:
    lines.append(
        f"| {col} | {describe_numeric(b, col)} | {describe_numeric(a, col)} | {smd_continuous(a[col], b[col]):.3f} |"
    )

binary_cols = [
    "gender_female",
    "admission_emergency",
    "occu_farmer",
    "hftype_both",
    "MI",
    "PVD",
    "CVD",
    "DEMENTIA",
    "COPD",
    "DIABETES",
    "CKD",
    "TUMOR",
    "LIVER_DZ",
    "ARF",
    "LOOP_DI",
    "MRA",
    "ACEIARB",
    "BETA_B",
    "DIG",
    "APLT",
    "ANTICO",
    "STATIN",
    "INOTRP",
    "NITRO",
    "TCM",
]
for col in binary_cols:
    pb = float(b[col].mean())
    pa = float(a[col].mean())
    lines.append(f"| {col} | {pb*100:.1f}% | {pa*100:.1f}% | {smd_binary(a[col], b[col]):.3f} |")
lines.append("")

lines.append("## 主结局 logistic 回归（DAMA 效应）")
lines.append("")
lines.append("| 模型 | N | 事件 | OR | 95% CI | P |")
lines.append("|---|---:|---:|---:|---:|---:|")

models = {
    "comp3_未调整": [],
    "comp3_人口+严重度": ["AGECAT", "GENDER", "NYHA", "KILLIP", "CCI", "ADM_WAY"],
    "comp3_再加BNP/肾功能": [
        "AGECAT",
        "GENDER",
        "NYHA",
        "KILLIP",
        "CCI",
        "ADM_WAY",
        "bnp_log",
        "creat_log",
        "ALB",
        "HB",
    ],
    "comp6_未调整": [],
    "comp6_人口+严重度": ["AGECAT", "GENDER", "NYHA", "KILLIP", "CCI", "ADM_WAY"],
    "DEATH28_未调整": [],
    "DEATH28_年龄性别": ["AGECAT", "GENDER"],
    "DEATH6M_未调整": [],
    "DEATH6M_年龄性别NYHA": ["AGECAT", "GENDER", "NYHA"],
}

for outcome, label in [
    ("comp3", "comp3_未调整"),
    ("comp3", "comp3_人口+严重度"),
    ("comp3", "comp3_再加BNP/肾功能"),
    ("comp6", "comp6_未调整"),
    ("comp6", "comp6_人口+严重度"),
    ("DEATH28", "DEATH28_未调整"),
    ("DEATH28", "DEATH28_年龄性别"),
    ("DEATH6M", "DEATH6M_未调整"),
    ("DEATH6M", "DEATH6M_年龄性别NYHA"),
]:
    result = run_logit(cohort, outcome, models[label], label)
    if "error" in result:
        lines.append(f"| {label} | 失败：{result['error']} |")
    else:
        lines.append(
            f"| {label} | {result['n']} | {result['events']} | {result['OR']} | "
            f"{result['CI_lo']}-{result['CI_hi']} | {result['p']:.4f} |"
        )
lines.append("")
lines.append("> 说明：以上为初步结果；AGECAT 为年龄组；GENDER 以男性为参考；BNP/CREAT 经自然对数转换。")

report = "\n".join(lines)
with open("DAMA_preliminary_results.md", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("\nSaved: DAMA_preliminary_results.md")
