import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact
from firthmodels import FirthLogisticRegression
from statsmodels.imputation.mice import MICEData


SEED = 20260908
DF = pd.read_csv("HF.csv")


def log_transform(x: pd.Series) -> pd.Series:
    return np.log(x.replace(0, np.nan))


def fit_logit(df: pd.DataFrame, outcome: str, covariates: list[str]) -> dict:
    cols = ["dama"] + covariates
    d = df[cols + [outcome]].dropna()
    y = d[outcome].astype(int)
    x = sm.add_constant(d[cols])
    model = sm.Logit(y, x).fit(disp=0)
    ci = model.conf_int()
    orv = math.exp(model.params["dama"])
    lo = math.exp(ci.loc["dama", 0])
    hi = math.exp(ci.loc["dama", 1])
    return {
        "n": len(d),
        "events": int(y.sum()),
        "or": orv,
        "lo": lo,
        "hi": hi,
        "p": float(model.pvalues["dama"]),
    }


def pooled_mice(df: pd.DataFrame, outcome: str, covariates: list[str], n_imp: int = 5) -> dict:
    cols = ["dama", outcome] + covariates
    d = df[cols].copy()
    micedata = MICEData(d, rng=np.random.default_rng(SEED))
    fits = []
    for _ in range(n_imp):
        micedata.update_all(10)
        imp = micedata.next_sample()
        y = imp[outcome].astype(int)
        x = sm.add_constant(imp[["dama"] + covariates])
        model = sm.Logit(y, x).fit(disp=0)
        fits.append({"beta": model.params["dama"], "var": model.bse["dama"] ** 2})

    q = np.mean([f["beta"] for f in fits])
    within = np.mean([f["var"] for f in fits])
    between = np.var([f["beta"] for f in fits], ddof=1)
    total_var = within + (1 + 1 / n_imp) * between
    se = math.sqrt(total_var)
    z = q / se
    p = 2 * (1 - __import__("scipy").stats.norm.cdf(abs(z)))
    return {
        "n": len(d),
        "or": math.exp(q),
        "lo": math.exp(q - 1.96 * se),
        "hi": math.exp(q + 1.96 * se),
        "p": p,
    }


def firth_logit(df: pd.DataFrame, outcome: str, covariates: list[str]) -> dict:
    cols = ["dama"] + covariates
    d = df[cols + [outcome]].dropna()
    y = d[outcome].astype(int).to_numpy()
    x = d[cols].to_numpy(float)
    model = FirthLogisticRegression(fit_intercept=True)
    model.fit(x, y)
    if not model.converged_:
        return {"n": len(d), "events": int(y.sum()), "error": "Firth not converged"}
    beta = model.coef_[0]
    ci = model.conf_int(method="pl")[0]
    p = float(model.lrt().lrt_pvalues_[0])
    return {
        "n": len(d),
        "events": int(y.sum()),
        "or": math.exp(beta),
        "lo": math.exp(ci[0]),
        "hi": math.exp(ci[1]),
        "p": p,
    }


def propensity_weighting(df: pd.DataFrame, outcome: str, covariates: list[str]) -> dict:
    cols = ["dama", outcome] + covariates
    d = df[cols].dropna().copy()
    y_out = d[outcome].astype(int)
    y_trt = d["dama"].astype(int)
    x_ps = sm.add_constant(d[covariates])
    ps_model = sm.Logit(y_trt, x_ps).fit(disp=0)
    ps = ps_model.predict(x_ps)
    p_treated = y_trt.mean()
    w = np.where(
        y_trt == 1,
        p_treated / ps,
        (1 - p_treated) / (1 - ps),
    )

    x_out = sm.add_constant(d[["dama"]])
    weighted = sm.GLM(
        y_out,
        x_out,
        family=sm.families.Binomial(),
        freq_weights=w,
    ).fit(cov_type="HC0")
    ci = weighted.conf_int()
    return {
        "n": len(d),
        "events": int(y_out.sum()),
        "or": math.exp(weighted.params["dama"]),
        "lo": math.exp(ci.loc["dama", 0]),
        "hi": math.exp(ci.loc["dama", 1]),
        "p": float(weighted.pvalues["dama"]),
        "max_weight": float(w.max()),
        "weight_mean": float(w.mean()),
    }


def fmt(r: dict) -> str:
    if "error" in r:
        return f"失败：{r['error']}"
    return f"{r['or']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})，P={r['p']:.4f}，N={r['n']}，事件={r['events']}"


cohort = DF[DF["HOSPOUT"].isin([1, 3])].copy()
cohort["dama"] = (cohort["HOSPOUT"] == 3).astype(int)
cohort["comp3"] = ((cohort["DEATH3M"] == 1) | (cohort["READM3M"] == 1)).astype(int)
cohort["comp6"] = ((cohort["DEATH6M"] == 1) | (cohort["READM6M"] == 1)).astype(int)
cohort["gender_female"] = (cohort["GENDER"] == 2).astype(int)
cohort["admission_emergency"] = (cohort["ADM_WAY"] == 1).astype(int)
cohort["hftype_both"] = (cohort["HFTYPE"] == 3).astype(int)
cohort["ckd"] = (cohort["CKD"] == 1).astype(int)
cohort["pvd"] = (cohort["PVD"] == 1).astype(int)
cohort["liver_dz"] = (cohort["LIVER_DZ"] == 1).astype(int)
cohort["bnp_log"] = log_transform(cohort["BNP"])
cohort["creat_log"] = log_transform(cohort["CREAT"])

dama = cohort[cohort["dama"] == 1]
control = cohort[cohort["dama"] == 0]

full_firth_cov = [
    "AGECAT",
    "gender_female",
    "NYHA",
    "KILLIP",
    "CCI",
    "admission_emergency",
    "hftype_both",
    "bnp_log",
    "creat_log",
    "ALB",
    "HB",
]

ps_cov = [
    "AGECAT",
    "gender_female",
    "NYHA",
    "KILLIP",
    "CCI",
    "admission_emergency",
    "hftype_both",
    "ckd",
    "pvd",
    "liver_dz",
    "ALB",
    "HB",
    "Sodium",
]

lines = []
lines.append("# DAMA 正式统计结果：以全因死亡为主要结局")
lines.append("")
lines.append(f"纳入 {len(cohort)} 例；自动出院 {len(dama)} 例；正常出院 {len(control)} 例")
lines.append("")

lines.append("## 1. 主要结局：28 天全因死亡")
lines.append("")
n0 = int(control["DEATH28"].sum())
n1 = int(dama["DEATH28"].sum())
lines.append(
    f"- 正常出院：{n0}/{len(control)}（{n0/len(control)*100:.1f}%）；"
    f"自动出院：{n1}/{len(dama)}（{n1/len(dama)*100:.1f}%）"
)
lines.append(
    f"- Fisher 精确检验 P={fisher_exact([[n0, len(control)-n0], [n1, len(dama)-n1]])[1]:.3e}"
)
crude = firth_logit(cohort, "DEATH28", [])
adj = firth_logit(cohort, "DEATH28", full_firth_cov)
iptw = propensity_weighting(cohort, "DEATH28", ps_cov)
short_los = cohort[cohort["LOS"] >= 3]
los_adj = firth_logit(short_los, "DEATH28", full_firth_cov)
lines.append(f"- Firth 未调整：{fmt(crude)}")
lines.append(f"- Firth 完整调整：{fmt(adj)}")
lines.append(
    f"- IPTW 加权：{fmt({k: v for k, v in iptw.items() if k in ['n','events','or','lo','hi','p']})}"
)
lines.append(f"- 排除住院 < 3 天后 Firth 完整调整：{fmt(los_adj)}")
lines.append("")

lines.append("## 2. 死亡结局的完整 Firth 调整")
lines.append("")
lines.append("| 结局 | 正常出院 | 自动出院 | OR (95% CI) | P |")
lines.append("|---|---:|---:|---|---:|")
for outcome in ["DEATH28", "DEATH3M", "DEATH6M"]:
    n0 = int(control[outcome].sum())
    n1 = int(dama[outcome].sum())
    r = firth_logit(cohort, outcome, full_firth_cov)
    lines.append(
        f"| {outcome} | {n0}/{len(control)} ({n0/len(control)*100:.1f}%) | "
        f"{n1}/{len(dama)} ({n1/len(dama)*100:.1f}%) | "
        f"{r['or']:.2f} ({r['lo']:.2f}-{r['hi']:.2f}) | {r['p']:.3e} |"
    )
lines.append("")

lines.append("## 3. 90 天复合终点（死亡或再入院，次要结局）")
lines.append("")
for label, cov in [
    ("未调整", []),
    ("人口特征+严重度", ["AGECAT", "gender_female", "NYHA", "KILLIP", "CCI", "admission_emergency"]),
    ("完整模型", full_firth_cov),
]:
    r = fit_logit(cohort, "comp3", cov)
    lines.append(f"- {label} OR：{r['or']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})，P={r['p']:.4f}")
r = pooled_mice(cohort, "comp3", full_firth_cov)
lines.append(f"- 多重插补完整模型 OR：{r['or']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})，P={r['p']:.4f}")
r = propensity_weighting(cohort, "comp3", ps_cov)
lines.append(f"- IPTW OR：{r['or']:.3f} ({r['lo']:.3f}-{r['hi']:.3f})，P={r['p']:.4f}")
lines.append("")

lines.append("## 4. 统计说明")
lines.append("")
lines.append(
    "- AGECAT 为年龄组；GENDER 以男性为参考；BNP 与肌酐经自然对数转换。"
)
lines.append(
    "- 死亡结局事件数少，采用 Firth 惩罚 logistic 回归和 profile likelihood CI。"
)
lines.append(
    "- IPTW 与多重插补用于稳健性检验；自动出院仍可能存在未测量混杂，论文不能使用因果表述。"
)

report = "\n".join(lines)
with open("DAMA_formal_results.md", "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("\nSaved: DAMA_formal_results.md")
