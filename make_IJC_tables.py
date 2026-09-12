# -*- coding: utf-8 -*-
"""Generate reproducible publication tables for the IJC submission package."""

import math

import numpy as np
import pandas as pd
from pathlib import Path


DF = pd.read_csv("HF.csv")
cohort = DF[DF["HOSPOUT"].isin([1, 3])].copy()
cohort["dama"] = (cohort["HOSPOUT"] == 3).astype(int)
dama = cohort[cohort["dama"] == 1]
routine = cohort[cohort["dama"] == 0]

PACKAGE = Path(__file__).resolve().parent


def continuous_summary(x: pd.Series) -> str:
    s = x.dropna()
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}-{s.quantile(0.75):.1f})"


def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    a = a.dropna().to_numpy(dtype=float)
    b = b.dropna().to_numpy(dtype=float)
    pooled = math.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def smd_binary(a: pd.Series, b: pd.Series) -> float:
    pa = float(a.mean())
    pb = float(b.mean())
    den = math.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pa - pb) / den) if den > 0 else 0.0


def count_pct(a: pd.Series, value: object) -> str:
    n = int((a == value).sum())
    return f"{n} ({n / len(a) * 100:.1f})"


def binary_cell(a: pd.Series) -> str:
    n = int(a.sum())
    return f"{n} ({n / len(a) * 100:.1f})"


def binary_row(a: pd.Series, b: pd.Series) -> tuple[str, str]:
    return binary_cell(a), binary_cell(b)


rows: list[tuple[str, str, str, str]] = []

# Demographics and hospitalization.
for label, code in [
    ("21-29", 1),
    ("30-39", 2),
    ("40-49", 3),
    ("50-59", 4),
    ("60-69", 5),
    ("70-79", 6),
    ("80-89", 7),
    (">89", 8),
]:
    rows.append(
        (
            f"Age group, years: {label}",
            count_pct(routine["AGECAT"], code),
            count_pct(dama["AGECAT"], code),
            "",
        )
    )

rows.append(("Age group ordinal, median (IQR)", continuous_summary(routine["AGECAT"]), continuous_summary(dama["AGECAT"]), ""))
rows.append(
    (
        "Female sex",
        *binary_row((routine["GENDER"] == 2).astype(int), (dama["GENDER"] == 2).astype(int)),
    )
)

for label, code in [("II", 2), ("III", 3), ("IV", 4)]:
    rows.append(("NYHA class: " + label, count_pct(routine["NYHA"], code), count_pct(dama["NYHA"], code), ""))
for label, code in [("I", 1), ("II", 2), ("III", 3), ("IV", 4)]:
    rows.append(("Killip class: " + label, count_pct(routine["KILLIP"], code), count_pct(dama["KILLIP"], code), ""))

rows.append(("Heart failure type: left-sided", count_pct(routine["HFTYPE"], 1), count_pct(dama["HFTYPE"], 1), ""))
rows.append(("Heart failure type: right-sided", count_pct(routine["HFTYPE"], 2), count_pct(dama["HFTYPE"], 2), ""))
rows.append(("Heart failure type: both", count_pct(routine["HFTYPE"], 3), count_pct(dama["HFTYPE"], 3), ""))
rows.append(
    (
        "Emergency admission",
        *binary_row((routine["ADM_WAY"] == 1).astype(int), (dama["ADM_WAY"] == 1).astype(int)),
    )
)
rows.append(("Charlson comorbidity index, median (IQR)", continuous_summary(routine["CCI"]), continuous_summary(dama["CCI"]), ""))
rows.append(("Length of hospital stay, days, median (IQR)", continuous_summary(routine["LOS"]), continuous_summary(dama["LOS"]), ""))
rows.append(("No. of prescribed drug classes, median (IQR)", continuous_summary(routine["NDRUGS"]), continuous_summary(dama["NDRUGS"]), ""))

# Clinical measurements.
rows.append(("BMI, kg/m2, median (IQR)", continuous_summary(routine["BMI"]), continuous_summary(dama["BMI"]), ""))
rows.append(("Systolic blood pressure, mmHg, median (IQR)", continuous_summary(routine["SBP"]), continuous_summary(dama["SBP"]), ""))
rows.append(("Heart rate, beats/min, median (IQR)", continuous_summary(routine["PULSE"]), continuous_summary(dama["PULSE"]), ""))
rows.append(("BNP, pg/mL, median (IQR)", continuous_summary(routine["BNP"]), continuous_summary(dama["BNP"]), ""))
rows.append(("Creatinine, umol/L, median (IQR)", continuous_summary(routine["CREAT"]), continuous_summary(dama["CREAT"]), ""))
rows.append(("eGFR, mL/min/1.73 m2, median (IQR)", continuous_summary(routine["EGFR"]), continuous_summary(dama["EGFR"]), ""))
rows.append(("Albumin, g/L, median (IQR)", continuous_summary(routine["ALB"]), continuous_summary(dama["ALB"]), ""))
rows.append(("Hemoglobin, g/L, median (IQR)", continuous_summary(routine["HB"]), continuous_summary(dama["HB"]), ""))
rows.append(("Serum sodium, mmol/L, median (IQR)", continuous_summary(routine["Sodium"]), continuous_summary(dama["Sodium"]), ""))

cont_cols = [
    ("BMI", "BMI"),
    ("SBP", "Systolic blood pressure"),
    ("PULSE", "Heart rate"),
    ("BNP", "BNP"),
    ("CREAT", "Creatinine"),
    ("EGFR", "eGFR"),
    ("ALB", "Albumin"),
    ("HB", "Hemoglobin"),
    ("Sodium", "Serum sodium"),
    ("LOS", "Length of hospital stay"),
    ("NDRUGS", "No. of drug classes"),
    ("AGECAT", "Age group ordinal"),
]
cont_smd = {label: smd_continuous(dama[col], routine[col]) for col, label in cont_cols}

# Comorbidities.
for label, col in [
    ("Prior myocardial infarction", "MI"),
    ("Peripheral vascular disease", "PVD"),
    ("Cerebrovascular disease", "CVD"),
    ("Dementia", "DEMENTIA"),
    ("COPD", "COPD"),
    ("Diabetes", "DIABETES"),
    ("Chronic kidney disease", "CKD"),
    ("Solid tumor", "TUMOR"),
    ("Liver disease", "LIVER_DZ"),
    ("Acute renal failure", "ARF"),
]:
    rows.append((label, *binary_row((routine[col] == 1).astype(int), (dama[col] == 1).astype(int))))

# Medications.
for label, col in [
    ("Loop diuretic", "LOOP_DI"),
    ("Mineralocorticoid receptor antagonist", "MRA"),
    ("ACEI/ARB", "ACEIARB"),
    ("Beta-blocker", "BETA_B"),
    ("Digoxin", "DIG"),
    ("Antiplatelet", "APLT"),
    ("Anticoagulant", "ANTICO"),
    ("Statin", "STATIN"),
    ("Inotrope", "INOTRP"),
    ("Nitrate", "NITRO"),
    ("Chinese patent medicine", "TCM"),
]:
    rows.append((label, *binary_row((routine[col] == 1).astype(int), (dama[col] == 1).astype(int))))

# Fill SMDs for rows not captured by helper.
smd_map = {
    "Female sex": smd_binary((dama["GENDER"] == 2).astype(int), (routine["GENDER"] == 2).astype(int)),
    "Emergency admission": smd_binary((dama["ADM_WAY"] == 1).astype(int), (routine["ADM_WAY"] == 1).astype(int)),
    "Charlson comorbidity index, median (IQR)": smd_continuous(dama["CCI"], routine["CCI"]),
    "Length of hospital stay, days, median (IQR)": cont_smd["Length of hospital stay"],
    "No. of prescribed drug classes, median (IQR)": cont_smd["No. of drug classes"],
    "BMI, kg/m2, median (IQR)": cont_smd["BMI"],
    "Systolic blood pressure, mmHg, median (IQR)": cont_smd["Systolic blood pressure"],
    "Heart rate, beats/min, median (IQR)": cont_smd["Heart rate"],
    "BNP, pg/mL, median (IQR)": cont_smd["BNP"],
    "Creatinine, umol/L, median (IQR)": cont_smd["Creatinine"],
    "eGFR, mL/min/1.73 m2, median (IQR)": cont_smd["eGFR"],
    "Albumin, g/L, median (IQR)": cont_smd["Albumin"],
    "Hemoglobin, g/L, median (IQR)": cont_smd["Hemoglobin"],
    "Serum sodium, mmol/L, median (IQR)": cont_smd["Serum sodium"],
    "Age group ordinal, median (IQR)": cont_smd["Age group ordinal"],
    "Prior myocardial infarction": smd_binary((dama["MI"] == 1).astype(int), (routine["MI"] == 1).astype(int)),
    "Peripheral vascular disease": smd_binary((dama["PVD"] == 1).astype(int), (routine["PVD"] == 1).astype(int)),
    "Cerebrovascular disease": smd_binary((dama["CVD"] == 1).astype(int), (routine["CVD"] == 1).astype(int)),
    "Dementia": smd_binary((dama["DEMENTIA"] == 1).astype(int), (routine["DEMENTIA"] == 1).astype(int)),
    "COPD": smd_binary((dama["COPD"] == 1).astype(int), (routine["COPD"] == 1).astype(int)),
    "Diabetes": smd_binary((dama["DIABETES"] == 1).astype(int), (routine["DIABETES"] == 1).astype(int)),
    "Chronic kidney disease": smd_binary((dama["CKD"] == 1).astype(int), (routine["CKD"] == 1).astype(int)),
    "Solid tumor": smd_binary((dama["TUMOR"] == 1).astype(int), (routine["TUMOR"] == 1).astype(int)),
    "Liver disease": smd_binary((dama["LIVER_DZ"] == 1).astype(int), (routine["LIVER_DZ"] == 1).astype(int)),
    "Acute renal failure": smd_binary((dama["ARF"] == 1).astype(int), (routine["ARF"] == 1).astype(int)),
    "Loop diuretic": smd_binary((dama["LOOP_DI"] == 1).astype(int), (routine["LOOP_DI"] == 1).astype(int)),
    "Mineralocorticoid receptor antagonist": smd_binary((dama["MRA"] == 1).astype(int), (routine["MRA"] == 1).astype(int)),
    "ACEI/ARB": smd_binary((dama["ACEIARB"] == 1).astype(int), (routine["ACEIARB"] == 1).astype(int)),
    "Beta-blocker": smd_binary((dama["BETA_B"] == 1).astype(int), (routine["BETA_B"] == 1).astype(int)),
    "Digoxin": smd_binary((dama["DIG"] == 1).astype(int), (routine["DIG"] == 1).astype(int)),
    "Antiplatelet": smd_binary((dama["APLT"] == 1).astype(int), (routine["APLT"] == 1).astype(int)),
    "Anticoagulant": smd_binary((dama["ANTICO"] == 1).astype(int), (routine["ANTICO"] == 1).astype(int)),
    "Statin": smd_binary((dama["STATIN"] == 1).astype(int), (routine["STATIN"] == 1).astype(int)),
    "Inotrope": smd_binary((dama["INOTRP"] == 1).astype(int), (routine["INOTRP"] == 1).astype(int)),
    "Nitrate": smd_binary((dama["NITRO"] == 1).astype(int), (routine["NITRO"] == 1).astype(int)),
    "Chinese patent medicine": smd_binary((dama["TCM"] == 1).astype(int), (routine["TCM"] == 1).astype(int)),
}

for i, row in enumerate(rows):
    if len(row) == 3 or row[3] == "":
        value = smd_map.get(row[0], float("nan"))
        smd_cell = f"{value:.3f}" if not math.isnan(value) else ""
        rows[i] = (row[0], row[1], row[2], smd_cell)

lines = [
    "| Characteristic | Routine discharge (n=1890) | DAMA (n=107) | SMD |",
    "|---|---:|---:|---:|",
]
for label, routine_cell, dama_cell, smd_cell in rows:
    lines.append(f"| {label} | {routine_cell} | {dama_cell} | {smd_cell} |")

lines.append("")
missing_lines = [
    "Missing values: CCI 5 (0.3%); BNP 34 (1.7%); creatinine 22 (1.1%); eGFR 62 (3.1%); albumin 100 (5.0%); hemoglobin 27 (1.4%); serum sodium 10 (0.5%). Detailed counts are in Supplementary Table 3.",
]
lines.append("Footnotes: Values are n (%) unless otherwise stated. Percentages use non-missing values as denominators. SMD, standardized mean difference; positive values indicate a higher frequency or value in the DAMA group. CCI, Charlson comorbidity index; eGFR, estimated glomerular filtration rate; ACEI/ARB, angiotensin-converting enzyme inhibitor/angiotensin receptor blocker.")
lines.append("")
lines.extend(missing_lines)

with open(PACKAGE / "DAMA_table1_IJC.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

package = [
    "# Tables for IJC Submission",
    "",
    "## Table 1. Baseline characteristics according to discharge status",
    "",
    *lines,
    "",
    "Note: This file contains the only main-text table (Table 1). Clinical outcome estimates, sensitivity analyses, missing-data counts, and event-time analyses are reported in Supplementary Tables 1-4 in the accompanying supplementary material.",
    "",
]
with open(PACKAGE / "04_Tables_IJC.md", "w", encoding="utf-8") as f:
    f.write("\n".join(package))

print("\n".join(lines))
print("\nSaved: DAMA_table1_IJC.md")
print("Saved: 04_Tables_IJC.md")
