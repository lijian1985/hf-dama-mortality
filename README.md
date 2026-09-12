# Analysis code — Discharge against medical advice and short-term mortality in hospitalized heart failure

Analysis code for the manuscript:

> **Discharge against medical advice and short-term all-cause mortality among hospitalized patients with heart failure: a retrospective cohort study using a Chinese public database**
>
> Jian Li, Lichun Zhou

## Data source

The analyses use the publicly available *Hospitalized patients with heart failure: integrating electronic
healthcare records and external outcome data* database from Zigong Fourth People's Hospital, Sichuan, China
(PhysioNet, version 1.3).

- PhysioNet: <https://physionet.org/content/heart-failure-zigong/1.3/>
- DOI: [10.13026/5m60-vs44](https://doi.org/10.13026/5m60-vs44)

The scripts expect a single analytic file named `HF.csv` in the working directory. `HF.csv` (the analytic
file with the derived variables used here) is **not redistributed** in this repository; it can be obtained
from the corresponding author on reasonable request, or reconstructed from the public source files using the
variable definitions published with the database. `HF.csv` is listed in `.gitignore` so that it cannot be
committed accidentally.

## Design in brief

- Retrospective cohort of 2008 hospitalized patients with heart failure (2016-2019); 11 in-hospital deaths
  were excluded, leaving 1997 patients (107 discharged against medical advice [DAMA], 1890 routine discharge).
- Primary outcome: all-cause death within 28 days from the index admission.
- Sparse-event primary analysis: Firth penalized logistic regression with profile-likelihood confidence
  intervals.
- Sensitivity analyses: unadjusted Firth model, fully adjusted complete-case model, pooled Firth models across
  five multiply imputed datasets, stabilized inverse probability of treatment weighting (IPTW), and exclusion
  of hospital stays shorter than 3 days.
- E-values for the fully adjusted and IPTW estimates.
- Admission-anchored cause-specific proportional-hazards models for readmission (death before readmission
  treated as censoring), based on the recorded event-time fields.

## Files

| File | Purpose |
|---|---|
| `DAMA_analysis.py` | Descriptive / baseline-characteristics analysis (standardized mean differences). |
| `DAMA_formal_analysis.py` | Main analysis: Firth regression, IPTW, multiple imputation, 3-month composite. Writes `DAMA_formal_results.md`. |
| `DAMA_revision_analysis.py` | Revision analyses: missing-data table, multiply-imputed primary endpoint, E-values, adjusted readmission models, admission-anchored time-to-event models, death-event timing. Writes `DAMA_revision_results.md`. |
| `make_IJC_tables.py` | Publication tables (Table 1 and supplementary tables). |
| `figure_style.py` | Shared matplotlib styling. |
| `make_figure1_flow.py` | Figure 1: participant flow diagram. |
| `make_figure2_forest.py` | Figure 2: forest plot of adjusted odds ratios. |
| `make_figure3_outcomes.py` | Figure 3: mortality and readmission rates with 95% Wilson confidence intervals. |

## Reproducing the analysis

```bash
python -m pip install -r requirements.txt

# place HF.csv in this directory, then:
python DAMA_formal_analysis.py
python DAMA_revision_analysis.py
python make_IJC_tables.py
python make_figure1_flow.py
python make_figure2_forest.py
python make_figure3_outcomes.py
```

The figure scripts import `figure_style.py`, so keep all files in the same directory.

## Environment

- Python 3.12
- `numpy`, `pandas`, `scipy`, `statsmodels`, `firthmodels`, `matplotlib` (see `requirements.txt`)

Random seed: `20260908` (used for multiple imputation).

## Reporting guideline

The study is reported according to STROBE.

## License

Released under the MIT License — see `LICENSE`.

## Citation

If you use this code, please cite the manuscript above.
