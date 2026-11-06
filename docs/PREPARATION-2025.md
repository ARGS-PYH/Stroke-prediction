# BRFSS 2025 analytical dataset preparation

**Status:** Initial clean analytical extract created and validated locally. No model has been trained.

## Output and cohort

The script `scripts/prepare_brfss.py` reads the official CDC ZIP archive directly, extracts the XPT file temporarily, and streams it in chunks. It creates the local file:

`data/processed/brfss2025_stroke_history_analysis.csv.gz`

This is respondent-level research data. It is intentionally excluded from Git and should remain in the researcher's local environment.

| Preparation check | Result |
|---|---:|
| Source records read | 356,158 |
| Valid Yes/No stroke-history outcome | 354,950 |
| Excluded unknown / refused / blank outcome | 1,208 |
| Positive outcome (reported prior stroke) | 16,353 |
| Negative outcome (no reported prior stroke) | 338,597 |
| Unweighted positive share in retained cohort | 4.6071% |
| Weighted positive share in retained Yes/No cohort | 3.6622% |
| Negative-to-positive ratio | 20.705:1 |

The 3.6622% weighted figure uses only respondents with a valid binary outcome. The codebook's 3.65% figure is its weighted percentage for the Yes response in the full codebook tabulation; the denominator treatment is therefore not identical. State the denominator whenever reporting prevalence.

## Recoding rules

| Source variable | Output field | Preparation rule |
|---|---|---|
| `CVDSTRK3` | `stroke_history` | 1 -> 1 (Yes); 2 -> 0 (No); 7, 9 and blank are excluded from the binary-outcome cohort |
| `_AGE80` | `age_years_capped_at_80` | Keep numeric age; 80 represents age 80 or older |
| `SEXVAR` | `sex` | 1 -> Male; 2 -> Female |
| `_RFHYPE6` | `hypertension` | 1 -> 0 (No); 2 -> 1 (Yes); 9 -> missing |
| `_MICHD` | `chd_or_mi_history` | 1 -> 1 (reported MI or CHD); 2 -> 0; blank -> missing |
| `DIABETE4` | `diabetes_status` | Keep codes 1–4 as separate categories (diabetes, pregnancy-only diabetes, no diabetes, prediabetes/borderline); 7 and 9 -> missing |
| `_BMI5` | `bmi` | Divide by 100; blank and 9999 -> missing |
| `_SMOKER3` | `smoking_status` | Keep four smoking categories; 9 -> missing |
| `EXERANY2` | `physical_activity_30d` | 1 -> 1 (Yes); 2 -> 0 (No); 7, 9 and blank -> missing |
| `_LLCPWT` | `survey_weight` | Preserve as a survey weight; not a predictor |
| `_STSTR` | `survey_stratum` | Preserve as survey-design metadata |
| `_PSU` | `survey_psu` | Preserve as survey-design metadata |

No predictor imputation or class resampling is performed during this step. Unknown/refused predictor responses are not silently coded as “No.” Missingness is profiled and retained for a later train-only preprocessing pipeline.

## Missingness in the retained cohort

| Field | Missing | Share |
|---|---:|---:|
| `hypertension` | 4,109 | 1.158% |
| `chd_or_mi_history` | 3,657 | 1.030% |
| `diabetes_status` | 620 | 0.175% |
| `bmi` | 32,305 | 9.101% |
| `smoking_status` | 20,494 | 5.774% |
| `physical_activity_30d` | 931 | 0.262% |
| `age_years_capped_at_80` | 0 | 0.000% |
| `sex` | 0 | 0.000% |

Observed BMI values range from 12.02 to 98.43. These values are reported as observed; no records are removed solely because of that range at this stage. Age values range from 18 to 80, with 80 top-coded (80+).

## Survey design

The prepared extract preserves `_LLCPWT`, `_STSTR`, and `_PSU` as `survey_weight`, `survey_stratum`, and `survey_psu`. They must not be fed to the model as ordinary predictors. Weight-aware descriptive summaries and carefully designed evaluation are still required; this preparation step does not calculate design-based standard errors or confidence intervals.

## Reproduce locally

```bash
python -m pip install -r requirements-research.txt
python scripts/prepare_brfss.py --input data/raw/LLCP2025XPT.zip
```

The script writes the compressed analysis extract and an aggregate-only `preparation_profile.json` under `data/processed/`. It does not write respondent-level rows into the reports directory.

Run the unit tests with:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Interpretation and limits

This is a cross-sectional US survey. `CVDSTRK3` records self-reported history of having ever been told the respondent had a stroke. The prepared dataset supports research on classification of reported prior stroke history; it does not establish future stroke incidence, temporal precedence, causality, clinical diagnosis, or validated individual risk. The US survey population also does not represent Nigeria.
