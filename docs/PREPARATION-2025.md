# BRFSS 2025 analytical dataset preparation

**Status:** Revised full-sample analytical extract created. All source respondents are retained; no machine-learning model has been trained.

## Source and provenance

- Official data page: https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html
- Official XPT archive: https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip
- Official codebook: https://www.cdc.gov/brfss/annual_data/2025/zip/codebook25_llcp-v2-508.zip
- Archive name: `LLCP2025XPT.zip`
- Archive size: 64,642,184 bytes
- Archive SHA-256: `d26a8b391aa2966b58dc2a3ffddf8da050e9f38eb143d0f5ca7b3d21ece08f98`
- ZIP integrity check: passed
- XPT member: `LLCP2025.XPT ` (the internal filename has a trailing space)
- XPT uncompressed size: 799,971,280 bytes
- XPT SHA-256: `d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2`

## Revised full-sample extract

The preparation script now writes all source respondents to:

`data/processed/brfss2025_stroke_history_analysis.csv.gz`

| Check | Result |
|---|---:|
| Source records read | 356,158 |
| Source records retained in prepared extract | 356,158 |
| Valid Yes/No stroke-history responses | 354,950 |
| Yes (reported prior stroke) | 16,353 |
| No (no reported prior stroke) | 338,597 |
| Unknown / refused / blank outcome responses | 1,208 |
| Unweighted Yes share among valid Yes/No responses | 4.6071% |
| Weighted Yes share among valid Yes/No responses | 3.6622% |
| No-to-Yes ratio among valid outcomes | 20.705:1 |

**Important correction:** the earlier extract contained only the 354,950 valid Yes/No outcome records. The current version keeps all 356,158 respondents and stores the 1,208 unknown/refused/blank outcome values as missing in `stroke_history`. They are excluded only when creating the binary classifier's training or evaluation data. Keeping them in the full prepared file preserves the original sample information for survey-design analysis and domain estimation.

## Recoding rules

| Source variable | Output field | Preparation rule |
|---|---|---|
| `CVDSTRK3` | `stroke_history` | 1 -> 1 (Yes); 2 -> 0 (No); 7, 9 and blank remain missing |
| `_AGE80` | `age_years_capped_at_80` | Keep numeric single-year ages 18–79; 80 represents age 80 or older (top-coded) |
| `SEXVAR` | `sex` | 1 -> Male; 2 -> Female |
| `_RFHYPE6` | `hypertension` | 1 -> 0 (No); 2 -> 1 (Yes); 9 -> missing |
| `_MICHD` | `chd_or_mi_history` | 1 -> 1 (reported MI or CHD); 2 -> 0; blank -> missing |
| `DIABETE4` | `diabetes_status` | Keep codes 1–4 as distinct categories; 7 and 9 -> missing |
| `_BMI5` | `bmi` | Divide by 100; blank and 9999 -> missing |
| `_SMOKER3` | `smoking_status` | Keep four smoking categories; 9 -> missing |
| `EXERANY2` | `physical_activity_30d` | 1 -> 1 (Yes); 2 -> 0 (No); 7, 9 and blank -> missing |
| `_LLCPWT` | `survey_weight` | Preserve as a survey analysis weight; not a predictor |
| `_STSTR` | `survey_stratum` | Preserve as survey-design metadata |
| `_PSU` | `survey_psu` | Preserve as survey-design metadata |

Unknown/refused predictor responses are not silently coded as “No.” No predictor imputation or class resampling occurs in this preparation step.

## Missingness among valid Yes/No outcome respondents

| Predictor | Missing | Share |
|---|---:|---:|
| BMI | 32,305 | 9.101% |
| Smoking status | 20,494 | 5.774% |
| Hypertension | 4,109 | 1.158% |
| Heart attack / CHD history | 3,657 | 1.030% |
| Physical activity | 931 | 0.262% |
| Diabetes status | 620 | 0.175% |
| Age | 0 | 0.000% |
| Sex | 0 | 0.000% |

Observed BMI values range from 12.02 to 98.43. No records are removed solely because of the observed BMI range. Raw `_AGE80` values range from 18 to 80. Values 18–79 are single-year ages; 80 is top-coded and represents age 80 or older.

## Survey design fields

The full prepared file contains:

- `_LLCPWT` -> `survey_weight`
- `_STSTR` -> `survey_stratum`
- `_PSU` -> `survey_psu`

The full file has 2,090 strata and 356,158 unique `(survey_stratum, survey_psu)` pairs. There are 94 strata with one respondent/PSU. The codebook describes `_PSU` as the primary sampling unit (equal to the annual sequence number); its identifiers should be treated as nested within strata.

Do not use these three fields as model predictors. They are retained for weighted training/evaluation and survey-design analysis. CDC's 2025 guidance documents use of the final weight, strata and PSU; the R example sets `options(survey.lonely.psu = "adjust")` for singleton-PSU variance handling.

## Reproduce locally

```bash
python -m pip install -r requirements-research.txt
python scripts/prepare_brfss.py --input data/raw/LLCP2025XPT.zip
```

The script writes the full-sample respondent-level extract and an aggregate-only `preparation_profile.json` under `data/processed/`. The extract remains local and is excluded from GitHub.

## Interpretation and limits

This is a cross-sectional US survey. `CVDSTRK3` records self-reported history of having ever been told the respondent had a stroke. The prepared dataset does not establish future stroke incidence, temporal precedence, causality, clinical diagnosis or validated individual risk. The US survey population also does not represent Nigeria.
