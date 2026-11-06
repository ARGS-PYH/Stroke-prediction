# BRFSS 2025 dataset audit

**Audit status:** Initial structural and data-quality check completed  
**Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS), 2025 LLCP public-use file  
**Scope:** Aggregate inspection only; no model has been trained.

## Source and integrity

- Official data page: https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html
- Original archive: https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip
- Official 2025 codebook: https://www.cdc.gov/brfss/annual_data/2025/zip/codebook25_llcp-v2-508.zip
- Uploaded archive size: 64,642,184 bytes
- Archive SHA-256: `d26a8b391aa2966b58dc2a3ffddf8da050e9f38eb143d0f5ca7b3d21ece08f98`
- ZIP integrity check: passed
- Archive member: `LLCP2025.XPT ` (the member name has a trailing space)
- Uncompressed XPT size: 799,971,280 bytes
- Extracted XPT SHA-256: `d99262f8854f018a600c3538ee78e4308569af700abe6b92b36180336c24dec2`

The audit script normalizes the archive member name when extracting it into a temporary directory. Keep the original ZIP unchanged for reproducibility.

## Structure

| Check | Result |
|---|---:|
| Respondent records | 356,158 |
| Variables | 283 |
| Shortlisted audit variables present | 12 of 12 |
| Distinct sample-design strata values | 2,090 |
| Distinct PSU values | 19,779 |

The selected variables are `CVDSTRK3`, `_AGE80`, `SEXVAR`, `_RFHYPE6`, `_MICHD`, `DIABETE4`, `_BMI5`, `_SMOKER3`, `EXERANY2`, `_LLCPWT`, `_STSTR`, and `_PSU`.

## Outcome profile

The codebook identifies `CVDSTRK3` as the question “(Ever told) (you had) a stroke.” It is a self-reported history measure.

| Response | Code | Records |
|---|---:|---:|
| Yes | 1 | 16,353 |
| No | 2 | 338,597 |
| Don't know / Not sure | 7 | 1,041 |
| Refused | 9 | 164 |
| Blank / missing | — | 3 |

- Valid yes/no responses: 354,950
- Unweighted “Yes” among valid yes/no responses: **4.6071%**
- Weighted “Yes” among nonblank outcome responses using `_LLCPWT`: **3.6507%**
- No-to-yes class ratio: approximately **20.7:1**

Do not convert codes 7 or 9, or blank responses, to “No.” They are non-substantive outcome responses and should be handled separately.

## Missingness in selected fields

| Variable | Blank / missing | Share of all records |
|---|---:|---:|
| `CVDSTRK3` | 3 | 0.001% |
| `_AGE80` | 0 | 0.000% |
| `SEXVAR` | 0 | 0.000% |
| `_RFHYPE6` | 0 | 0.000% |
| `_MICHD` | 4,013 | 1.127% |
| `DIABETE4` | 6 | 0.002% |
| `_BMI5` | 32,562 | 9.143% |
| `_SMOKER3` | 0 | 0.000% |
| `EXERANY2` | 2 | 0.001% |

These are blank-value counts only. Some categorical fields also contain explicit unknown/refused codes, which must be treated according to the 2025 codebook rather than as ordinary categories or negative responses. `_BMI5` is stored as BMI multiplied by 100; the observed nonblank range was 12.02 to 98.43 after scaling.

## Methodological implications

1. The outcome is **reported prior stroke history**, not future stroke incidence. The defensible project framing is explainable classification of self-reported stroke history, not prospective stroke-risk prediction or clinical diagnosis.
2. The positive class is uncommon. Accuracy alone is not an appropriate headline metric. Report sensitivity/recall, specificity, precision, F1, PR-AUC, ROC-AUC, and the confusion matrix.
3. BRFSS uses a complex survey design. The file includes final respondent weights (`_LLCPWT`), strata (`_STSTR`), and primary sampling units (`_PSU`). This audit calculates a weighted percentage but does **not** calculate design-based standard errors or confidence intervals. Those require a survey-design-aware analysis.
4. BMI has a material missing-data share. Missingness should be described and handled within the training/evaluation workflow; do not silently drop or impute before defining the analysis population.
5. BRFSS is a US adult survey. Findings should not be generalized to Nigeria without separate evidence.

## Reproducing the audit

Place the official ZIP archive in `data/raw/LLCP2025XPT.zip`. The script accepts the ZIP directly and handles the trailing space in its internal XPT filename.

```bash
python -m pip install -r requirements-research.txt
python scripts/audit_brfss.py --input data/raw/LLCP2025XPT.zip
```

The script writes aggregate files to `reports/generated/`:

- `dataset_audit.json`
- `variable_audit.csv`

It reads the XPT in chunks and does not write respondent-level records. The generated directory and all source/processed datasets are excluded from Git.

**Audit boundary:** this is a first-pass structure, outcome, and missingness audit. It is not a full statistical analysis, survey variance estimation, model evaluation, or clinical validation.
