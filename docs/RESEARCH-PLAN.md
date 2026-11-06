# Research plan: BRFSS 2025 stroke-history classification

## Working title

**Explainable Machine Learning for Classification of Self-Reported Stroke History Using the 2025 Behavioral Risk Factor Surveillance System (BRFSS)**

## 1. Research scope and interpretation

The selected source is the US Centers for Disease Control and Prevention (CDC) 2025 BRFSS annual public-use data. BRFSS is a cross-sectional health survey.

The outcome represents whether a respondent reports having ever been told by a health professional that they had a stroke. It is **not** a prospective outcome and does not establish whether a person will have a stroke in the future.

This project investigates classification of self-reported stroke history and patterns learned from survey responses. It must not be described as a validated clinical diagnostic tool or a predictor of future stroke incidence.

## 2. Primary sources

- CDC BRFSS 2025 annual data and documentation: https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html
- CDC BRFSS 2025 public-use XPT file: https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip
- CDC BRFSS 2025 codebook: https://www.cdc.gov/brfss/annual_data/2025/zip/codebook25_llcp-v2-508.zip

The source file and codebook have been inspected. Record the download date, source URL, file name, file size and SHA-256 checksum in the research log.

## 3. Research questions

1. Which available demographic, behavioural and self-reported health characteristics are associated with respondents reporting a previous stroke?
2. How do interpretable baseline models compare with tree-based classifiers on held-out data?
3. How do class imbalance, missing values and survey design affect measured model performance?
4. Can SHAP explanations describe global feature patterns and individual classifications without implying causation?

## 4. Data preparation protocol

1. Preserve the original CDC ZIP file unchanged in local, ignored `data/raw/`.
2. Verify outcome coding, candidate variables, missing-value codes and survey-design fields against the official codebook.
3. Confirm record counts, types, distributions and observed ranges.
4. Define a feature inclusion rationale before fitting models.
5. Do not silently convert unknown or refused responses into ordinary negative responses.
6. Keep imputation and any other learned transformations inside the training pipeline.

**Initial preparation completed:** `scripts/prepare_brfss.py` creates a local compressed analysis extract in chunks. It retains records with `CVDSTRK3` coded 1 (Yes) or 2 (No), maps these to a binary target, leaves predictor unknown/refused values missing, and preserves the survey weight, stratum and PSU separately. The resulting cohort has 354,950 records, comprising 16,353 Yes and 338,597 No responses. See [the preparation report](PREPARATION-2025.md).

No predictor imputation, feature selection, class resampling or model fitting has occurred at this stage.

## 5. Initial exploratory data analysis

The first-pass EDA has been completed and documented in [EDA-2025.md](EDA-2025.md). The prepared valid-outcome cohort contains 354,950 respondents: 16,353 Yes and 338,597 No. The unweighted Yes share is 4.61%; the `_LLCPWT`-weighted share among valid Yes/No records is 3.66%. The class ratio is approximately 20.7:1.

Descriptive weighted subgroup estimates show a higher share reporting prior stroke among older respondents and respondents reporting hypertension, heart attack/coronary heart disease, diabetes, current smoking, and no recent leisure-time physical activity. BMI shows no simple monotonic pattern. These results are unadjusted and cross-sectional; they do not establish temporal order or causality.

BMI missingness is 9.10% and smoking-status missingness is 5.77%. The retained domain contains 2,090 strata and 19,766 distinct PSU codes; 94 strata contain only one retained PSU after the binary-outcome filter. No survey-design standard errors, confidence intervals, or p-values were calculated in this EDA. A survey-design-aware variance strategy is required before inferential or population-level uncertainty claims.

The reproducible script is `scripts/eda_brfss.py`. Aggregate outputs are written locally to the ignored `reports/generated/` directory.

## 6. Survey design and evaluation

BRFSS uses a complex sample design. The prepared file preserves the final weight (`_LLCPWT`), stratum (`_STSTR`) and primary sampling unit (`_PSU`). Define how these inform population-level descriptive estimates and the evaluation strategy before making population-level claims.

Do not describe a naive random split as representative of the survey population without qualification. Document any design-aware validation approach and its limitations.

The positive outcome is a minority class. Report sensitivity/recall, specificity, precision, F1, precision-recall AUC, ROC-AUC, confusion matrix and calibration where appropriate. Accuracy alone is insufficient. Any resampling must occur only inside training folds.

## 7. Candidate modelling workflow

Start with transparent baselines and add complexity only when justified:

- Logistic Regression as an interpretable baseline.
- Random Forest as a non-linear ensemble baseline.
- A gradient-boosted tree model only if the environment and validation design support a fair comparison.

Use reproducible seeds, record package versions, preserve the final evaluation split, and tune without inspecting final test results.

## 8. Explainability

Use SHAP only after selecting and validating a model. Global summaries and individual explanations describe how the fitted model uses inputs; they do not establish medical causation.

## 9. Fairness, generalisability and ethics

BRFSS is a US population survey. Findings cannot automatically be generalised to Nigeria or Nigerian clinical populations. Discuss differences in sampling, healthcare access, demographics, measurement and self-reporting.

Do not use real identifiable patient data. Do not collect patient information in the demo. Do not present outputs as medical advice, diagnosis or a personal forecast of future stroke.

## 10. Reproducibility deliverables

- Dataset provenance and variable dictionary.
- Data-quality and exploratory-analysis report.
- Reproducible preprocessing pipeline and tests.
- Model comparison and validation report.
- Explainability analysis.
- Updated API and UI stating the study scope.
- Limitations, ethical considerations and future work.

## 11. Current status

The repository contains a prototype API with a manually defined score, a documented research plan, a first-pass data audit, and a tested data-preparation script. It does not yet contain a trained machine-learning model, a model-evaluation pipeline or SHAP explanations. These will be developed and tested before the project claims to implement them.

## Research integrity note

Any change to the outcome definition, cohort, features, validation strategy or evaluation metrics must be recorded with a rationale before final results are reported.
