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
- CDC 2025 complex sampling instructions: https://www.cdc.gov/brfss/annual_data/2025/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2025-508.pdf

The source file and codebook have been inspected. The source ZIP and XPT checksums are recorded in the preparation report.

## 3. Research questions

1. Which available demographic, behavioural and self-reported health characteristics are associated with respondents reporting a previous stroke?
2. How do interpretable baseline models compare with tree-based classifiers on held-out data?
3. How do class imbalance, missing values and survey design affect measured model performance?
4. Can SHAP explanations describe global feature patterns and individual classifications without implying causation?

## 4. Data preparation

The revised preparation script, scripts/prepare_brfss.py, retains all **356,158** source respondents in the prepared data file. The 1,208 respondents with unknown, refused or blank stroke-history answers remain present with a missing target. Only the **354,950** valid Yes/No outcome records are used when fitting and evaluating a binary classifier.

The target is CVDSTRK3, recoded as 1=Yes and 2=No. Unknown/refused/blank responses are not treated as No. Missing predictor responses remain missing, and preprocessing/imputation is performed only within training folds.

The full file preserves the CDC final weight (_LLCPWT), stratum (_STSTR) and PSU (_PSU) fields before outcome-domain filtering.

## 5. Feature selection

The initial prespecified candidate set contains eight fields:

- Age (_AGE80; numeric single-year age 18–79, with 80 top-coded as 80 or older)
- Sex (SEXVAR)
- Hypertension (_RFHYPE6)
- History of heart attack or coronary heart disease (_MICHD)
- Diabetes status (DIABETE4)
- Body mass index (_BMI5)
- Smoking status (_SMOKER3)
- Physical activity in the past 30 days (EXERANY2)

The feature set is chosen from codebook meaning, data quality and the research question—not solely from univariate p-values. A demographic baseline (age and sex) will be compared with the full eight-feature model. A sensitivity model will exclude CHD/MI history because the survey cannot establish whether that diagnosis preceded the respondent's stroke.

Survey weights, strata, PSUs, state identifiers and the stroke outcome itself are not model predictors. See FEATURE-SELECTION-VALIDATION.md for the variable rationale, missing-value strategy and exclusions.

## 6. Survey-aware validation

The 2025 BRFSS complex-sampling instructions identify _LLCPWT as the final respondent weight, _STSTR as the stratum and _PSU as the primary sampling unit. The codebook describes _PSU as the annual sequence number. The current prepared file has 2,090 strata, 356,158 unique nested stratum/PSU units and 94 strata with only one respondent/PSU.

The validation helper scripts/survey_cv.py randomly partitions respondent-level PSU units separately within each _STSTR stratum. In this 2025 file, each (stratum, PSU) pair identifies one respondent. The outcome is not used to create folds. The randomized starting offset gives each record marginal probability 1/K of assignment to any specified fold; the helper applies inverse-probability factors K for held-out weights and K/(K-1) for training weights before combining them with _LLCPWT.

The planned sequence is a five-fold outer assignment: one fold is reserved as a final approximately 20% holdout, and the remaining four folds form the development set. Four additional stratum-wise folds within that development set are used for model selection. The final holdout is not used to tune models or thresholds.

Singleton strata must not be silently discarded. CDC's 2025 R example uses the survey package and sets options(survey.lonely.psu = "adjust") for variance estimation. Weighted point estimates and design-based uncertainty estimates will be distinguished in the final report.

The detailed feature rationale and validation plan are documented in FEATURE-SELECTION-VALIDATION.md. The initial age summaries use the released _AGE80 variable, where 80 represents 80 or older.

## 7. Candidate modelling workflow

Start with transparent baselines and add complexity only when justified:

- Majority-class reference and weighted Logistic Regression comparisons have been fitted on the prespecified outer holdout; development-only out-of-fold predictions selected decision thresholds.
- Random Forest remains the next model comparison.

Use reproducible seeds, keep preprocessing inside each fold, and record package versions. Start with survey weights and no class rebalancing; compare any class weighting or resampling only as a training-fold sensitivity analysis.

## 8. Evaluation

Report sensitivity/recall, specificity, precision, F1, precision-recall AUC, ROC-AUC, confusion matrix and calibration where appropriate. Accuracy alone is insufficient.

Report weighted and unweighted metrics where useful, making the denominator and interpretation explicit. If a non-default decision threshold is selected, choose it using training-fold results only, never the final holdout.

Design-based confidence intervals for performance metrics require a separate variance-estimation procedure that accounts for the survey strata, PSU and singleton-stratum handling. Weighted point estimates alone are not design-based confidence intervals.

## 9. Explainability

Use SHAP only after selecting and validating a model. Global summaries and individual explanations describe how the fitted model uses inputs; they do not establish medical causation or future stroke risk.

## 10. Fairness, generalisability and ethics

BRFSS is a US population survey. Findings cannot automatically be generalised to Nigeria or Nigerian clinical populations. Discuss differences in sampling, healthcare access, demographics, measurement and self-reporting.

Evaluate performance across sex groups as a descriptive check. Do not claim fairness from one metric or from subgroup performance alone.

Do not use real identifiable patient data. Do not collect patient information in the demo. Do not present outputs as medical advice, diagnosis or a personal forecast of future stroke.

## 11. Reproducibility deliverables

- Dataset provenance and data dictionary.
- Data-quality and exploratory-analysis report.
- Reproducible full-sample preparation pipeline.
- Feature-selection and survey-aware validation specification.
- Model comparison and validation report.
- Explainability analysis.
- Automated tests for preparation and fold construction.
- Updated API and UI that clearly state the research scope.
- Limitations, ethical considerations and future work.

## 12. Current status

The repository contains an early FastAPI/HTML prototype, full-sample preparation and EDA scripts, prespecified feature/validation design, tested survey-aware fold utility, and an initial weighted Logistic Regression comparison. The baseline report records weighted/unweighted metrics and explicitly notes that no Random Forest, design-based confidence interval for predictive performance, SHAP explanation, or clinical validation has been completed.

## Research integrity note

Any change to the outcome definition, cohort, features, validation strategy or evaluation metrics must be recorded with a rationale before final results are reported.
