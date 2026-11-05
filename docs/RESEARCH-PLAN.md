# Research plan: BRFSS 2025 stroke-history classification

## Working title

**Explainable Machine Learning for Classification of Self-Reported Stroke History Using the 2025 Behavioral Risk Factor Surveillance System (BRFSS)**

## 1. Research scope and interpretation

The selected source is the US Centers for Disease Control and Prevention (CDC) 2025 BRFSS annual public-use data. BRFSS is a cross-sectional health survey.

The outcome available in this dataset represents whether a respondent reports having ever been told by a health professional that they had a stroke. It is **not** a prospective outcome and does not establish whether a person will have a stroke in the future.

Accordingly, this project will investigate classification of self-reported stroke history and the associations learned from survey responses. The system must not be described as a validated clinical diagnostic tool or a predictor of future stroke incidence.

## 2. Primary sources

- CDC BRFSS 2025 annual data and documentation: https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html
- CDC BRFSS 2025 public-use XPT file: https://www.cdc.gov/brfss/annual_data/2025/files/LLCP2025XPT.zip
- CDC BRFSS questionnaires, codebooks and technical documentation: use the official 2025 annual-data page above.

The dataset dictionary and codebook must be checked before any field is selected or recoded. Record the download date, source URL, file name, file size and SHA-256 checksum in the research log.

## 3. Research questions

1. Which available demographic, behavioural and self-reported health characteristics are associated with respondents reporting a previous stroke?
2. How do interpretable baseline models compare with tree-based classifiers on held-out data?
3. How do class imbalance, missing values and survey design affect measured model performance?
4. Can SHAP explanations describe the model's global feature patterns and individual classifications without implying causation?

## 4. Data preparation protocol

1. Preserve the original CDC XPT file unchanged in a local, ignored `data/raw/` directory.
2. Inspect the official codebook and identify the exact outcome coding, valid response categories, missing-value codes and survey-design variables.
3. Confirm record counts, duplicate rows, data types, ranges and invalid or implausible values.
4. Report both unweighted counts and appropriately weighted descriptive estimates where the survey design supports them.
5. Define a feature inclusion and exclusion rationale before model fitting. Avoid using variables that directly reveal the outcome or create target leakage.
6. Document missing-data decisions; do not silently convert unknown or refused responses into ordinary negative responses.
7. Keep all transformations inside a reproducible preprocessing pipeline.

## 5. Survey design and evaluation

BRFSS uses a complex sample design. The analysis must inspect the 2025 documentation for the prescribed final weight, stratum and primary sampling-unit fields and use them for population-level descriptive estimates and any modelling/evaluation approach that claims to represent the survey population.

Do not use a naive random split and then describe the resulting records as a representative sample without qualification. Define a reproducible, design-aware validation strategy and document its limitations.

Stroke-history responses may be a minority class. Report class prevalence and use metrics that are informative under imbalance, including sensitivity/recall, specificity, precision, F1, precision-recall AUC, ROC-AUC and a confusion matrix. Include calibration assessment when interpreting model scores. Accuracy alone is insufficient.

Any resampling or class rebalancing must happen only inside training folds. Never resample the full dataset before splitting.

## 6. Candidate modelling workflow

Start with transparent baselines and add complexity only when justified:

- Logistic Regression as an interpretable baseline.
- Random Forest as a non-linear ensemble baseline.
- A gradient-boosted tree model only if the environment and validation design support a fair comparison.

Use reproducible random seeds, record package versions, preserve the final evaluation split, and tune hyperparameters without looking at final test results.

## 7. Explainability

Use SHAP only after selecting and validating a model. Provide global summaries and carefully labelled individual explanations. Explain that these describe how the fitted model uses its inputs; they do not establish medical causation or prove that changing a feature changes stroke risk.

## 8. Fairness, generalisability and ethics

BRFSS is a US population survey. Findings cannot automatically be generalised to Nigeria or to Nigerian clinical populations. Discuss differences in sampling, healthcare access, demographics, measurement and self-reporting.

Do not use real identifiable patient data. Do not collect patient information in the demo. Do not present outputs as medical advice, diagnosis or a personal forecast of future stroke.

## 9. Reproducibility deliverables

- Dataset provenance and variable dictionary.
- Data-quality and exploratory-analysis notebook.
- Reproducible preprocessing pipeline.
- Model comparison and validation report.
- Explainability analysis.
- Automated tests for preprocessing and prediction code.
- Updated API and UI that clearly state the research scope.
- A report of limitations, ethical considerations and future work.

## 10. Current status

This repository currently contains a prototype API with a manually defined score. It does not yet contain a trained machine-learning model, a real model-evaluation pipeline or SHAP explanations. Those components will be developed and tested before the project claims to implement them.

## Research integrity note

The analysis plan may be refined after the codebook and data have been inspected. Any change to the outcome definition, cohort, features, validation strategy or evaluation metrics must be recorded with a rationale before final results are reported.
