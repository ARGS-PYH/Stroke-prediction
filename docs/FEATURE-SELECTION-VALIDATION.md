# Feature selection and validation design

**Status:** Initial predictor set and survey-aware validation procedure specified. No machine-learning model has been trained.

## 1. Task definition

The response is the BRFSS 2025 item `CVDSTRK3`: whether the respondent reports having ever been told by a health professional that they had a stroke. This is a cross-sectional classification task for **reported prior stroke history**. It is not a model of future stroke incidence.

A predictor may be associated with reported prior stroke without being known to have occurred before the stroke. Model explanations and coefficient estimates must therefore not be interpreted as causal effects or future-risk factors.

## 2. Prespecified primary candidate features

The initial feature set is based on the question wording, codebook definitions, completeness, and the first-pass EDA. It is not selected by searching for the smallest p-values or by ranking univariate associations. The raw `_AGE80` values were inspected in the XPT: ages 18–79 are represented as single-year values, and 80 is the top-coded value for age 80 or older. Codebook frequency tables also display grouped age ranges; those tables should not be confused with the raw numeric values.

| Prepared field | BRFSS source | Type | Rationale and treatment |
|---|---|---|---|
| `age_years_capped_at_80` | `_AGE80` | Numeric | Strong descriptive gradient in EDA. Raw values represent single-year ages from 18 through 79; 80 represents age 80 or older. Keep numeric initially, but examine non-linearity (for example, a spline or age-band sensitivity model) within training folds. |
| `sex` | `SEXVAR` | Categorical | Core demographic descriptor; evaluate model performance by sex as well as overall. |
| `hypertension` | `_RFHYPE6` | Binary categorical | Self-reported high blood pressure status. Preserve unknown/refused as missing. |
| `chd_or_mi_history` | `_MICHD` | Binary categorical | Self-reported coronary heart disease or myocardial infarction history. Retain in the full model, then run the sensitivity comparison described below. |
| `diabetes_status` | `DIABETE4` | Categorical | Retain diabetes, pregnancy-only diabetes, no diabetes, and prediabetes/borderline as distinct categories. Do not collapse pregnancy-only or prediabetes into diabetes/no diabetes. |
| `bmi` | `_BMI5` | Numeric | Convert the stored value by dividing by 100. Treat blank and 9999 as missing. |
| `smoking_status` | `_SMOKER3` | Categorical | Retain current every-day, current some-days, former, and never-smoked categories. Do not treat these as equally spaced numeric values. |
| `physical_activity_30d` | `EXERANY2` | Binary categorical | Indicates any leisure-time physical activity in the past 30 days (other than regular work), not total activity or clinical fitness. |

### Model comparisons

Use three prespecified comparisons:

1. **Demographic baseline:** age and sex only.
2. **Primary model:** all eight candidate features above.
3. **Sensitivity model:** the primary model without `chd_or_mi_history`.

The sensitivity model matters because CHD/MI has the largest crude association in the EDA, but the data do not establish whether this diagnosis preceded or followed the reported stroke. If the primary model's apparent improvement largely disappears when this feature is removed, that limitation must be discussed.

Sex is included as a demographic predictor, but performance will also be examined separately for female and male respondents. These results will be descriptive fairness checks, not proof of fairness or clinical suitability.

## 3. Variables deliberately not used as predictors

- `survey_weight` (`_LLCPWT`) is an analysis weight, not a respondent health characteristic.
- `survey_stratum` (`_STSTR`) and `survey_psu` (`_PSU`) describe the sampling design; they are used to construct validation folds and survey estimates, not as model features.
- `CVDSTRK3` is the target and must never appear in the predictor matrix.
- Record identifiers and state/geographic identifiers will not be included in the initial feature set. Geography may act as a proxy and is unnecessary for the stated individual-level classification task.

Unknown, refused, and blank values must not be recoded as “No.”

## 4. Missing-value and encoding policy

All learned preprocessing must be fitted inside each training fold:

- Numeric age and BMI: median imputation fitted on training data only; retain a missingness indicator for BMI.
- Categorical fields: one-hot encode; keep unknown/refused/missing as a distinct missing category.
- Binary Yes/No health fields: treat as categorical inputs, with missing kept separate.
- Do not create preprocessing statistics using the complete dataset before splitting.
- Do not remove every respondent with a missing predictor as the default strategy; that would discard a material share of the cohort, especially for BMI and smoking.

The initial model will not apply oversampling or synthetic examples. The 20.7:1 class imbalance is part of the observed outcome distribution. Any class-weighted or resampled variant must be compared as a training-only sensitivity analysis and must not alter the untouched test set.

## 5. Full-sample preservation

The prepared data file now retains all **356,158** source respondents. The 1,208 records with `CVDSTRK3` coded 7 (Don't know/Not sure), 9 (Refused), or blank have a missing `stroke_history` value rather than being deleted.

Only the **354,950** valid Yes/No records enter binary model fitting or classification metrics. Keeping the other source records in the prepared file allows survey design information to be retained before defining the valid-outcome domain. It avoids constructing a survey design only after dropping records with non-substantive outcome responses.

## 6. Survey-aware fold construction

CDC documentation specifies `_LLCPWT` as the final weight, `_STSTR` as the sample-design stratum, and `_PSU` as the primary sampling unit. The codebook describes `_PSU` as equal to the annual sequence number. In this 2025 file, the `(survey_stratum, survey_psu)` key is unique for every respondent, and a small number of PSU identifier values are reused across different strata. Keep PSU identifiers nested within strata.

Use the helper in `scripts/survey_cv.py`:

1. Randomly allocate respondents to five folds **separately inside each `_STSTR` stratum**. The target label is not used to assign folds.
2. Treat each `(survey_stratum, survey_psu)` pair as a design unit. In the 2025 LLCP file, `_PSU` is the annual sequence number and every stratum/PSU pair identifies one respondent; there are no multi-respondent PSU groups to split across folds. The helper checks that the pair is unique.
3. Use one fold as the final approximately 20% holdout; use the other four folds for development.
4. Within the development sample, create four new stratum-wise folds for model selection and hyperparameter tuning.
5. Apply the inverse probability of the randomized second-phase fold assignment. Because the fold helper uses a random starting offset within each stratum, each respondent has marginal probability `1/K` of entering any specified fold, even when the realised number in a stratum is uneven or the stratum has one PSU. Multiply held-out weights by `K` and training weights by `K/(K-1)`; with five outer folds these factors are `5` and `1.25`. For four inner folds they are `4` and `4/3`. The helper `fold_adjusted_weights` implements these adjustments; it must not use the realised fold fraction within a stratum as an inclusion probability.
6. Filter to valid Yes/No outcomes only after fold assignment. The fold construction itself therefore retains the full source sample structure.

This follows survey cross-validation guidance: form folds in a way that respects sampling strata and PSU units, and use sampling weights in both model fitting and test-error estimation. More precisely, this is a reproducible analyst-defined second-phase validation split stratified by `_STSTR`; it is not an official CDC cross-validation recipe. Fold assignment is not stratified on the outcome, so class proportions may vary slightly between folds. We will report those fold-level outcome counts and weighted prevalences.

The 2025 file has **2,090 strata**, including **94 strata with only one respondent/PSU**. Singleton strata are not to be silently discarded or treated as if their fold assignment were deterministic; the randomized offset gives each such respondent the same marginal `1/K` chance of entering a specified fold. CDC's 2025 R example uses `options(survey.lonely.psu = "adjust")` for variance estimation. We will follow CDC's documented singleton-PSU handling for design-based descriptive estimates and explicitly document the variance procedure used for any held-out performance intervals.

## 7. Model-fitting and evaluation sequence

1. Majority-class baseline (always predicts no reported prior stroke).
2. Weighted Logistic Regression baseline.
3. Weighted Random Forest comparison.

Start with the original survey weights (combined with fold-adjustment weights) and no class rebalancing. Normalize training weights to a stable scale if required by the estimator; preserve relative weights. Do not pass the survey weights, strata, or PSU identifiers into the feature matrix.

For the held-out data, report weighted and unweighted versions where appropriate:

- Sensitivity/recall for the Yes class.
- Specificity.
- Precision.
- F1 score.
- Precision-recall AUC.
- ROC-AUC.
- Confusion matrix.
- Calibration assessment if model scores are presented as probabilities.

Accuracy is supplementary only. Report a threshold of 0.5 and, if another operating threshold is chosen, select it using training-fold results only. Never tune a threshold on the final holdout.

Design-based confidence intervals for performance metrics are a separate requirement; weighted point estimates alone must not be described as design-based inference. They will be calculated using the survey design, singleton-PSU handling, and a documented variance procedure before making population-level uncertainty claims.

## 8. Interpretation limits

- The outcome is self-reported prior stroke history, not future stroke incidence.
- Diagnoses and behaviours are measured at survey time; their temporal order relative to a past stroke is unknown.
- Feature importance and SHAP explanations describe fitted-model behaviour, not medical causation.
- BRFSS is a US adult telephone survey. Its performance results do not validate use in Nigeria.
- No result from this academic analysis is a clinical diagnosis, medical advice, or personal forecast of future stroke.

## 9. References and source documents

- CDC, [2025 BRFSS annual data and documentation](https://www.cdc.gov/brfss/annual_data/2025/annual_2025.html).
- CDC, [Complex Sampling Weights and Preparing 2025 BRFSS Module Data for Analysis](https://www.cdc.gov/brfss/annual_data/2025/pdf/Complex-Sampling-Weights-and-Preparing-Module-Data-for-Analysis-2025-508.pdf).
- CDC, [2025 BRFSS weighting description](https://www.cdc.gov/brfss/annual_data/2025/pdf/2025-Weighting-Description-508.pdf).
- Zeng et al., [K-fold cross-validation for complex sample surveys](https://doi.org/10.1002/sta4.454), 2022.
- Oh et al., [Survey-Aware Machine Learning: A Guideline for Valid Population Health Inference](https://proceedings.mlr.press/v333/oh26a.html), 2026.
