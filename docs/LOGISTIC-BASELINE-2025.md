# Initial weighted Logistic Regression baseline

**Status:** Initial model comparison completed on one protected outer holdout. This is an academic classification experiment, not a clinical prediction system.

## 1. Evaluation design

- Prepared source sample: 356,158 respondents.
- Valid binary outcome cohort: 354,950 respondents.
- Outer validation: five folds assigned within each _STSTR stratum, before filtering unknown/refused/blank stroke outcomes.
- Final holdout: fold 0, containing 71,007 valid-outcome respondents (3,221 Yes).
- Development sample: 283,943 valid-outcome respondents (13,132 Yes).
- Threshold selection: four-fold stratum-wise out-of-fold predictions in the development sample only.
- Decision threshold: maximum weighted Youden's J on development OOF predictions. The final holdout was not used to choose a threshold.
- Regularization: Logistic Regression C=1.0, fixed in advance for this first comparison; no hyperparameter search.
- Weighting: _LLCPWT combined with inverse marginal fold-assignment probabilities. Training weights were rescaled to mean one for model fitting.

## 2. Preprocessing

All learned preprocessing was fitted inside each training fold:

- Age: _AGE80 modelled using a spline transformation. Ages 80 and older are top-coded to 80.
- BMI: training-fold median imputation, missingness indicator, and scaling.
- Categorical fields: one-hot encoding.
- Unknown/refused binary predictor responses: represented as a separate missing category.
- No SMOTE, oversampling, class weighting, or other class rebalancing was applied.

The three prespecified comparisons were:

1. Demographic baseline: age and sex.
2. Primary model: all eight features.
3. Sensitivity model: all features except CHD/MI history.

## 3. Untouched holdout results

Results below are weighted with the BRFSS survey weight and the outer-fold adjustment, using the threshold selected from the development sample.

| Model | Selected threshold | Recall | Specificity | Precision | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Demographic baseline | 0.0352 | 80.25% | 56.44% | 6.16% | 0.1145 | 0.7320 | 0.0773 |
| Primary eight features | 0.0339 | 78.71% | 70.92% | 8.80% | 0.1583 | 0.8134 | 0.1421 |
| Sensitivity without CHD/MI | 0.0316 | 81.13% | 65.16% | 7.66% | 0.1400 | 0.7833 | 0.1037 |

The majority-class reference (always predicts no reported prior stroke) has weighted ROC-AUC 0.500 and PR-AUC 0.0344 on this holdout.

### Interpretation

- The primary eight-feature model has the strongest ranking performance of the three comparisons: ROC-AUC 0.813 and PR-AUC 0.142.
- Removing CHD/MI history reduces ROC-AUC to 0.783 and PR-AUC to 0.104. This suggests that CHD/MI history contributes useful classification information in this dataset. It does not show that CHD/MI caused stroke or that it occurred before stroke.
- The selected threshold for the primary model produces 78.71% weighted sensitivity and 70.92% weighted specificity, but precision is only 8.80%. Many respondents classified positive do not report prior stroke.
- At the conventional 0.5 threshold, the primary model's recall is only 0.06% and specificity is nearly 100%. The high accuracy at 0.5 is therefore misleading under this class imbalance. The threshold must be reported alongside all threshold-dependent metrics.

## 4. Weighted and unweighted comparison

For the primary model at its development-selected threshold:

| Metric | Weighted | Unweighted |
|---|---:|---:|
| Recall | 78.71% | 82.02% |
| Specificity | 70.92% | 60.36% |
| Precision | 8.80% | 8.95% |
| F1 | 0.1583 | 0.1614 |
| ROC-AUC | 0.8134 | 0.7879 |
| PR-AUC | 0.1421 | 0.1473 |

Differences between weighted and unweighted measures reinforce why the survey weights should not be ignored. Weighted and unweighted precision estimates remain low.

## 5. Limitations and next steps

1. This is one initial outer holdout comparison. It does not yet include repeated outer validation or design-based confidence intervals for model performance.
2. C=1.0 was fixed; hyperparameters were not tuned.
3. The decision threshold was selected using weighted Youden's J. Other operating criteria may produce different sensitivity/precision trade-offs.
4. The outcome records self-reported prior stroke history. It does not represent future stroke incidence.
5. Some self-reported diagnoses and behaviours may have occurred after the respondent's stroke.
6. BRFSS is a US adult survey; these results do not validate clinical use or generalise directly to Nigeria.
7. No Random Forest, SHAP explanation, or clinical evaluation has been completed.

The next planned experiment is a weighted Random Forest comparison using the same outer holdout and development-only threshold-selection rule. Design-based uncertainty for performance metrics should be addressed before making population-level performance claims.
