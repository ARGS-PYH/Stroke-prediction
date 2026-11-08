# BRFSS 2025: Random Forest and Logistic Regression comparison

**Status:** Initial Random Forest comparison completed on the same prespecified outer holdout as the primary Logistic Regression baseline.

## 1. Cohort and validation

- Full prepared source sample: **356,158** respondents.
- Valid binary outcome records: **354,950**.
- Unknown, refused or blank stroke-history responses preserved in the full file with a missing target: **1,208**.
- Development cohort: **283,943** valid Yes/No respondents, including 13,132 Yes.
- Final holdout: **71,007** valid Yes/No respondents, including 3,221 Yes.
- Outer fold assignment: five folds, assigned within each `_STSTR` stratum before outcome filtering. Fold assignment did not use the stroke-history outcome.
- Threshold selection: four-fold development-only out-of-fold predictions; weighted Youden's J. The final holdout was not used for threshold selection.
- The same fold assignment, training cohort, validation cohort, weight adjustments and selected feature set were used for both algorithms.

The primary model uses the eight prespecified features: age, sex, hypertension, CHD/MI history, diabetes status, BMI, smoking status and physical activity in the past 30 days.

## 2. Model configurations

**Logistic Regression:** age spline, median BMI imputation with a missingness indicator and scaling, one-hot categorical encoding, (C=1.0), no class weighting or resampling.

**Random Forest:** 80 trees, maximum depth 12, minimum 75 observations per leaf, square-root feature selection, bootstrap sampling with 60% of training rows per tree, no class weighting. The final BRFSS weight combined with the fold-adjustment factor was supplied as `sample_weight` during fitting. Training weights were normalised to mean one without changing relative weights.

All preprocessing was fitted within each training fold. Unknown/refused predictor responses were represented as missing categories or values; they were not recoded as No. No SMOTE, oversampling, or class rebalancing was used.

## 3. Results on the untouched holdout

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Development-selected threshold | 0.0339 | 0.0355 |
| Weighted recall / sensitivity | 78.71% | **79.08%** |
| Weighted specificity | **70.92%** | 69.43% |
| Weighted precision | **8.80%** | 8.44% |
| Weighted F1 | **0.1583** | 0.1525 |
| Weighted ROC-AUC | 0.8134 | **0.8154** |
| Weighted PR-AUC (average precision) | 0.1421 | **0.1452** |
| Weighted Brier score | 0.0315 | **0.0311** |

The weighted majority-class reference (always predicts No) has ROC-AUC 0.500 and PR-AUC 0.0344 on this holdout. Both fitted models improve considerably on that ranking reference.

### Interpretation

- Random Forest has a very small advantage in ranking performance: ROC-AUC improves by about 0.0020 and PR-AUC by about 0.0031.
- Logistic Regression has higher specificity, precision and F1 at its development-selected threshold.
- The differences are small. Without design-based uncertainty intervals or repeated outer validation, we cannot conclude that one algorithm is genuinely superior.
- Both models have low precision at their selected thresholds (about 8–9%). Many positive classifications do not correspond to respondents reporting a previous stroke.
- At the conventional 0.5 threshold, sensitivity is effectively zero for both models. Threshold choice matters substantially under this class imbalance.

The practical conclusion is **near parity, with a trade-off**: Random Forest ranks slightly better; Logistic Regression offers slightly better positive-class precision and F1 at the selected operating point. Neither is a clear winner.

## 4. Important weighting limitation

The Random Forest receives the fold-adjusted BRFSS weights through scikit-learn's `sample_weight` fit argument. Its standard bootstrap mechanism still draws rows uniformly rather than using a probability-proportional-to-survey-weight bootstrap. This is therefore a survey-weighted machine-learning baseline, not a complete design-based Random Forest procedure.

The final metrics are weighted point estimates. We have not calculated design-based confidence intervals or accounted for design variance in model performance. The 94 singleton strata also require the documented BRFSS lonely-PSU handling for any design-based variance estimation.

## 5. Reproducibility

Run from the repository root after creating the full-sample local analysis extract:

```bash
python -m pip install -r requirements-research.txt
python scripts/train_model_comparison.py --primary-only
```

The script writes aggregate JSON/CSV metrics to `reports/generated/` and serialised models to `models/`. Both locations are excluded from Git. It can run all three prespecified feature sets by omitting `--primary-only`.

## 6. Limitations and next steps

1. This is one fixed outer holdout; repeated outer validation and design-based confidence intervals have not been completed.
2. Random Forest parameters were fixed for this baseline, not selected through a hyperparameter search.
3. The Random Forest bootstrap is uniform; a survey-probability-aware forest procedure is a possible methodological extension.
4. The outcome is self-reported previous stroke, not future stroke incidence. Predictor diagnoses or behaviours may have occurred after the reported stroke.
5. Low precision remains an issue. Thresholds should ultimately be selected according to a clearly justified research objective, not chosen to make a model look better.
6. The source represents a US survey population and does not validate clinical use or generalisation to Nigeria.

The next step is to assess model stability and subgroup performance, then decide whether a more explicitly survey-aware forest procedure is required before hyperparameter tuning and explainability.
