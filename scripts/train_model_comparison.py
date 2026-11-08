#!/usr/bin/env python3
"""Compare survey-weighted Logistic Regression and Random Forest baselines.

The outer and inner folds are assigned within BRFSS strata before filtering
respondents with non-substantive stroke-history responses. All learned
preprocessing is fitted within the respective training fold. The final holdout
is never used to select the decision threshold.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

try:
    from scripts.survey_cv import assign_survey_folds, fold_adjusted_weights
except ModuleNotFoundError:
    from survey_cv import assign_survey_folds, fold_adjusted_weights


TARGET = "stroke_history"
DESIGN_FIELDS = ["survey_stratum", "survey_psu", "survey_weight"]
STRING_CATEGORICALS = ["sex", "diabetes_status", "smoking_status"]
BINARY_CATEGORICALS = [
    "hypertension",
    "chd_or_mi_history",
    "physical_activity_30d",
]
FEATURE_SETS = {
    "demographic_baseline": ["age_years_capped_at_80", "sex"],
    "primary_eight_features": [
        "age_years_capped_at_80",
        "sex",
        "hypertension",
        "chd_or_mi_history",
        "diabetes_status",
        "bmi",
        "smoking_status",
        "physical_activity_30d",
    ],
    "sensitivity_without_chd_mi": [
        "age_years_capped_at_80",
        "sex",
        "hypertension",
        "diabetes_status",
        "bmi",
        "smoking_status",
        "physical_activity_30d",
    ],
}
RANDOM_FOREST_PARAMETERS = {
    "n_estimators": 80,
    "max_depth": 12,
    "min_samples_leaf": 75,
    "max_features": "sqrt",
    "bootstrap": True,
    "max_samples": 0.6,
    "class_weight": None,
    "n_jobs": -1,
    "random_state": 42,
}


def build_preprocessor(features: list[str], model_type: str) -> ColumnTransformer:
    """Build a preprocessing graph that is fitted only on training records."""
    is_logistic = model_type == "logistic_regression"
    transformers = []

    if "age_years_capped_at_80" in features:
        age_transformer = (
            SplineTransformer(n_knots=5, degree=3, include_bias=False)
            if is_logistic
            else StandardScaler()
        )
        transformers.append(
            ("age", age_transformer, ["age_years_capped_at_80"])
        )

    if "bmi" in features:
        bmi_steps = [
            ("impute", SimpleImputer(strategy="median", add_indicator=True))
        ]
        if is_logistic:
            bmi_steps.append(("scale", StandardScaler()))
        transformers.append(
            ("bmi", Pipeline(bmi_steps), ["bmi"])
        )

    strings = [column for column in STRING_CATEGORICALS if column in features]
    if strings:
        transformers.append(
            (
                "string_categories",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="Missing",
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                strings,
            )
        )

    binaries = [
        column for column in BINARY_CATEGORICALS if column in features
    ]
    if binaries:
        transformers.append(
            (
                "binary_categories",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant",
                                fill_value=-1,
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                binaries,
            )
        )

    return ColumnTransformer(transformers, remainder="drop")


def build_model(features: list[str], model_type: str) -> Pipeline:
    """Create one baseline estimator and its training-fold preprocessing."""
    if model_type == "logistic_regression":
        classifier = LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            class_weight=None,
            random_state=42,
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(**RANDOM_FOREST_PARAMETERS)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Pipeline(
        [
            ("preprocess", build_preprocessor(features, model_type)),
            ("classifier", classifier),
        ]
    )


def select_youden_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[float, float]:
    """Select a threshold by weighted Youden's J using development OOF scores."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true,
        probabilities,
        sample_weight=sample_weights,
    )
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5, 0.0

    youden_j = true_positive_rate[finite] - false_positive_rate[finite]
    best = int(np.argmax(youden_j))
    return float(thresholds[finite][best]), float(youden_j[best])


def evaluate(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
        sample_weight=sample_weights,
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions, sample_weight=sample_weights)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true, predictions, sample_weight=sample_weights
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                sample_weight=sample_weights,
                zero_division=0,
            )
        ),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(
            precision_score(
                y_true,
                predictions,
                sample_weight=sample_weights,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                sample_weight=sample_weights,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true, probabilities, sample_weight=sample_weights
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true, probabilities, sample_weight=sample_weights
            )
        ),
        "brier": float(
            brier_score_loss(
                y_true, probabilities, sample_weight=sample_weights
            )
        ),
        "confusion_matrix_weighted": [
            [float(tn), float(fp)],
            [float(fn), float(tp)],
        ],
    }


def run_comparison(
    input_path: Path,
    output_dir: Path,
    model_dir: Path,
    *,
    outer_splits: int = 5,
    inner_splits: int = 4,
    heldout_fold: int = 0,
    random_state: int = 42,
    model_types: tuple[str, ...] = (
        "logistic_regression",
        "random_forest",
    ),
    primary_only: bool = False,
) -> dict:
    """Run cross-validated threshold selection and final holdout comparison."""
    if outer_splits < 3 or inner_splits < 2:
        raise ValueError("Use at least 3 outer folds and 2 inner folds.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(
        input_path,
        compression="infer",
        low_memory=False,
    )
    if len(data) != 356_158:
        raise ValueError(
            "Expected the full 356,158-row file before outcome filtering; "
            f"received {len(data):,} rows."
        )

    required = (
        set(FEATURES_REQUIRED)
        | {TARGET}
        | set(DESIGN_FIELDS)
    )
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(
            "Prepared file is missing required fields: "
            + ", ".join(missing_columns)
        )

    # The fold allocation uses all source records, including records whose
    # outcome will later be excluded from binary model fitting.
    outer_folds = assign_survey_folds(
        data[DESIGN_FIELDS],
        n_splits=outer_splits,
        random_state=random_state,
    )
    outer_train_weights, outer_test_weights = fold_adjusted_weights(
        data[DESIGN_FIELDS],
        outer_folds,
        heldout_fold=heldout_fold,
    )

    valid_outcome = data[TARGET].notna().to_numpy()
    development_mask = outer_folds != heldout_fold
    holdout_mask = outer_folds == heldout_fold
    train_mask = development_mask & valid_outcome
    test_mask = holdout_mask & valid_outcome

    if not train_mask.any() or not test_mask.any():
        raise ValueError("Training or holdout cohort is empty.")

    development = data.loc[development_mask].reset_index(drop=True)
    development_valid = development[TARGET].notna().to_numpy()
    inner_folds = assign_survey_folds(
        development[DESIGN_FIELDS],
        n_splits=inner_splits,
        random_state=random_state + 100,
    )

    y_test = data.loc[test_mask, TARGET].astype(int).to_numpy()
    w_test = outer_test_weights[test_mask]
    holdout = data.loc[test_mask]

    feature_sets = (
        {"primary_eight_features": FEATURE_SETS["primary_eight_features"]}
        if primary_only
        else FEATURE_SETS
    )

    results = {
        "dataset": "CDC BRFSS 2025 LLCP",
        "task": "classification of self-reported prior stroke history",
        "n_source": int(len(data)),
        "n_valid_outcome": int(valid_outcome.sum()),
        "n_outcome_missing": int((~valid_outcome).sum()),
        "outer_holdout": {
            "n_splits": outer_splits,
            "heldout_fold": heldout_fold,
            "random_state": random_state,
            "development_n": int(train_mask.sum()),
            "development_yes": int(
                data.loc[train_mask, TARGET].sum()
            ),
            "holdout_n": int(test_mask.sum()),
            "holdout_yes": int(y_test.sum()),
        },
        "threshold_selection": {
            "inner_folds": inner_splits,
            "method": (
                "weighted Youden's J on development-only "
                "out-of-fold predictions"
            ),
            "holdout_used_for_threshold_selection": False,
        },
        "fold_weight_adjustment": {
            "outer_train_factor": outer_splits / (outer_splits - 1),
            "outer_test_factor": outer_splits,
            "inner_train_factor": inner_splits / (inner_splits - 1),
            "inner_validation_factor": inner_splits,
        },
        "random_forest_parameters": RANDOM_FOREST_PARAMETERS,
        "models": {},
        "limitations": [
            "One fixed outer holdout; results are not repeated-CV estimates.",
            "No design-based confidence intervals for predictive performance.",
            "The target is self-reported prior stroke history, not future stroke incidence.",
            "Some diagnoses or behaviours may have occurred after reported stroke.",
            "BRFSS is a US survey; results do not validate use in Nigeria.",
            (
                "The scikit-learn forest receives fold-adjusted survey sample "
                "weights, but its bootstrap draws records uniformly; this is "
                "not a fully design-based Random Forest procedure."
            ),
        ],
    }

    y_majority = np.zeros(len(y_test), dtype=int)

    for model_type in model_types:
        if model_type not in {"logistic_regression", "random_forest"}:
            raise ValueError(f"Unsupported model type: {model_type}")

        for feature_name, features in feature_sets.items():
            start_time = time.time()
            oof_probabilities = np.full(len(development), np.nan)
            oof_weights = np.zeros(len(development))

            for fold in sorted(np.unique(inner_folds)):
                train_fold_weights, validation_fold_weights = (
                    fold_adjusted_weights(
                        development[DESIGN_FIELDS],
                        inner_folds,
                        heldout_fold=int(fold),
                    )
                )
                inner_train = (
                    (inner_folds != fold) & development_valid
                )
                inner_validation = (
                    (inner_folds == fold) & development_valid
                )

                estimator = build_model(features, model_type)
                fit_weights = train_fold_weights[inner_train]
                fit_weights = fit_weights / fit_weights.mean()

                estimator.fit(
                    development.loc[inner_train, features],
                    development.loc[
                        inner_train, TARGET
                    ].astype(int).to_numpy(),
                    classifier__sample_weight=fit_weights,
                )
                oof_probabilities[inner_validation] = (
                    estimator.predict_proba(
                        development.loc[inner_validation, features]
                    )[:, 1]
                )
                oof_weights[inner_validation] = (
                    validation_fold_weights[inner_validation]
                )
                print(
                    f"{model_type}/{feature_name}: inner fold "
                    f"{int(fold) + 1}/{len(np.unique(inner_folds))} complete",
                    flush=True,
                )

            oof_mask = (
                development_valid
                & np.isfinite(oof_probabilities)
                & (oof_weights > 0)
            )
            y_oof = (
                development.loc[oof_mask, TARGET].astype(int).to_numpy()
            )
            threshold, youden_j = select_youden_threshold(
                y_oof,
                oof_probabilities[oof_mask],
                oof_weights[oof_mask],
            )

            final_model = build_model(features, model_type)
            final_weights = outer_train_weights[train_mask]
            final_weights = final_weights / final_weights.mean()
            final_model.fit(
                data.loc[train_mask, features],
                data.loc[train_mask, TARGET].astype(int).to_numpy(),
                classifier__sample_weight=final_weights,
            )
            holdout_probabilities = final_model.predict_proba(
                holdout[features]
            )[:, 1]

            selected_weighted = evaluate(
                y_test, holdout_probabilities, w_test, threshold
            )
            results["models"].setdefault(model_type, {})[feature_name] = {
                "features": features,
                "training_seconds": round(time.time() - start_time, 1),
                "development_oof": {
                    "threshold": threshold,
                    "weighted_youden_j": youden_j,
                    "metrics": evaluate(
                        y_oof,
                        oof_probabilities[oof_mask],
                        oof_weights[oof_mask],
                        threshold,
                    ),
                },
                "holdout": {
                    "weighted_at_0_5": evaluate(
                        y_test,
                        holdout_probabilities,
                        w_test,
                        0.5,
                    ),
                    "weighted_at_selected_threshold": selected_weighted,
                    "unweighted_at_selected_threshold": evaluate(
                        y_test,
                        holdout_probabilities,
                        np.ones(len(y_test)),
                        threshold,
                    ),
                },
            }

            joblib.dump(
                final_model,
                model_dir / f"{model_type}_{feature_name}.joblib",
            )
            print(
                f"{model_type}/{feature_name}: "
                f"ROC-AUC={selected_weighted['roc_auc']:.4f}; "
                f"PR-AUC={selected_weighted['pr_auc']:.4f}; "
                f"elapsed={round(time.time() - start_time, 1)}s",
                flush=True,
            )

    results["majority_class_reference"] = {
        "rule": "always predict no reported prior stroke",
        "weighted_holdout": evaluate(
            y_test, y_majority, w_test, 0.5
        ),
    }

    json_path = output_dir / "model_comparison_metrics.json"
    csv_path = output_dir / "model_comparison_metrics.csv"
    json_path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    rows = []
    for model_type, models in results["models"].items():
        for feature_name, comparison in models.items():
            weighted = comparison["holdout"][
                "weighted_at_selected_threshold"
            ]
            unweighted = comparison["holdout"][
                "unweighted_at_selected_threshold"
            ]
            rows.append(
                {
                    "model": model_type,
                    "feature_set": feature_name,
                    "selected_threshold": comparison[
                        "development_oof"
                    ]["threshold"],
                    **{
                        f"weighted_{key}": weighted[key]
                        for key in (
                            "recall",
                            "specificity",
                            "precision",
                            "f1",
                            "roc_auc",
                            "pr_auc",
                        )
                    },
                    **{
                        f"unweighted_{key}": unweighted[key]
                        for key in (
                            "recall",
                            "specificity",
                            "precision",
                            "f1",
                            "roc_auc",
                            "pr_auc",
                        )
                    },
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"Aggregate metrics: {json_path}")
    print(f"Comparison table: {csv_path}")
    return results


FEATURES_REQUIRED = {
    column
    for feature_list in FEATURE_SETS.values()
    for column in feature_list
}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=repository_root
        / "data/processed/brfss2025_stroke_history_analysis.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository_root / "reports/generated",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=repository_root / "models",
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()

    run_comparison(
        args.input,
        args.output_dir,
        args.model_dir,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        heldout_fold=args.heldout_fold,
        random_state=args.random_state,
        primary_only=args.primary_only,
    )


if __name__ == "__main__":
    main()
