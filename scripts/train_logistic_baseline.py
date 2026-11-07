#!/usr/bin/env python3
"""Weighted Logistic Regression comparisons with a protected BRFSS holdout.

Outer and inner folds are assigned within _STSTR before filtering unknown
outcomes. The decision threshold is selected from development-only OOF scores
using weighted Youden's J; the final holdout is never used for selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

try:
    from scripts.survey_cv import assign_survey_folds, fold_adjusted_weights
except ModuleNotFoundError:
    from survey_cv import assign_survey_folds, fold_adjusted_weights

TARGET = "stroke_history"
STRING_CATS = ["sex", "diabetes_status", "smoking_status"]
BINARY_CATS = ["hypertension", "chd_or_mi_history", "physical_activity_30d"]
MODEL_FEATURES = {
    "demographic_baseline": ["age_years_capped_at_80", "sex"],
    "primary_eight_features": [
        "age_years_capped_at_80", "sex", "hypertension", "chd_or_mi_history",
        "diabetes_status", "bmi", "smoking_status", "physical_activity_30d",
    ],
    "sensitivity_without_chd_mi": [
        "age_years_capped_at_80", "sex", "hypertension", "diabetes_status",
        "bmi", "smoking_status", "physical_activity_30d",
    ],
}
DESIGN = ["survey_stratum", "survey_psu", "survey_weight"]


def build_pipeline(features: list[str], c_value: float) -> Pipeline:
    transforms = []
    if "age_years_capped_at_80" in features:
        transforms.append((
            "age_spline",
            SplineTransformer(n_knots=5, degree=3, include_bias=False),
            ["age_years_capped_at_80"],
        ))
    if "bmi" in features:
        transforms.append((
            "bmi",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]),
            ["bmi"],
        ))
    strings = [c for c in STRING_CATS if c in features]
    if strings:
        transforms.append((
            "string_categories",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            strings,
        ))
    binaries = [c for c in BINARY_CATS if c in features]
    if binaries:
        transforms.append((
            "binary_categories",
            Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            binaries,
        ))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers=transforms, remainder="drop")),
        ("classifier", LogisticRegression(
            C=c_value, max_iter=2000, solver="lbfgs",
            class_weight=None, random_state=42,
        )),
    ])


def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1], sample_weight=w
    ).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred, sample_weight=w)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=w)),
        "sensitivity_recall": float(recall_score(y, pred, sample_weight=w, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, pred, sample_weight=w, zero_division=0)),
        "f1": float(f1_score(y, pred, sample_weight=w, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)),
        "pr_auc_average_precision": float(average_precision_score(y, p, sample_weight=w)),
        "brier_score": float(brier_score_loss(y, p, sample_weight=w)),
        "confusion_matrix_weighted": [[float(tn), float(fp)], [float(fn), float(tp)]],
    }


def youden_threshold(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, p, sample_weight=w)
    valid = np.isfinite(thresholds)
    if not valid.any():
        return 0.5, 0.0
    j = tpr[valid] - fpr[valid]
    best = int(np.argmax(j))
    return float(thresholds[valid][best]), float(j[best])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "data/processed/brfss2025_stroke_history_analysis.csv.gz"
    ))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--heldout-fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0)
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Prepared dataset not found: {args.input}")
    if args.n_splits < 3 or args.inner_splits < 2 or args.c_value <= 0:
        raise SystemExit("Use >=3 outer folds, >=2 inner folds, and positive C.")

    data = pd.read_csv(args.input, compression="infer", low_memory=False)
    required = set(sum(MODEL_FEATURES.values(), [])) | {TARGET} | set(DESIGN)
    absent = sorted(required - set(data.columns))
    if absent:
        raise SystemExit("Missing columns: " + ", ".join(absent))

    # Fold assignment occurs on all source respondents, before outcome filtering.
    outer = assign_survey_folds(
        data[DESIGN], n_splits=args.n_splits, random_state=args.random_state
    )
    outer_train_w, outer_test_w = fold_adjusted_weights(
        data[DESIGN], outer, heldout_fold=args.heldout_fold
    )
    valid = data[TARGET].notna().to_numpy()
    development_mask = outer != args.heldout_fold
    holdout_mask = outer == args.heldout_fold
    train_mask = development_mask & valid
    test_mask = holdout_mask & valid
    if not train_mask.any() or not test_mask.any():
        raise SystemExit("Empty training or held-out cohort after outcome filtering.")

    dev = data.loc[development_mask].reset_index(drop=True)
    dev_valid = dev[TARGET].notna().to_numpy()
    inner = assign_survey_folds(
        dev[DESIGN], n_splits=args.inner_splits,
        random_state=args.random_state + 100,
    )
    test_y = data.loc[test_mask, TARGET].astype(int).to_numpy()
    test_w = outer_test_w[test_mask]
    holdout = data.loc[test_mask]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": "CDC BRFSS 2025 LLCP",
        "task": "classification of self-reported prior stroke history",
        "algorithm": "weighted Logistic Regression",
        "outer_holdout": {
            "folds": args.n_splits,
            "heldout_fold": args.heldout_fold,
            "random_state": args.random_state,
            "development_valid_n": int(train_mask.sum()),
            "development_yes_n": int(data.loc[train_mask, TARGET].sum()),
            "holdout_valid_n": int(test_mask.sum()),
            "holdout_yes_n": int(test_y.sum()),
        },
        "threshold_selection": {
            "inner_folds": args.inner_splits,
            "method": "weighted Youden's J on development-only out-of-fold predictions",
            "hyperparameter_tuning": "none; C fixed before holdout evaluation",
        },
        "weighting": {
            "base": "_LLCPWT",
            "outer_train_factor": args.n_splits / (args.n_splits - 1),
            "outer_test_factor": args.n_splits,
            "inner_train_factor": args.inner_splits / (args.inner_splits - 1),
            "inner_validation_factor": args.inner_splits,
            "relative_training_weights_rescaled_to_mean_one": True,
        },
        "comparisons": {},
        "limitations": [
            "Initial Logistic Regression comparison; Random Forest has not yet been fitted.",
            "C is fixed and has not been hyperparameter tuned.",
            "Weighted point estimates have no design-based confidence intervals.",
            "Outcome is self-reported prior stroke history, not future stroke incidence.",
            "Diagnoses or behaviours may have occurred after the reported stroke.",
            "This US survey does not validate clinical use or generalisation to Nigeria.",
        ],
    }

    for name, features in MODEL_FEATURES.items():
        oof_p = np.full(len(dev), np.nan)
        oof_w = np.zeros(len(dev))
        for fold in sorted(np.unique(inner)):
            train_w, validation_w = fold_adjusted_weights(
                dev[DESIGN], inner, heldout_fold=int(fold)
            )
            inner_train = (inner != fold) & dev_valid
            inner_validation = (inner == fold) & dev_valid
            model = build_pipeline(features, args.c_value)
            weights = train_w[inner_train]
            weights = weights / weights.mean()
            model.fit(
                dev.loc[inner_train, features],
                dev.loc[inner_train, TARGET].astype(int).to_numpy(),
                classifier__sample_weight=weights,
            )
            oof_p[inner_validation] = model.predict_proba(
                dev.loc[inner_validation, features]
            )[:, 1]
            oof_w[inner_validation] = validation_w[inner_validation]

        oof_mask = dev_valid & np.isfinite(oof_p) & (oof_w > 0)
        threshold, youden_j = youden_threshold(
            dev.loc[oof_mask, TARGET].astype(int).to_numpy(),
            oof_p[oof_mask], oof_w[oof_mask],
        )
        oof_metrics = metrics(
            dev.loc[oof_mask, TARGET].astype(int).to_numpy(),
            oof_p[oof_mask], oof_w[oof_mask], threshold,
        )

        final = build_pipeline(features, args.c_value)
        final_weights = outer_train_w[train_mask]
        final_weights = final_weights / final_weights.mean()
        final.fit(
            data.loc[train_mask, features],
            data.loc[train_mask, TARGET].astype(int).to_numpy(),
            classifier__sample_weight=final_weights,
        )
        holdout_p = final.predict_proba(holdout[features])[:, 1]
        result["comparisons"][name] = {
            "features": features,
            "development_oof": {
                "selected_threshold": threshold,
                "weighted_youden_j": youden_j,
                "weighted_metrics": oof_metrics,
            },
            "untouched_holdout": {
                "weighted_metrics_at_0_5": metrics(test_y, holdout_p, test_w, 0.5),
                "weighted_metrics_at_selected_threshold": metrics(test_y, holdout_p, test_w, threshold),
                "unweighted_metrics_at_selected_threshold": metrics(
                    test_y, holdout_p, np.ones(len(test_y)), threshold
                ),
            },
        }
        joblib.dump(final, args.model_dir / f"{name}.joblib")

    result["majority_class_reference"] = {
        "rule": "always predict no reported prior stroke",
        "weighted_holdout_metrics": metrics(
            test_y, np.zeros(len(test_y)), test_w, 0.5
        ),
    }
    (args.output_dir / "logistic_baseline_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    table = []
    for name, comp in result["comparisons"].items():
        w = comp["untouched_holdout"]["weighted_metrics_at_selected_threshold"]
        u = comp["untouched_holdout"]["unweighted_metrics_at_selected_threshold"]
        table.append({
            "model": name,
            "development_selected_threshold": comp["development_oof"]["selected_threshold"],
            "weighted_recall": w["sensitivity_recall"],
            "weighted_specificity": w["specificity"],
            "weighted_precision": w["precision"],
            "weighted_f1": w["f1"],
            "weighted_roc_auc": w["roc_auc"],
            "weighted_pr_auc": w["pr_auc_average_precision"],
            "unweighted_recall": u["sensitivity_recall"],
            "unweighted_specificity": u["specificity"],
            "unweighted_precision": u["precision"],
            "unweighted_f1": u["f1"],
            "unweighted_roc_auc": u["roc_auc"],
            "unweighted_pr_auc": u["pr_auc_average_precision"],
        })
    pd.DataFrame(table).to_csv(
        args.output_dir / "logistic_baseline_comparison.csv", index=False
    )

    for name, comp in result["comparisons"].items():
        w = comp["untouched_holdout"]["weighted_metrics_at_selected_threshold"]
        print(
            name,
            f"threshold={comp['development_oof']['selected_threshold']:.4f}",
            f"recall={w['sensitivity_recall']:.3f}",
            f"specificity={w['specificity']:.3f}",
            f"PR-AUC={w['pr_auc_average_precision']:.3f}",
            f"ROC-AUC={w['roc_auc']:.3f}",
        )
    print("Aggregate metrics:", args.output_dir / "logistic_baseline_metrics.json")


if __name__ == "__main__":
    main()
