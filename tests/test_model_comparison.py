import numpy as np
import pandas as pd

from scripts.train_model_comparison import (
    FEATURE_SETS,
    build_model,
    evaluate,
    select_youden_threshold,
)


def toy_data(n=120):
    rng = np.random.default_rng(23)
    age = rng.integers(18, 81, size=n)
    x = pd.DataFrame({
        "age_years_capped_at_80": age,
        "sex": np.where(np.arange(n) % 2 == 0, "Female", "Male"),
        "hypertension": rng.choice([0.0, 1.0, np.nan], size=n),
        "chd_or_mi_history": rng.choice([0.0, 1.0, np.nan], size=n),
        "diabetes_status": rng.choice(
            ["Diabetes", "No diabetes", "Prediabetes or borderline diabetes", None],
            size=n,
        ),
        "bmi": rng.choice([22.5, 27.0, 33.0, np.nan], size=n),
        "smoking_status": rng.choice(
            ["Never smoked", "Former smoker", "Current smoker - every day", None],
            size=n,
        ),
        "physical_activity_30d": rng.choice([0.0, 1.0, np.nan], size=n),
    })
    y = ((age >= 60) | (np.arange(n) % 9 == 0)).astype(int)
    return x, y


def test_both_model_pipelines_fit_with_missing_values():
    x, y = toy_data()
    weights = np.ones(len(y))
    features = FEATURE_SETS["primary_eight_features"]

    for model_type in ("logistic_regression", "random_forest"):
        model = build_model(features, model_type)
        model.fit(x, y, classifier__sample_weight=weights)
        probabilities = model.predict_proba(x)[:, 1]
        assert probabilities.shape == (len(y),)
        assert np.isfinite(probabilities).all()
        assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_threshold_selection_uses_weighted_youden_j():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    probabilities = np.array([0.01, 0.02, 0.20, 0.25, 0.30, 0.35, 0.80, 0.90])
    threshold, score = select_youden_threshold(
        y, probabilities, np.ones(len(y))
    )
    assert 0 < threshold < 1
    assert score > 0


def test_metrics_include_class_sensitive_measures():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.7, 0.8])
    result = evaluate(y, probabilities, np.ones(4), threshold=0.5)
    assert result["recall"] == 1.0
    assert result["specificity"] == 1.0
    assert result["pr_auc"] > 0.9
