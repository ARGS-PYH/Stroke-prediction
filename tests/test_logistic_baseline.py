import numpy as np
import pandas as pd

from scripts.train_logistic_baseline import (
    MODEL_FEATURES, build_pipeline, metrics, youden_threshold
)


def toy_data(n=80):
    rng = np.random.default_rng(17)
    age = rng.integers(18, 81, size=n)
    x = pd.DataFrame({
        "age_years_capped_at_80": age,
        "sex": np.where(np.arange(n) % 2 == 0, "Female", "Male"),
        "hypertension": np.where(age >= 55, 1.0, 0.0),
        "chd_or_mi_history": np.where(age >= 65, 1.0, 0.0),
        "diabetes_status": np.where(np.arange(n) % 4 == 0, "Diabetes", "No diabetes"),
        "bmi": rng.normal(27, 4, size=n),
        "smoking_status": np.where(np.arange(n) % 3 == 0, "Former smoker", "Never smoked"),
        "physical_activity_30d": (np.arange(n) % 2).astype(float),
    })
    x.loc[::13, "hypertension"] = np.nan
    x.loc[::17, "chd_or_mi_history"] = np.nan
    x.loc[::19, "diabetes_status"] = None
    x.loc[::11, "bmi"] = np.nan
    x.loc[::23, "smoking_status"] = None
    x.loc[::29, "physical_activity_30d"] = np.nan
    y = ((age >= 60) | (np.arange(n) % 11 == 0)).astype(int)
    return x, y


def test_primary_logistic_pipeline_fits_with_missing_inputs():
    x, y = toy_data()
    features = MODEL_FEATURES["primary_eight_features"]
    model = build_pipeline(features, c_value=1.0)
    model.fit(x[features], y, classifier__sample_weight=np.ones(len(y)))
    p = model.predict_proba(x[features])[:, 1]
    assert p.shape == (len(y),)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_youden_threshold_uses_only_supplied_development_predictions():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.01, 0.02, 0.20, 0.25, 0.30, 0.35, 0.80, 0.90])
    threshold, score = youden_threshold(y, p, np.ones(len(y)))
    assert 0 < threshold < 1
    assert score > 0


def test_metrics_report_class_sensitive_measures():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.7, 0.8])
    m = metrics(y, p, np.ones(4), threshold=0.5)
    assert m["sensitivity_recall"] == 1.0
    assert m["specificity"] == 1.0
    assert m["pr_auc_average_precision"] > 0.9
