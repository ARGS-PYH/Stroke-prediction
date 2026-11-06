import numpy as np
import pandas as pd

from scripts.prepare_brfss import make_analysis_frame


def sample_chunk():
    return pd.DataFrame({
        "CVDSTRK3": [1, 2, 7, 9, np.nan],
        "_AGE80": [45, 80, 55, 60, 30],
        "SEXVAR": [1, 2, 1, 2, 1],
        "_RFHYPE6": [1, 2, 9, 1, 2],
        "_MICHD": [2, 1, np.nan, 2, 1],
        "DIABETE4": [1, 2, 3, 4, 7],
        "_BMI5": [2500, 9999, np.nan, 3025, 1800],
        "_SMOKER3": [1, 2, 3, 4, 9],
        "EXERANY2": [1, 2, 7, 9, 1],
        "_LLCPWT": [10.0, 20.0, 30.0, 40.0, 50.0],
        "_STSTR": [10001, 10001, 10002, 10002, 10003],
        "_PSU": [2025000001, 2025000002, 2025000003, 2025000004, 2025000005],
    })


def test_only_valid_binary_outcome_rows_are_retained():
    cleaned, stats = make_analysis_frame(sample_chunk())
    assert len(cleaned) == 2
    assert cleaned["stroke_history"].tolist() == [1, 0]
    assert stats["rows_retained"] == 2


def test_age_sex_and_binary_predictors_are_recoded_correctly():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["age_years_capped_at_80"].tolist() == [45, 80]
    assert cleaned["sex"].tolist() == ["Male", "Female"]
    assert cleaned["hypertension"].tolist() == [0, 1]
    assert cleaned["chd_or_mi_history"].tolist() == [0, 1]
    assert cleaned["physical_activity_30d"].tolist() == [1, 0]


def test_diabetes_categories_are_not_collapsed():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["diabetes_status"].tolist() == [
        "Diabetes", "Pregnancy-only diabetes"
    ]


def test_bmi_sentinel_and_missing_values_are_missing():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["bmi"].iloc[0] == 25.0
    assert pd.isna(cleaned["bmi"].iloc[1])


def test_survey_design_fields_are_retained():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["survey_weight"].tolist() == [10.0, 20.0]
    assert cleaned["survey_stratum"].tolist() == [10001, 10001]
    assert cleaned["survey_psu"].tolist() == [2025000001, 2025000002]


def test_unknown_predictor_codes_remain_missing_not_negative():
    raw = sample_chunk().iloc[[0]].copy()
    raw.loc[:, "_RFHYPE6"] = 9
    raw.loc[:, "_MICHD"] = np.nan
    raw.loc[:, "DIABETE4"] = 7
    raw.loc[:, "_BMI5"] = 9999
    raw.loc[:, "_SMOKER3"] = 9
    raw.loc[:, "EXERANY2"] = 7
    cleaned, _ = make_analysis_frame(raw)
    for column in [
        "hypertension", "chd_or_mi_history", "diabetes_status", "bmi",
        "smoking_status", "physical_activity_30d",
    ]:
        assert pd.isna(cleaned.iloc[0][column]), column
