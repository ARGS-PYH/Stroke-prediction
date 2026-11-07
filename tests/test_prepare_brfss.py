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


def test_all_source_rows_are_preserved_and_nonbinary_outcomes_stay_missing():
    cleaned, stats = make_analysis_frame(sample_chunk())
    assert len(cleaned) == 5
    assert cleaned["stroke_history"].iloc[:2].tolist() == [1, 0]
    assert cleaned["stroke_history"].iloc[2:].isna().all()
    assert stats["rows_written"] == 5
    assert stats["valid_binary_outcome_n"] == 2
    assert stats["valid_outcome_counts"] == {1: 1, 0: 1}
    assert stats["excluded_from_binary_ml_counts"] == {
        "7": 1, "9": 1, "blank_or_missing": 1
    }


def test_age_sex_and_binary_predictors_are_recoded_correctly():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["age_years_capped_at_80"].tolist() == [45, 80, 55, 60, 30]
    assert cleaned["sex"].tolist() == ["Male", "Female", "Male", "Female", "Male"]
    assert cleaned["hypertension"].iloc[:2].tolist() == [0, 1]
    assert pd.isna(cleaned["hypertension"].iloc[2])
    assert cleaned["physical_activity_30d"].iloc[:2].tolist() == [1, 0]
    assert pd.isna(cleaned["physical_activity_30d"].iloc[2])


def test_diabetes_categories_are_not_collapsed():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["diabetes_status"].tolist()[:4] == [
        "Diabetes", "Pregnancy-only diabetes", "No diabetes",
        "Prediabetes or borderline diabetes",
    ]
    assert pd.isna(cleaned["diabetes_status"].iloc[4])


def test_bmi_sentinel_and_missing_values_are_missing():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["bmi"].iloc[0] == 25.0
    assert pd.isna(cleaned["bmi"].iloc[1])
    assert pd.isna(cleaned["bmi"].iloc[2])


def test_survey_design_fields_are_retained_for_all_rows():
    cleaned, _ = make_analysis_frame(sample_chunk())
    assert cleaned["survey_weight"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]
    assert cleaned["survey_stratum"].tolist() == [10001, 10001, 10002, 10002, 10003]
    assert cleaned["survey_psu"].tolist() == [
        2025000001, 2025000002, 2025000003, 2025000004, 2025000005
    ]


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
