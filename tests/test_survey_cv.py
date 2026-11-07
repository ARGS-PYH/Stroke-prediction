import numpy as np
import pandas as pd
import pytest

from scripts.survey_cv import assign_survey_folds, fold_adjusted_weights


def sample_design():
    return pd.DataFrame({
        "survey_stratum": [10, 10, 10, 20, 20, 30, 30, 30, 30, 30],
        "survey_psu": list(range(1, 11)),
        "survey_weight": [10.0, 20.0, 30.0, 10.0, 20.0, 1, 2, 3, 4, 5],
    })


def test_fold_assignment_is_reproducible_and_balanced_inside_strata():
    df = sample_design()
    first = assign_survey_folds(df, n_splits=5, random_state=42)
    second = assign_survey_folds(df, n_splits=5, random_state=42)
    assert np.array_equal(first, second)
    assert set(first) == {0, 1, 2, 3, 4}
    for _, positions in df.groupby("survey_stratum").indices.items():
        counts = np.bincount(first[positions], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_fold_assignment_does_not_depend_on_outcome():
    df = sample_design()
    first = assign_survey_folds(df, n_splits=5, random_state=7)
    df["stroke_history"] = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    second = assign_survey_folds(df, n_splits=5, random_state=7)
    assert np.array_equal(first, second)


def test_fold_weights_adjust_for_second_phase_fold_probability():
    df = sample_design()
    folds = assign_survey_folds(df, n_splits=5, random_state=42)
    heldout = 0
    train_w, test_w = fold_adjusted_weights(df, folds, heldout_fold=heldout)
    test_mask = folds == heldout
    train_mask = ~test_mask
    assert np.all(test_w[test_mask] > 0)
    assert np.all(train_w[train_mask] > 0)
    assert np.all(test_w[train_mask] == 0)
    assert np.all(train_w[test_mask] == 0)


def test_invalid_stratum_psu_key_is_rejected():
    df = sample_design()
    df.loc[1, "survey_psu"] = df.loc[0, "survey_psu"]
    with pytest.raises(ValueError):
        assign_survey_folds(df, n_splits=5)
